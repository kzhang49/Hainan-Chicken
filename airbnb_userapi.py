import pandas as pd
import numpy as np
import pickle
import sklearn
import scipy
import copy

#*******************In/Out function***********************
def open_file(csv_file):
    reader = pd.read_csv(csv_file)
    return reader

def output_file(dataframe, string):
    dataframe.to_csv(string, index=False)

#*******************Clean data funcion********************
def clean_data(dataframe, dataframe_listing):
    heating = []
    wifi = []
    pets = []
    Washer_dryer = []
    gym = []
    # keep all avaliable dates records and count avaliable nights
    number = dataframe[dataframe.available.str[0:] != "f"].groupby('listing_id').listing_id.count()

    # generate new dataframe about sum of avaliable nights
    df_nights = pd.DataFrame({'id': number.index, 'sum_of_nights': number.values})

    # Create month column
    dataframe['date'] = pd.to_datetime(dataframe['date'], errors='coerce')
    dataframe['month'] = dataframe['date'].dt.month

    # replace data type of price to float
    dfa = replace_price_to_float(dataframe)

    # Calculate mean price for each month
    df_mean_price = calculate_mean_price(dfa)

    # Change the format of calendar dataframe
    lists_id = []
    for i in df_mean_price['listing_id']:
        if i not in lists_id:
            lists_id.append(i)
    df_id = pd.DataFrame(lists_id, columns=['id'])
    df_calendar_final = recursive_for_calendar(df_mean_price, df_id, 1)

    # merge sum_of_nights with listing table
    dataframe_listing = pd.merge(dataframe_listing, df_nights, on='id', how='left')
    dataframe_listing = dataframe_listing.fillna(0)

    # delete unimportance features
    dataframe_listing = dataframe_listing[dataframe_listing.columns[[0, 1, 4, 5, 6, 7, 9, 17, 25, 26, 37, 39, 41, 48, 49,
                           50, 51, 52, 53, 54, 55, 56, 58, 60, 89, 91, 96]]]

    # make sure beds > 0
    df_listing = dataframe_listing[dataframe_listing.beds != 0]

    # replace data type of price to float
    df_listing = replace_price_to_float(df_listing)

    # Calculate sum of amenities and Categorize amenities
    sum_of_amenities = sum_amenities(df_listing)
    df_listing['sum_of_amenities'] = sum_of_amenities
    df_listing['heating'] = split(df_listing, 'Heating', heating)
    df_listing['wifi'] = split(df_listing, '"Wireless Internet"', wifi)
    df_listing['pets_allowed'] = split(df_listing, '"Pets allowed"', pets)
    df_listing['Washer_dryer'] = split(df_listing, 'Washer,Dryer', Washer_dryer)
    df_listing['gym'] = split(df_listing, 'Gym', gym)

    # Reorder listing file
    df_listing = df_listing[['id','listing_url','name','summary','space','description','neighborhood_overview','picture_url',
                           'host_response_time','host_response_rate','street','neighbourhood_cleansed','city','latitude',
                           'longitude','is_location_exact','property_type','room_type','accommodates','bathrooms','bedrooms',
                           'beds','amenities','heating','wifi','pets_allowed','Washer_dryer','gym','sum_of_amenities','price',
                           'instant_bookable','cancellation_policy','sum_of_nights']]


    # Out put cleansed calendar csv
    output_file(df_calendar_final, "data/cleansed_calendar.csv")
    output_file(df_listing, "data/cleansed_listings.csv")

#******************Encoding data (prepare for machine learning)*************************
def change_price_to_cat(df):
    price_cat = []
    for row in range(len(df['price'])):
        if df['price'][row] <= 250 and 0 <= df['price'][row]:
            price_cat.append('$0-250')
        elif df['price'][row] <= 500 and 250 < df['price'][row]:
            price_cat.append('$250-500')
        elif 500 < df['price'][row]:
            price_cat.append('$500-4000')
    return price_cat

def change_sum_nights_to_cat(df):
    sum_of_nights_cat = []
    for row in range(len(df['sum_of_nights'])):
        if df['sum_of_nights'][row] <= 365 and 300 < df['sum_of_nights'][row]:
            sum_of_nights_cat.append(0)
        elif df['sum_of_nights'][row] <= 300 and 200 < df['sum_of_nights'][row]:
            sum_of_nights_cat.append(1)
        elif df['sum_of_nights'][row] <= 200 and 100 < df['sum_of_nights'][row]:
            sum_of_nights_cat.append(2)
        elif df['sum_of_nights'][row] <= 100 and 0 <= df['sum_of_nights'][row]:
            sum_of_nights_cat.append(3)
    return sum_of_nights_cat

def change_response_time_to_cat(df):
    host_response_time_cat = []
    for row in range(len(df['host_response_time'])):
        if 'within an hour' in df['host_response_time'][row]:
            host_response_time_cat.append(1)
        elif 'within a few hours' in df['host_response_time'][row]:
            host_response_time_cat.append(2)
        elif 'within a day' in df['host_response_time'][row]:
            host_response_time_cat.append(3)
        elif 'a few days or more' in df['host_response_time'][row]:
            host_response_time_cat.append(4)
        else:
            host_response_time_cat.append(0)
    return host_response_time_cat

def change_response_rate_to_cat(df):
    responserate_cat = []
    df['host_response_rate'] = df['host_response_rate'].str.replace('%', '')
    df['host_response_rate'] = df['host_response_rate'].astype('float64') / 100
    for row in range(len(df['host_response_rate'])):
        if df['host_response_rate'][row] <= 80 and 0 <= df['host_response_rate'][row]:
            responserate_cat.append(0)
        elif df['host_response_rate'][row] <= 90 and 80 < df['host_response_rate'][row]:
            responserate_cat.append(1)
        elif df['host_response_rate'][row] <= 100 and 90 < df['host_response_rate'][row]:
            responserate_cat.append(2)
    return responserate_cat

def encoding_data(df):
    '''using find and replace or label encoding to encode cleansed data: change object to float or int'''
    df['price'] = change_price_to_cat(df)
    df["sum_of_nights"] = change_sum_nights_to_cat(df)
    df["host_response_time"] = change_response_time_to_cat(df)
    df['sum_of_nights'] = df['sum_of_nights'].astype('float64')
    df["host_response_rate"] = change_response_rate_to_cat(df)
    obj = df.select_dtypes(include=['object']).copy()
    df["property_type"] = obj_encoding_label_encoding(obj, "property_type")
    df["neighbourhood_cleansed"] = obj_encoding_label_encoding(obj, "neighbourhood_cleansed")
    df["cancellation_policy"] = obj_encoding_label_encoding(obj, "cancellation_policy")
    obj_i = obj_encoding_find_replace(obj)
    df["room_type"] = obj_i["room_type"]
    df["price"] = obj_i["price"]
    df["is_location_exact"] = obj_i["is_location_exact"]
    df["instant_bookable"] = obj_i["instant_bookable"]

    return df


#******************Feature Engeneering***********************************
def eliminating_unimportance_feature(df):
    df_final = df.drop(['wifi', 'heating', 'is_location_exact'], axis = 1)
    return df_final

def one_hot_encoding(df):
    df_1hot = df.copy()
    print(df_1hot.dtypes)
    X_dummies = pd.get_dummies(df_1hot, columns=df_1hot.columns, drop_first=True)  # one hot encoding
    print(X_dummies.shape)
    print(X_dummies.head())
    return X_dummies

#******************Helper function************************
def calculate_mean_price(df):
    clear_nums = {'t': 1, 'f': 0}
    df.replace(clear_nums, inplace=True)
    mul = []
    for i, row in df.iterrows():
        mul.append(row[2] * row[3])
    df['summ'] = mul
    summ = df.groupby(['listing_id', 'month']).summ.sum()
    ava_count = df.groupby(['listing_id', 'month']).available.sum()
    monthly_price = summ / ava_count
    dfb = monthly_price.reset_index().fillna(0)
    return dfb

def recursive_for_calendar(df_mean_price, df, flag):
    price_list = []
    if flag == 13:
        return df
    else:
        for i, row in df_mean_price.iterrows():
            if int(row[1]) == flag:
                price_list.append(row[2])
        df[str(flag)] = price_list
        flag += 1
        return recursive_for_calendar(df_mean_price,df, flag)

def replace_price_to_float(df):
    dfa = df.fillna('$0.00')
    dfa['price'] = dfa['price'].str.replace('$', '')
    dfa['price'] = dfa['price'].str.replace(',', '')
    dfa['price'] = dfa['price'].astype('float64')
    return dfa


def split_amenities(each_loop, string, l):
    if string in each_loop:
        l.append(1)
    else:
        l.append(0)


def split(df, string:str, l:list) -> list:
    for i in df['amenities']:
        split_amenities(i, string, l)
    return l


def sum_amenities(df) -> list:
    length = []
    for i in df['amenities']:
        s = i.replace("{", "").replace("}", "").split(",")
        length.append(len(s))
    return length

def obj_encoding_find_replace(obj) :
    cleanup_nums = {"room_type": {"Entire home/apt": 0, "Private room": 1,
                                  "Shared room": 2},
                    "instant_bookable": {"f": 0, "t": 1},
                    "price": {"$0-250": 0, "$250-500": 1, "$500-4000": 2},
                    "is_location_exact": {"f": 0, "t": 1}}
    obj.replace(cleanup_nums, inplace=True)
    return obj

def obj_encoding_label_encoding(obj, string) -> list:
    obj[string] = obj[string].astype('category')
    new_string = string + "_cat"
    obj[new_string] = obj[string].cat.codes
    return obj[new_string]

def convert_string_to_int(string):
    return int(string)

#**************API**********************


def main():
    pkl_filename = "pickle_model.pkl"
    listings = open_file('data/listings.csv')
    calendar = open_file('data/calendar.csv')
    clean_data(calendar, listings)
    listings = pd.read_csv('data/cleansed_listings.csv')
    new_listings = listings.drop(['id', 'city', 'listing_url', 'picture_url', 'neighborhood_overview',
                          'summary', 'space', 'description', 'street', 'name', 'amenities', 'latitude', 'longitude'],
                         axis=1)
    encoded_df = encoding_data(new_listings)
    df_n = encoded_df.drop(['price'], axis = 1)
    df_final = eliminating_unimportance_feature(df_n)
    X_dummies = one_hot_encoding(df_final)
    file = open(pkl_filename, 'rb')
    model = pickle.load(file)
    print("Loaded model :: ", model)


if __name__ == "__main__":
    main()