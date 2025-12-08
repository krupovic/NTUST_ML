# Revised logic based on Kaggle solution order
from project_imports import *
from data_understanding import load_and_preprocess_data

def investigate_features(df, name):
    for colum in df:
        unique_values = np.unique(df[colum])
        nr_values = len(unique_values)
        if nr_values < 10:
            print(f"The number of unique values for features {colum} : {nr_values} --- {unique_values}")
        else:
            print(f"The number of unique values for features {colum} : {nr_values}")

def decode_time_signature(df):
    df["TIMESTAMP"] = [float(time) for time in df["TIMESTAMP"]]
    df["data_time"] = [datetime.datetime.fromtimestamp(time, datetime.timezone.utc) for time in df["TIMESTAMP"]]
    return df

def create_time_features(df):
    df["year"] = pd.to_datetime(df["data_time"]).dt.year
    df["month"] = pd.to_datetime(df["data_time"]).dt.month
    df["week"] = pd.to_datetime(df["data_time"]).dt.isocalendar().week
    df["day"] = pd.to_datetime(df["data_time"]).dt.day
    df["hour"] = pd.to_datetime(df["data_time"]).dt.hour
    df["min"] = pd.to_datetime(df["data_time"]).dt.minute
    df["weekday"] = pd.to_datetime(df["data_time"]).dt.weekday
    return df

def encode_call_type(df):
    encoder = OneHotEncoder(handle_unknown='ignore')
    encoder_df = pd.DataFrame(encoder.fit_transform(df[['CALL_TYPE']]).toarray())
    df = df.join(encoder_df)
    df.rename(columns={0:'call_type_a', 1:'call_type_b',2:'call_type_c'}, inplace=True)
    return df

def extract_first_last_coords(df):
    # 1st lon
    lists_1st_lon = []
    for i in range(len(df["POLYLINE"])):
        if df["POLYLINE"][i] == '[]':
            k=0
            lists_1st_lon.append(k)
        else:
            k = re.sub(r"[[|[|]|]|]]", "", str(df["POLYLINE"][i])).split(",")[0]
            lists_1st_lon.append(k)
    df["lon_1st"] = lists_1st_lon

    # 1st lat
    lists_1st_lat = []
    for i in range(len(df["POLYLINE"])):
        if df["POLYLINE"][i] == '[]':
            k=0
            lists_1st_lat.append(k)
        else:
            k = re.sub(r"[[|[|]|]|]]", "", str(df["POLYLINE"][i])).split(",")[1]
            lists_1st_lat.append(k)
    df["lat_1st"] = lists_1st_lat

    # last lon
    lists_last_lon = []
    for i in range(len(df["POLYLINE"])):
        if df["POLYLINE"][i] == '[]':
            k=0
            lists_last_lon.append(k)
        else:
            k = re.sub(r"[[|[|]|]|]]", "", str(df["POLYLINE"][i])).split(",")[-2]
            lists_last_lon.append(k)
    df["lon_last"] = lists_last_lon

    # last lat
    lists_last_lat = []
    for i in range(len(df["POLYLINE"])):
        if df["POLYLINE"][i] == '[]':
            k=0
            lists_last_lat.append(k)
        else:
            k = re.sub(r"[[|[|]|]|]]", "", str(df["POLYLINE"][i])).split(",")[-1]
            lists_last_lat.append(k)
    df["lat_last"] = lists_last_lat
    return df

def clean_and_cast(df):
    df = df.query("lon_last != 0")
    df["lon_1st"] = [float(k) for k in df["lon_1st"]]
    df["lat_1st"] = [float(k) for k in df["lat_1st"]]
    df["lon_last"] = [float(k) for k in df["lon_last"]]
    df["lat_last"] = [float(k) for k in df["lat_last"]]
    df['call_type_a']= [int(k) for k in df["call_type_a"]]
    df['call_type_b'] =[int(k) for k in df["call_type_b"]]
    df['call_type_c']= [int(k) for k in df["call_type_c"]]
    return df

def main():
    train, test, sample, location = load_and_preprocess_data()

    print("# INVESTIGATING ALL ELEMENTS WITHIN EACH FEATURE IN TRAINING DATA")
    investigate_features(train, "train")
    print("# INVESTIGATING ALL ELEMENTS WITHIN EACH FEATURE IN TESTING DATA")
    investigate_features(test, "test")

    print("# DECODING TIME SIGNATURE TRAIN DATA")
    train = decode_time_signature(train)
    print("# DECODING TIME SIGNATURE TEST DATA")
    test = decode_time_signature(test)

    print(train["data_time"].value_counts())
    print(test["data_time"].value_counts())

    print("# CREATING TIME BASED FEATURES TRAINING DATA")
    train = create_time_features(train)
    print("# CREATING TIME BASED FEATURES TESTING DATA")
    test = create_time_features(test)

    print("# ENCODING CALL TYPE FOR TRAINING DATA TO PREPARE FOR MODELING")
    final_train = encode_call_type(train)
    print("# ENCODING CALL TYPE FOR TEST DATA")
    final_test = encode_call_type(test)

    print("# EXTRACTING 1st/LAST LATITUDE/LONGITUDE FOR TRAINING DATA")
    final_train = extract_first_last_coords(final_train)
    print("# EXTRACTING 1st/LAST LATITUDE/LONGITUDE FOR TESTING DATA")
    final_test = extract_first_last_coords(final_test)

    print("# DELETE LON & LAT HAVE '0' IN TRAINING DATA")
    train = clean_and_cast(final_train)
    print("# DELETE LON & LAT HAVE '0' IN TESTING DATA")
    test = clean_and_cast(final_test)

    print("Final train shape:", train.shape)
    print("Final test shape:", test.shape)

    # Save processed tables with clear names
    train.to_csv("train_processed.csv", index=False)
    test.to_csv("test_processed.csv", index=False)

if __name__ == "__main__":
    main()
