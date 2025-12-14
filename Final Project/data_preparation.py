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
    for col in ['call_type_a', 'call_type_b', 'call_type_c']:
        df[col] = df[col].astype(int)
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

    # Drop high-missing origin identifiers no longer needed for modeling
    drop_cols = ["ORIGIN_CALL", "ORIGIN_STAND"]
    final_train = final_train.drop(columns=drop_cols, errors="ignore")
    final_test = final_test.drop(columns=drop_cols, errors="ignore")

    print("Final train shape:", final_train.shape)
    print("Final test shape:", final_test.shape)

    # Save processed tables with clear names
    final_train.to_csv("train_processed.csv", index=False)
    final_test.to_csv("test_processed.csv", index=False)

if __name__ == "__main__":
    main()
