import os
import datetime
import pandas as pd
import numpy as np
from geopy import distance
from geopy.point import Point
from scipy import stats
from sklearn.model_selection import train_test_split
import math
from data_evaluation import *
from Feature_Engineering import *

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

def abs_path(*paths):
    return os.path.join(BASE_DIR, *paths)

def remove_outliers(df, target, feature, method='zscore', factor=3):
    if target not in df.columns or feature not in df.columns:
        raise ValueError(f"Columns '{target}' or '{feature}' not found in DataFrame")

    if method == 'zscore':
        z_scores = np.abs(stats.zscore(df[[target, feature]]))
        filtered_entries = (z_scores < factor).all(axis=1)
    elif method == 'iqr':
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        filtered_entries = (df[feature] >= lower_bound) & (df[feature] <= upper_bound)
    else:
        raise ValueError("Method must be either 'zscore' or 'iqr'")

    return df[filtered_entries].copy()



def delete_infrequent_categories(df, categorical_features, threshold=5):
    for feature in categorical_features:
        category_counts = df[feature].value_counts()
        infrequent_categories = category_counts[category_counts < threshold].index
        df = df[~df[feature].isin(infrequent_categories)]
    return df



def marge_spilite(df_train, df_test):
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
    df_combined = df_combined.sample(frac=1, random_state=7).reset_index(drop=True)

    train_size = 0.90
    val_size = 0.05
    test_size = 0.05

    df_train_val, df_test = train_test_split(df_combined, test_size=test_size, random_state=42)
    relative_val_size = val_size / (train_size + val_size)
    df_train, df_val = train_test_split(df_train_val, test_size=relative_val_size, random_state=42)

    print(f'Train set size: {len(df_train)}')
    print(f'Validation set size: {len(df_val)}')
    print(f'Test set size: {len(df_test)}')
    return df_train, df_val, df_test



if __name__ == "__main__":
    train = pd.read_csv(abs_path("Full_data", "train.csv"))
    val = pd.read_csv(abs_path("Full_data", "val.csv"))

    train['trip_duration'] = np.log1p(train['trip_duration'])
    val['trip_duration'] = np.log1p(val['trip_duration'])

    # some feature engineering
    for df in (train, val):
        df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
        df['distance_haversine'] = df.apply(haversine_distance, axis=1)
        df['direction'] = df.apply(calculate_direction, axis=1)
        df['distance_manhattan'] = df.apply(manhattan_distance, axis=1)
        bins = [0, 2, 5, 8, 11, 12]
        labels = ['0', '1', '2', '3', '4']
        df["pickup_hour"] = df["pickup_datetime"].dt.hour
        df["pickup_day"] = df["pickup_datetime"].dt.day
        df["pickup_dayofweek"] = df["pickup_datetime"].dt.dayofweek
        df["pickup_month"] = df["pickup_datetime"].dt.month
        df['pickup_Season'] = pd.cut(df["pickup_month"], bins=bins, labels=labels, right=False, ordered=False)
        df.drop(columns=['id', 'pickup_datetime'], inplace=True)
        df.reset_index(drop=True, inplace=True)

    categorical_features = [
        'vendor_id', 'passenger_count', "pickup_hour", "pickup_day",
        "pickup_dayofweek", "pickup_month", "pickup_Season", 'store_and_fwd_flag'
    ]

    train = delete_infrequent_categories(train, categorical_features, threshold=5)

    directory = abs_path("saving_data", "1")
    os.makedirs(directory, exist_ok=True)
    train.to_csv(abs_path(directory, "train.csv"), index=False)
    val.to_csv(abs_path(directory, "val.csv"), index=False)

    print(f"Data has been saved successfully inside {directory}")
