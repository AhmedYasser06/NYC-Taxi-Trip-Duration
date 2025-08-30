import os
import numpy as np
import pandas as pd
from data_preparing import *
from Feature_Engineering import *
from data_training import *
from data_preprocessing import *
from data_evaluation import *
from loading_model import load_model

# Base project directory (folder where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def abs_path(*paths):
    """Helper to build absolute paths inside the project."""
    return os.path.join(BASE_DIR, *paths)

def test_preparing(data: pd.DataFrame):
    data['trip_duration'] = np.log1p(data['trip_duration'])
    data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"]) 

    data['distance_haversine'] = data.apply(haversine_distance, axis=1)
    data['direction'] = data.apply(calculate_direction, axis=1)
    data['distance_manhattan'] = data.apply(manhattan_distance, axis=1)

    bins = [0, 2, 5, 8, 11, 12]
    labels = ['0', '1', '2', '3', '4']  

    data["pickup_hour"] = data["pickup_datetime"].dt.hour
    data["pickup_day"] = data["pickup_datetime"].dt.day
    data["pickup_dayofweek"] = data["pickup_datetime"].dt.dayofweek
    data["pickup_month"] = data["pickup_datetime"].dt.month
    data['pickup_Season'] = pd.cut(data["pickup_month"], bins=bins, labels=labels, right=False, ordered=False) 

    data.drop(columns=['id', 'pickup_datetime'], inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data 


if __name__ == "__main__":

    train_path = abs_path("Full_data", "train.csv")
    val_path   = abs_path("Full_data", "val.csv")
    test_path  = abs_path("Full_data", "test.csv")
    model_path = abs_path("models", "latest_model.pkl")

    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    modeling_pipeline = load_model(model_path)

    data_preprocessor = modeling_pipeline['data_preprocessor']
    training_features = modeling_pipeline['selected_feature_names']
    model = modeling_pipeline['model']

    # test set
    df_test = pd.read_csv(test_path)
    df_test = test_preparing(df_test)
    df_test_processed = data_preprocessor.transform(df_test[training_features])
    rmse, r2, _ = predict_eval(model, df_test_processed, df_test[target], 'test')
