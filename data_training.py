import os
import joblib
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import r2_score, root_mean_squared_error
from data_evaluation import *


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# make sure "models" folder exists
os.makedirs(MODELS_DIR, exist_ok=True)

seed = 31 
degree = 6 
do_feature_selection = False  # do feature selection using Lasso
do_add_speed = False 
speed_model, data_preprocessor_speed = None, None 
np.random.seed(seed)

target = 'trip_duration'

# Base feature sets
numeric_features = [
    "dropoff_longitude", "dropoff_latitude", "distance_haversine", "distance_manhattan", "direction",
]

categorical_features = [
    "passenger_count", "pickup_day", "pickup_month", "pickup_Season",
    "pickup_dayofweek", "pickup_hour", "store_and_fwd_flag", "vendor_id"
]

# Feature set for the optional speed model
speed_numeric_features = [
    "pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
    "distance_haversine", "distance_manhattan", "direction",
]

speed_categorical_features = [
    "passenger_count", "pickup_day", "pickup_month", "pickup_Season",
    "pickup_dayofweek", "pickup_hour", "store_and_fwd_flag", "vendor_id"
]


def data_preprocessing_pipeline(categorical_features=None, numeric_features=None):
    if categorical_features is None:
        categorical_features = []
    if numeric_features is None:
        numeric_features = []

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    return Pipeline(steps=[('preprocessor', column_transformer)])

def log_transform(x):
    # Safe log1p of nonnegative values; clip negatives to 0 before log1p
    return np.log1p(np.maximum(x, 0))

def with_suffix(_, names: list[str]):
    return [name + '__log' for name in names]

def pipeline(train_df, val_df, do_feature_selection=True):
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)
    train_features = numeric_features + categorical_features

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree)),
        ('log', LogFeatures),
    ])

    categorical_transformer = Pipeline(steps=[
        ('ohe', OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ])

    column_transformer = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

    data_preprocessor = Pipeline(steps=[('preprocessor', column_transformer)])

    X_train = data_preprocessor.fit_transform(train_df[train_features])
    X_val = data_preprocessor.transform(val_df[train_features])

    if do_feature_selection:
        lasso_cv = LassoCV(cv=5, max_iter=5000, random_state=seed)
        lasso_cv.fit(X_train, train_df[target])
        selected_feature_indices = [i for i, coef in enumerate(lasso_cv.coef_) if coef != 0]

        all_feature_names = data_preprocessor.named_steps['preprocessor'].get_feature_names_out()
        selected_feature_names = all_feature_names[selected_feature_indices]

        print("LassoCV selected features:", selected_feature_names)

        X_train_sel = X_train[:, selected_feature_indices]
        X_val_sel = X_val[:, selected_feature_indices]
    else:
        X_train_sel = X_train
        X_val_sel = X_val
        # For downstream transform we still need to pass original raw columns
        selected_feature_names = train_features

    ridge = Ridge(alpha=1, random_state=seed)
    ridge.fit(X_train_sel, train_df[target])

    train_rmse, train_r2, _ = predict_eval(ridge, X_train_sel, train_df[target], "train")
    val_rmse, val_r2, _ = predict_eval(ridge, X_val_sel, val_df[target], "val")

    return ridge, selected_feature_names, data_preprocessor, train_rmse, train_r2, val_rmse, val_r2

def add_speed(df, speed_data_preprocessor, model=None, train_or_predict="train", validate=False):
    speed_train_features = speed_categorical_features + speed_numeric_features

    if train_or_predict == "train":
        X = speed_data_preprocessor.fit_transform(df[speed_train_features])
    else:
        X = speed_data_preprocessor.transform(df[speed_train_features])

    if train_or_predict == "train":
        df_speed = df["distance_haversine"] / (df['trip_duration'])
        local_speed_model = Ridge(alpha=1, random_state=seed)
        local_speed_model.fit(X, df_speed)
        df["speed"] = local_speed_model.predict(X)

        rmse = root_mean_squared_error(df_speed, df["speed"])
        r2 = r2_score(df_speed, df["speed"])
        print(f"Speed Model Train RMSE: {rmse:.4f}, Train R2: {r2:.4f}")
        # mutate global numeric_features so main model can use it
        if "speed" not in numeric_features:
            numeric_features.append("speed")
        return local_speed_model, df
    else:
        df["speed"] = model.predict(X)

        if validate:
            df_speed = df["distance_haversine"] / (df['trip_duration'])
            rmse = root_mean_squared_error(df_speed, df["speed"])
            r2 = r2_score(df_speed, df["speed"])
            print(f"Test Speed Model RMSE: {rmse:.4f}, Test R2: {r2:.4f}")

        return df["speed"]

if __name__ == "__main__":
    data_version = 0
    data_path = os.path.join(PROJECT_ROOT, "saving_data", str(data_version))
    train_path = os.path.join(data_path, "train.csv")
    val_path = os.path.join(data_path, "val.csv")

    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    if do_add_speed:
        data_preprocessor_speed = data_preprocessing_pipeline(
            categorical_features=speed_categorical_features,
            numeric_features=speed_numeric_features
        )

        speed_model, df_train = add_speed(df_train, data_preprocessor_speed, train_or_predict="train")
        df_val["speed"] = add_speed(
            df_val, data_preprocessor_speed, model=speed_model,
            train_or_predict="predict", validate=True
        )
        # train_features is defined inside pipeline; nothing else to do here.

    model, selected_feature_names, data_preprocessor, train_rmse, train_r2, val_rmse, val_r2 = pipeline(
        df_train, df_val, do_feature_selection
    )

    now = datetime.now().strftime("%Y%m%d_%H%M%S")  # timestamp

    filename = os.path.join(
        MODELS_DIR,
        f'model_{now}_Train_RMSE_{train_rmse:.2f}_R2_{train_r2:.2f}_'
        f'Val_RMSE_{val_rmse:.2f}_R2_{val_r2:.2f}.pkl'
    )

    model_data = {
        'model': model,
        'speed_model': speed_model,
        'data_preprocessor_speed': data_preprocessor_speed,
        'data_path': data_path,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'selected_feature_names': selected_feature_names,
        'data_preprocessor': data_preprocessor,
        'data_version': data_version,
        'random_seed': seed
    }

    latest_path = os.path.join(MODELS_DIR, "latest_model.pkl")
    joblib.dump(model_data, latest_path)
    joblib.dump(model_data, filename)
    print(f"Saved: {latest_path}")
    print(f"Saved snapshot: {filename}")
