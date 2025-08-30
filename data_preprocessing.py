import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

degree = 6
seed = 31

def log_transform(x):
    # Safe log1p of nonnegative values; clip negatives to 0 before log1p
    return np.log1p(np.maximum(x, 0))

def with_suffix(_, names: list[str]):
    return [name + '__log' for name in names]

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


def full_pipeline(categorical_features, numeric_features):
    """Builds the preprocessing pipeline with scaling, poly features, log transform, and OHE"""
    LogFeatures = FunctionTransformer(log_transform, feature_names_out=with_suffix)

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

    return Pipeline(steps=[('preprocessor', column_transformer)])
