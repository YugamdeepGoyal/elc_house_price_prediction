import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUM_COLS = [
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "rooms_per_hh",
    "bedroom_ratio",
    "pop_per_hh",
]
CAT_COLS = ["ocean_proximity"]


def add_ratios(X):
    X = X.copy()
    X["rooms_per_hh"] = X["total_rooms"] / X["households"]
    X["bedroom_ratio"] = X["total_bedrooms"] / X["total_rooms"]
    X["pop_per_hh"] = X["population"] / X["households"]
    return X


def log_fix(X):
    X = X.copy()
    skewed = [
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "rooms_per_hh",
        "pop_per_hh",
    ]
    for col in skewed:
        X[col] = np.log1p(X[col])
    return X


def clip_scaled(X):
    # a few rows have extreme ratios, keep them from wrecking the ANN
    return np.clip(X, -4, 4)


def get_tree_preprocessor():
    # trees don't care about scale, so just impute and encode
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


def get_nn_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )
