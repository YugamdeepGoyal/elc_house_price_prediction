import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


class GeoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=2, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state, n_init=10
        )
        self.kmeans_.fit(X[["longitude", "latitude"]])
        return self

    def transform(self, X):
        X = X.copy()
        X["geo_cluster"] = self.kmeans_.predict(X[["longitude", "latitude"]])
        return X


def skew_fix(X):
    X = X.copy()
    numeric_cols = [
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]
    for col in numeric_cols:
        X[col] = np.log1p(X[col])
    return X


def binning(X):
    X = X.copy()
    X["age_bin"] = pd.cut(
        X["housing_median_age"],
        bins=[0, 15, 30, 45, 52],
        labels=["new", "mid", "old", "very_old"],
        include_lowest=True,
    )
    return X


def get_preprocessor():
    numeric_features = [
        "longitude",
        "latitude",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
    ]

    categorical_features = ["ocean_proximity", "geo_cluster", "age_bin"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features),

            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )
    return preprocessor
