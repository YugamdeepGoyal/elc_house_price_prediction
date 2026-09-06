import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class GeoTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        self.kmeans_ = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self.kmeans_.fit(X[["longitude", "latitude"]])
        return self

    def transform(self, X):
        X = X.copy()
        X['geo_cluster'] = self.kmeans_.predict(X[["longitude", "latitude"]])
        return X