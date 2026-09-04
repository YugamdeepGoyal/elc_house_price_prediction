from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)

    print(f"Mean absolute Error: {mae}")
    print(f"Mean squared Error: {mse}")
    print(f"Root Mean squared error: {rmse}")
    print(f"R2 Score: {r2}")


def plot_residual_curve(y_true, y_pred, model):
    pass