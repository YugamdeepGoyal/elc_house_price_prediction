from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np


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
    plt.figure(figsize=(8, 5))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true-y_pred
    sns.regplot(x=y_pred, y=residuals, lowess=True, 
                scatter_kws={"alpha": 0.3, "color": "green"}, 
                line_kws={"color": "red"})
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residual Plot with Lowess Curve")
    plt.axhline(0, color="black", linestyle="--")
    plt.tight_layout()
    plt.savefig(f"images/{model}_residual_curve.png")
    plt.show()
