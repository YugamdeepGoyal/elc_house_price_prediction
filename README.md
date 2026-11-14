# California House Price Prediction using Machine Learning

Assignment 3: predict median house value (a **regression** problem) with a Feedforward
Artificial Neural Network (ANN) and compare it with two classical ML regression models.

- **Dataset:** California Housing Prices (`housing.csv`, 20,640 rows, from Kaggle: `camnugent/california-housing-prices`)
- **Target:** `median_house_value`
- **Models compared:** ANN (Keras), Linear Regression, Random Forest Regressor
- **Metrics:** MAE, MSE, RMSE, R²

## Project structure

```
elc_house_price_prediction/
├── eda.ipynb              # Task 1 & 2: data inspection, preprocessing checks, EDA plots
├── linear_pipeline.ipynb  # Linear Regression model (with geographic clustering)
├── tree_and_ann.ipynb     # Random Forest and ANN models, evaluation plots
├── src/
│   ├── __init__.py
│   ├── load_data.py       # downloads the dataset, creates images/ and models/, 80/20 split
│   ├── metrics.py         # MAE / MSE / RMSE / R² printing and residual plot
│   ├── linear_pipeline.py # preprocessing for Linear Regression (log fix, age bins, k-means geo clusters)
│   └── tree_nn_pipeline.py# preprocessing for Random Forest and ANN (ratio features, log fix, scaling, clipping)
├── images/                # all plots saved by the notebooks
├── models/                # saved models (created automatically when the notebooks run)
├── House Price Prediction using Machine Learning.pdf
└── README.md
```

## How to run

1. Install the requirements (Python 3.10+):
   ```
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow kagglehub filelock optuna joblib jupyter
   ```
   `scikit-learn` 1.4 or newer is needed (`root_mean_squared_error` is used in `src/metrics.py`).
2. Open the project **from its root folder**, because the notebooks import from `src/` and save to `images/` and `models/` using relative paths.
3. Run the notebooks in this order (each one runs top to bottom):
   1. `eda.ipynb`
   2. `linear_pipeline.ipynb`
   3. `tree_and_ann.ipynb`

The dataset is downloaded automatically on first run by `kagglehub`, so an internet connection is required.
The data split is always 80% training / 20% testing with `random_state=42`.

## What each part does

### Data preprocessing (Task 1)
- Dataset structure, missing values and outliers are inspected in `eda.ipynb`.
- Only `total_bedrooms` has missing values (207 rows). They are imputed with the **median**, fitted on the training set only.
- `ocean_proximity` is one-hot encoded.
- Skewed columns (rooms, bedrooms, population, households, income) are log-transformed.
- Numerical features are **standardized** with `StandardScaler`.
- Features and target are separated and split 80/20.
- The 500,001 value cap in the target is noted and handled in the analysis.

### Exploratory data analysis (Task 2)
Plots saved in `images/`: price distribution, price vs median income, price vs house age,
price vs rooms (and rooms per household), price by ocean proximity, geographic scatter plot of
price (latitude/longitude), correlation heatmap, box plots, KDE plots, pair plots, and k-means elbow/silhouette plots.

### Models (Tasks 3 and 4)
| Model | Notes |
|---|---|
| **Linear Regression** | Pipeline with log transform, age bins and 600 k-means geographic clusters (k chosen by 5-fold cross-validation) |
| **Random Forest** | 200 trees, tuned with `GridSearchCV` (`min_samples_leaf`, `max_features`); extra features: rooms/household, bedroom ratio, people/household |
| **ANN** | Keras Sequential, ReLU hidden layers, **single linear output neuron** (no classification activation), Adam optimizer, MSE loss, early stopping on a 15% validation split |

ANN experiments compared different numbers of layers/neurons, learning rates and batch sizes:
`(64, 32)`, `(128, 64, 32)` and `(256, 128, 64)` hidden units, learning rate `1e-3` / `5e-4`, batch size `64` / `128`, up to 200 epochs.
The best configuration was **(64, 32) units, learning rate 0.001, batch size 64**.

### Results on the test set (Task 5)

| Metric | ANN | Linear Regression | Random Forest |
|---|---|---|---|
| MAE | 51,734 | 38,103 | **31,459** |
| MSE | 4.92 × 10⁹ | 2.90 × 10⁹ | **2.31 × 10⁹** |
| RMSE | 70,142 | 53,836 | **48,114** |
| R² | 0.625 | 0.779 | **0.823** |

Plots for each model (actual vs predicted, residual plot, ANN training vs validation loss, Random Forest feature importance) are saved in `images/`.

### Analysis (Task 6)
- The **Random Forest** gives the lowest RMSE and the highest R², so it performs best overall.
- The ANN did **not** outperform the classical models here. With only about 20,000 tabular rows, tree ensembles tend to do better than small neural networks.
- Linear Regression shows heteroscedastic residuals (error grows with price), so a purely linear fit is limited.
- Random Forest feature importance ranks **median income** first (about 0.33), followed by **inland location** (about 0.15) and **people per household** (about 0.11). Longitude and latitude also contribute.
- The full written answers to all Task 6 questions are in the PDF report.

## Saved outputs
- `images/` contains all figures.
- `models/` contains `linear_regression.joblib`, `random_forest.joblib`, `ann.keras` and `ann_preprocessing.joblib`.