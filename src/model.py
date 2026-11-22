import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src import config
from src.baseline import vectorized_black_76


def train_chronological_split(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15):
    """
    Splits the options dataset chronologically or stratified by moneyness/expiry.
    Returns train, validation, and test slices.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def train_microstructure_residual_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> XGBRegressor:
    """
    Trains an XGBoost regressor to predict the implied volatility residual:
        Delta_sigma = sigma_market - sigma_SVI

    Using genuine order book microstructure features:
    - log_moneyness: ln(K / F)
    - T: time to maturity
    - relative_spread: (ask - bid) / mid
    - orderbook_imbalance: (bid_depth - ask_depth) / total_depth
    - open_interest: total outstanding contracts

    Properly passes early_stopping_rounds to prevent overfitting.
    """
    params = config.XGB_IV_PARAMS.copy()
    
    model = XGBRegressor(
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        **params
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean((actual - predicted)**2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual > 1e-4
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100.0)
