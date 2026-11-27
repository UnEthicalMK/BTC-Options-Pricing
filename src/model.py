import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from src import config


def train_stratified_split(
    df: pd.DataFrame,
    group_col: str = "expiry_str",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = config.RANDOM_STATE,
):
    """
    Performs a reproducible stratified split across expiry slices so train,
    validation, and test partitions each span the full maturity term structure.
    """
    rng = np.random.default_rng(random_state)
    train_idx, val_idx, test_idx = [], [], []

    groups = df[group_col].unique() if group_col in df.columns else [None]
    for grp in groups:
        sub_idx = (
            df.index[df[group_col] == grp].to_numpy().copy()
            if grp is not None
            else df.index.to_numpy().copy()
        )
        rng.shuffle(sub_idx)
        n = len(sub_idx)
        train_end = max(1, int(n * train_ratio)) if n >= 3 else n
        val_end = max(train_end, int(n * (train_ratio + val_ratio))) if n >= 3 else n
        train_idx.extend(sub_idx[:train_end])
        val_idx.extend(sub_idx[train_end:val_end])
        test_idx.extend(sub_idx[val_end:])

    return (
        df.loc[train_idx].copy().reset_index(drop=True),
        df.loc[val_idx].copy().reset_index(drop=True),
        df.loc[test_idx].copy().reset_index(drop=True),
    )


def train_microstructure_residual_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> XGBRegressor:
    """
    Trains an XGBoost regressor to predict the implied volatility residual:
        Delta_sigma = sigma_market - sigma_SVI
    using order book microstructure features with early stopping on validation loss.
    """
    model = XGBRegressor(
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        **config.XGB_IV_PARAMS.copy(),
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
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    mask = actual > 1e-4
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(actual[mask] - predicted[mask]) / actual[mask]) * 100.0)
