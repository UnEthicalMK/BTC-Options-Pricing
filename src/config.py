import os
from pathlib import Path

# Repository Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

# Snapshot and Processed Filenames
RAW_OPTIONS_FILE = "deribit_options_chain_2024-01-01_OPTIONS.csv.gz"
SNAPSHOT_FILE = "btc_settlement_snapshot_20240101_0800.csv"
MODEL_FILE = "xgboost_microstructure_residual.json"

# Financial Constants
# Deribit crypto options are inverse coin-margined referenced to forward/futures (Black-76).
# Cash interest discounting is zero (r=0) because settlement occurs in cryptocurrency.
BLACK76_RATE = 0.0

# Legacy benchmark comparison constant
LEGACY_RISK_FREE_RATE = 0.05

# Filtering thresholds for liquid trading bounds
MIN_DELTA = 0.05
MAX_DELTA = 0.95

# SVI Optimization Parameters
SVI_PARAM_BOUNDS = (
    [-1.0, 0.0001, -0.999, -2.0, 0.001],  # Lower: [a, b, rho, m, sigma]
    [5.0,  5.0,     0.999,  2.0, 3.0]    # Upper: [a, b, rho, m, sigma]
)

# Machine Learning Hyperparameters (XGBoost for Microstructure Residual in IV Space)
XGB_IV_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 250,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "min_child_weight": 3,
}

EARLY_STOPPING_ROUNDS = 25
