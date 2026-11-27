import os
from pathlib import Path

# Repository Root Directory
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
SAMPLE_DATA_DIR = os.path.join(ROOT_DIR, "data", "samples")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
FIGURES_DIR = os.path.join(ROOT_DIR, "figures")

# Snapshot and Model Filenames
RAW_OPTIONS_FILE = "deribit_options_chain_2024-01-01_OPTIONS.csv.gz"
SNAPSHOT_FILE = "btc_settlement_snapshot_20240101_0800.csv"
MODEL_FILE = "xgboost_microstructure_residual.json"

# Financial Constants
# Legacy benchmark constant (Deribit Black-76 uses r = 0 due to coin-margining)
LEGACY_RISK_FREE_RATE = 0.05

# SVI Optimization Parameters & Arbitrage Verification Grid
SVI_PARAM_BOUNDS = (
    [-1.0, 0.0001, -0.999, -2.0, 0.001],  # Lower: [a, b, rho, m, sigma]
    [5.0,  5.0,     0.999,  2.0, 3.0],    # Upper: [a, b, rho, m, sigma]
)
SVI_ARB_K_GRID_MIN = -1.0
SVI_ARB_K_GRID_MAX = 1.0

# Reproducibility & Machine Learning Hyperparameters
RANDOM_STATE = 42

XGB_IV_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.03,
    "n_estimators": 250,
    "objective": "reg:squarederror",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.05,
    "reg_lambda": 1.5,
    "min_child_weight": 3,
}

EARLY_STOPPING_ROUNDS = 25
