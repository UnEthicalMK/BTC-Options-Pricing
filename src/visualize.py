import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src import config
from src.svi_model import svi_implied_vol

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "font.sans-serif": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})


def plot_svi_volatility_smiles(df: pd.DataFrame, calibrated_models: dict, fig_dir: str = config.FIGURES_DIR):
    """
    Plots calibrated SVI volatility smiles vs. Deribit market IV across all maturities.
    """
    os.makedirs(fig_dir, exist_ok=True)
    expiries = [e for e in df["expiry_str"].unique() if str(e) in calibrated_models]

    n_exp = len(expiries)
    if n_exp == 0:
        return

    cols = min(3, n_exp)
    rows = int(np.ceil(n_exp / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.0 * rows), squeeze=False)
    fig.suptitle("Deribit BTC Options: Arbitrage-Free SVI Smiles vs. Mark IV", fontsize=15, y=0.995)

    for idx, exp in enumerate(expiries):
        ax = axes[idx // cols, idx % cols]
        sub = df[df["expiry_str"] == exp].sort_values("log_moneyness")
        m = calibrated_models[str(exp)]
        p = m["params"]
        T_val = m["T"]

        fine_k = np.linspace(sub["log_moneyness"].min() - 0.05, sub["log_moneyness"].max() + 0.05, 200)
        fine_iv = svi_implied_vol(fine_k, T_val, p["a"], p["b"], p["rho"], p["m"], p["sigma"]) * 100.0

        ax.scatter(sub["log_moneyness"], sub["implied_vol"] * 100.0, color="crimson", s=24, label="Deribit Mark IV", zorder=3)
        ax.plot(fine_k, fine_iv, color="royalblue", linewidth=2.0, label=f"SVI Fit (RMSVE: {m['rmsve']*100:.2f}%)", zorder=2)
        ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)

        ax.set_title(f"Expiry: {exp} (T = {T_val * 365.25:.1f} days)", pad=8)
        ax.set_xlabel("Log-Moneyness ln(K / F)")
        ax.set_ylabel("Implied Volatility (%)")
        ax.legend(loc="upper right", frameon=True, fontsize=9)

    for idx in range(n_exp, rows * cols):
        axes[idx // cols, idx % cols].axis("off")

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "01_svi_volatility_smiles.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [visualize] Volatility smile plot saved to {save_path}")


def plot_microstructure_importance(model, features: list, fig_dir: str = config.FIGURES_DIR):
    """
    Plots feature importance of the XGBoost microstructure residual regressor.
    """
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(9, 5))

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    importance.plot(kind="barh", color="teal", edgecolor="black", alpha=0.85)

    plt.title("XGBoost Microstructure Feature Importance (Gain Weighting)", fontsize=13, pad=12)
    plt.xlabel("Relative Gain Score", fontsize=11)
    plt.ylabel("Input Features", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "02_microstructure_feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [visualize] Feature importance plot saved to {save_path}")


def plot_four_tier_comparison(metrics: dict, fig_dir: str = config.FIGURES_DIR):
    """
    Plots Out-of-Sample Volatility RMSVE and Dollar Mark-Price RMSE across the 4 model tiers.
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    names_short = ["Naive BS\n(r=0.05, 40%)", "Black-76\nFlat ATM", "SVI Surface\n(Parametric)", "Hybrid Engine\n(SVI + XGB OOS)"]
    colors = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]

    vol_vals = list(metrics["vol_rmsve"].values())
    bars1 = ax1.bar(names_short, vol_vals, color=colors, edgecolor="black", alpha=0.9)
    ax1.set_title("Out-of-Sample Volatility Error (RMSVE %)", fontsize=13, pad=12)
    ax1.set_ylabel("RMSVE (Percentage Points)", fontsize=11)
    ax1.set_ylim(0, max(vol_vals) * 1.15)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + max(vol_vals) * 0.02, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    price_vals = list(metrics["price_rmse"].values())
    bars2 = ax2.bar(names_short, price_vals, color=colors, edgecolor="black", alpha=0.9)
    ax2.set_title("Out-of-Sample Option Pricing Error vs. Mark (USD RMSE)", fontsize=13, pad=12)
    ax2.set_ylabel("RMSE ($ USD)", fontsize=11)
    ax2.set_ylim(0, max(price_vals) * 1.15)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + max(price_vals) * 0.02, f"${yval:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "03_four_tier_benchmark_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [visualize] Benchmark comparison plot saved to {save_path}")
