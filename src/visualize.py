import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src import config
from src.svi_model import svi_implied_vol

# Professional publication plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'font.sans-serif': 'DejaVu Sans',
    'axes.titlesize': 13,
    'axes.labelsize': 11
})


def plot_svi_volatility_smiles(df: pd.DataFrame, calibrated_models: dict, fig_dir: str = config.FIGURES_DIR):
    """
    Plots the calibrated SVI volatility smiles vs. market bid/ask/mid quotes
    across all active maturity slices.
    """
    os.makedirs(fig_dir, exist_ok=True)
    expiries = [e for e in df['expiry_str'].unique() if str(e) in calibrated_models]

    n_exp = len(expiries)
    if n_exp == 0:
        return

    cols = min(3, n_exp)
    rows = int(np.ceil(n_exp / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.0 * rows), squeeze=False)
    fig.suptitle("Deribit BTC Options: Calibrated SVI Volatility Smiles vs. Market Quotes", fontsize=15, y=0.99)

    for idx, exp in enumerate(expiries):
        ax = axes[idx // cols, idx % cols]
        sub = df[df['expiry_str'] == exp].sort_values('log_moneyness')
        m = calibrated_models[str(exp)]
        p = m['params']
        T_val = m['T']

        # Fine log-moneyness grid for smooth SVI curve
        k_min = sub['log_moneyness'].min() - 0.05
        k_max = sub['log_moneyness'].max() + 0.05
        fine_k = np.linspace(k_min, k_max, 200)
        fine_iv = svi_implied_vol(fine_k, T_val, p['a'], p['b'], p['rho'], p['m'], p['sigma']) * 100.0

        # Plot market mid quotes
        ax.scatter(sub['log_moneyness'], sub['implied_vol'] * 100.0, color='crimson', s=25, label='Deribit Mark IV', zorder=3)
        # Plot SVI fitted smile
        ax.plot(fine_k, fine_iv, color='royalblue', linewidth=2.0, label=f'SVI Fit (RMSVE: {m["rmsve"]*100:.2f}%)', zorder=2)

        # Plot ATM vertical dashed line
        ax.axvline(0.0, color='gray', linestyle=':', alpha=0.7)

        ax.set_title(f"Expiry: {exp} (T = {T_val*365.25:.1f} days)", pad=8)
        ax.set_xlabel("Log-Moneyness ln(K / F)")
        ax.set_ylabel("Implied Volatility (%)")
        ax.legend(loc='upper right', frameon=True, fontsize=9)

    # Hide any unused subplots
    for idx in range(n_exp, rows * cols):
        axes[idx // cols, idx % cols].axis('off')

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "01_svi_volatility_smiles.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [visualize] Volatility smile plot saved to {save_path}")


def plot_microstructure_importance(model, features: list, fig_dir: str = config.FIGURES_DIR):
    """
    Plots the feature importance of the XGBoost microstructure residual regressor.
    """
    os.makedirs(fig_dir, exist_ok=True)
    plt.figure(figsize=(9, 5))

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
    importance.plot(kind='barh', color='teal', edgecolor='black', alpha=0.85)

    plt.title("XGBoost Microstructure Feature Importance (Gain Weighting)", fontsize=13, pad=12)
    plt.xlabel("Relative Gain Score", fontsize=11)
    plt.ylabel("Microstructure Order Book Features", fontsize=11)
    plt.tight_layout()

    save_path = os.path.join(fig_dir, "02_microstructure_feature_importance.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [visualize] Feature importance plot saved to {save_path}")


def plot_four_tier_comparison(metrics: dict, fig_dir: str = config.FIGURES_DIR):
    """
    Plots a multi-tier comparison of Volatility RMSVE and Dollar Price RMSE.
    """
    os.makedirs(fig_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # 1. Volatility RMSVE
    vol_data = metrics['vol_rmsve']
    names_short = ["Naive BS\n(r=0.05, 40%)", "Black-76\nFlat ATM", "SVI Surface\n(Parametric)", "Hybrid Engine\n(SVI + XGB)"]
    vol_vals = list(vol_data.values())
    colors = ['#d9534f', '#f0ad4e', '#5bc0de', '#5cb85c']

    bars1 = ax1.bar(names_short, vol_vals, color=colors, edgecolor='black', alpha=0.9)
    ax1.set_title("Volatility Surface Error (RMSVE %)", fontsize=13, pad=12)
    ax1.set_ylabel("RMSVE (Percentage Points)", fontsize=11)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.2f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Price RMSE
    price_data = metrics['price_rmse']
    price_vals = list(price_data.values())

    bars2 = ax2.bar(names_short, price_vals, color=colors, edgecolor='black', alpha=0.9)
    ax2.set_title("Option Pricing Error (Dollar RMSE)", fontsize=13, pad=12)
    ax2.set_ylabel("RMSE ($ USD)", fontsize=11)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 5.0, f"${yval:,.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(fig_dir, "03_four_tier_benchmark_comparison.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  [visualize] Benchmark comparison plot saved to {save_path}")
