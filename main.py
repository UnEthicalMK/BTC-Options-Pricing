import os
import numpy as np
from src import config
from src.data_loader import extract_clean_btc_snapshot
from src.baseline import price_options_dataframe
from src.svi_model import calibrate_svi_surface
from src.model import (
    train_stratified_split,
    train_microstructure_residual_model,
)
from src.sanity_checks import run_four_tier_sanity_checks
from src.visualize import (
    plot_svi_volatility_smiles,
    plot_microstructure_importance,
    plot_four_tier_comparison,
)


def run_pipeline():
    print("=" * 72)
    print("  INITIALIZING: Deribit BTC Options Hybrid Pricing & Volatility Engine")
    print("        (Black-76 + Arbitrage-Free SVI + XGBoost Microstructure)")
    print("=" * 72)

    # 1. Load clean BTC 08:00 UTC settlement snapshot
    print("\n[1/6] Loading clean BTC settlement snapshot (08:00 UTC)...")
    df = extract_clean_btc_snapshot()
    print(f"  Loaded {len(df):,} valid Bitcoin options across {df['expiry_str'].nunique()} maturities.")
    print(f"  BTC Forward Price Range: ${df['underlying_price'].min():,.2f} – ${df['underlying_price'].max():,.2f}")

    # 2. Tier 0 & Tier 1 analytical baselines
    print("\n[2/6] Evaluating Analytical Benchmarks (Tier 0 & Tier 1)...")
    df["bs_legacy_price"] = price_options_dataframe(
        df, sigma=0.40, model="black_scholes", r=config.LEGACY_RISK_FREE_RATE
    )

    atm_vols = {
        exp: grp.loc[grp["log_moneyness"].abs().idxmin(), "implied_vol"]
        for exp, grp in df.groupby("expiry_str")
    }
    df["atm_iv"] = df["expiry_str"].map(atm_vols)
    df["b76_flat_price"] = price_options_dataframe(df, df["atm_iv"].values, model="black_76")

    # 3. Tier 2 parametric SVI volatility surface (sequentially arbitrage-constrained)
    print("\n[3/6] Calibrating Arbitrage-Free SVI Volatility Surface...")
    calibrated_models, df = calibrate_svi_surface(df)

    for exp, res in calibrated_models.items():
        p = res["params"]
        arb_str = "ARB-FREE" if res["is_arbitrage_free"] else "WARNING: ARB"
        print(
            f"  Expiry {exp:<10} (T={res['T'] * 365.25:5.1f}d) | RMSVE: {res['rmsve'] * 100:5.2f}% | "
            f"a={p['a']:6.3f}, b={p['b']:5.3f}, rho={p['rho']:6.3f} | [{arb_str}]"
        )

    # 4. Tier 3 XGBoost microstructure residual regressor (strict OOS evaluation on test_df)
    print("\n[4/6] Training XGBoost Microstructure Residual Regressor...")
    df["iv_residual_target"] = df["implied_vol"] - df["svi_iv"]
    feature_cols = [
        "log_moneyness",
        "T",
        "relative_spread",
        "orderbook_imbalance",
        "open_interest",
    ]

    train_df, val_df, test_df = train_stratified_split(df, group_col="expiry_str")
    xgb_model = train_microstructure_residual_model(
        train_df[feature_cols],
        train_df["iv_residual_target"].values,
        val_df[feature_cols],
        val_df["iv_residual_target"].values,
    )

    # Predict strictly on held-out out-of-sample test set
    test_pred_res = xgb_model.predict(test_df[feature_cols])
    test_df["predicted_iv_residual"] = test_pred_res
    test_df["hybrid_iv"] = np.maximum(test_df["svi_iv"] + test_pred_res, 0.05)
    test_df["hybrid_price"] = price_options_dataframe(test_df, test_df["hybrid_iv"].values, model="black_76")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, config.MODEL_FILE)
    xgb_model.save_model(model_save_path)
    print(f"  XGBoost residual model saved to {model_save_path} (best iteration: {xgb_model.best_iteration})")

    # 5. Leakage-free benchmark report & arbitrage audit
    print("\n[5/6] Running Quantitative Benchmark & Arbitrage Audit...")
    metrics = run_four_tier_sanity_checks(df, test_df, calibrated_models)

    # 6. Export diagnostic visualizations
    print("\n[6/6] Generating High-Resolution Diagnostic Plots...")
    plot_svi_volatility_smiles(df, calibrated_models)
    plot_microstructure_importance(xgb_model, feature_cols)
    plot_four_tier_comparison(metrics)

    print("\n" + "=" * 72)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  All figures saved to: {config.FIGURES_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    run_pipeline()
