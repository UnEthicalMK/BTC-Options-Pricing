import os
import pandas as pd
import numpy as np
from src import config
from src.data_loader import extract_clean_btc_snapshot
from src.baseline import (
    vectorized_black_76,
    vectorized_black_scholes,
)
from src.svi_model import calibrate_svi_surface
from src.model import (
    train_chronological_split,
    train_microstructure_residual_model,
)
from src.sanity_checks import run_four_tier_sanity_checks
from src.visualize import (
    plot_svi_volatility_smiles,
    plot_microstructure_importance,
    plot_four_tier_comparison,
)


def run_pipeline():
    print("=" * 65)
    print("  INITIALIZING: Deribit BTC Options True Hybrid Pricing Engine")
    print("       (Black-76 + SVI Surface + XGBoost Microstructure)")
    print("=" * 65)

    # ------------------------------------------------------------------ #
    # Step 1: Ingest & Clean BTC 08:00 UTC Snapshot                      #
    # ------------------------------------------------------------------ #
    print("\n[1/6] Loading clean BTC settlement snapshot (08:00 UTC)...")
    df = extract_clean_btc_snapshot()
    print(f"  Loaded {len(df):,} valid Bitcoin options across {df['expiry_str'].nunique()} maturities.")
    print(f"  BTC Forward Price Range: ${df['underlying_price'].min():,.2f} – ${df['underlying_price'].max():,.2f}")

    # ------------------------------------------------------------------ #
    # Step 2: Tier 0 & Tier 1 Analytical Baselines                       #
    # ------------------------------------------------------------------ #
    print("\n[2/6] Evaluating Analytical Benchmarks (Tier 0 & Tier 1)...")

    # Tier 0: Naive Legacy Black-Scholes (r = 0.05, constant 40% historical vol)
    # Applied to spot price with cash discounting
    df['bs_legacy_price'] = vectorized_black_scholes(
        S=df['underlying_price'].values,
        K=df['strike'].values,
        T=df['T'].values,
        r=config.LEGACY_RISK_FREE_RATE,
        sigma=0.40,
        option_type='call'
    )
    put_mask = df['option_type'].str.lower() == 'put'
    if put_mask.any():
        df.loc[put_mask, 'bs_legacy_price'] = vectorized_black_scholes(
            S=df.loc[put_mask, 'underlying_price'].values,
            K=df.loc[put_mask, 'strike'].values,
            T=df.loc[put_mask, 'T'].values,
            r=config.LEGACY_RISK_FREE_RATE,
            sigma=0.40,
            option_type='put'
        )

    # Tier 1: Black-76 with Flat ATM Volatility per Expiry Slice
    # Find ATM contract for each expiry (closest to k = 0)
    atm_vols = df.groupby('expiry_str').apply(
        lambda g: g.loc[g['log_moneyness'].abs().idxmin(), 'implied_vol']
    ).to_dict()
    df['atm_iv'] = df['expiry_str'].map(atm_vols)

    df['b76_flat_price'] = vectorized_black_76(
        F=df['underlying_price'].values,
        K=df['strike'].values,
        T=df['T'].values,
        sigma=df['atm_iv'].values,
        option_type='call'
    )
    if put_mask.any():
        df.loc[put_mask, 'b76_flat_price'] = vectorized_black_76(
            F=df.loc[put_mask, 'underlying_price'].values,
            K=df.loc[put_mask, 'strike'].values,
            T=df.loc[put_mask, 'T'].values,
            sigma=df.loc[put_mask, 'atm_iv'].values,
            option_type='put'
        )

    # ------------------------------------------------------------------ #
    # Step 3: Tier 2 Parametric SVI Volatility Surface                   #
    # ------------------------------------------------------------------ #
    print("\n[3/6] Calibrating SVI (Stochastic Volatility Inspired) Surface...")
    calibrated_models, df = calibrate_svi_surface(df)

    for exp, res in calibrated_models.items():
        p = res['params']
        arb_str = "ARB-FREE" if res['is_arbitrage_free'] else "WARNING: ARB"
        print(f"  Expiry {exp:<10} (T={res['T']*365.25:5.1f}d) | RMSVE: {res['rmsve']*100:5.2f}% | "
              f"a={p['a']:6.3f}, b={p['b']:5.3f}, rho={p['rho']:6.3f} | [{arb_str}]")

    # ------------------------------------------------------------------ #
    # Step 4: Tier 3 Machine Learning Microstructure Residual Regressor  #
    # ------------------------------------------------------------------ #
    print("\n[4/6] Training XGBoost Microstructure Residual Regressor...")

    # Target: Implied Volatility Residual
    df['iv_residual_target'] = df['implied_vol'] - df['svi_iv']

    feature_cols = [
        'log_moneyness',
        'T',
        'relative_spread',
        'orderbook_imbalance',
        'open_interest'
    ]

    train_df, val_df, test_df = train_chronological_split(df, train_ratio=0.70, val_ratio=0.15)

    X_train, y_train = train_df[feature_cols], train_df['iv_residual_target'].values
    X_val,   y_val   = val_df[feature_cols],   val_df['iv_residual_target'].values
    X_test           = test_df[feature_cols]

    xgb_model = train_microstructure_residual_model(X_train, y_train, X_val, y_val)

    # Predict residual on entire dataset for complete surface representation
    df['predicted_iv_residual'] = xgb_model.predict(df[feature_cols])
    # Hybrid IV: SVI baseline + Microstructure correction (bounded strictly positive)
    df['hybrid_iv'] = np.maximum(df['svi_iv'] + df['predicted_iv_residual'], 0.05)

    # Evaluate Black-76 Hybrid Option Prices
    df['hybrid_price'] = vectorized_black_76(
        F=df['underlying_price'].values,
        K=df['strike'].values,
        T=df['T'].values,
        sigma=df['hybrid_iv'].values,
        option_type='call'
    )
    if put_mask.any():
        df.loc[put_mask, 'hybrid_price'] = vectorized_black_76(
            F=df.loc[put_mask, 'underlying_price'].values,
            K=df.loc[put_mask, 'strike'].values,
            T=df.loc[put_mask, 'T'].values,
            sigma=df.loc[put_mask, 'hybrid_iv'].values,
            option_type='put'
        )

    # Save model artifact
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    model_save_path = os.path.join(config.MODEL_DIR, config.MODEL_FILE)
    xgb_model.save_model(model_save_path)
    print(f"  XGBoost residual model saved to {model_save_path}")

    # ------------------------------------------------------------------ #
    # Step 5: Quantitative Sanity Checks & 4-Tier Benchmark Report       #
    # ------------------------------------------------------------------ #
    print("\n[5/6] Running 4-Tier Quantitative Benchmark Report...")
    metrics = run_four_tier_sanity_checks(df)

    # ------------------------------------------------------------------ #
    # Step 6: Diagnostic Visualization Suite                             #
    # ------------------------------------------------------------------ #
    print("\n[6/6] Generating High-Resolution Diagnostic Plots...")
    plot_svi_volatility_smiles(df, calibrated_models)
    plot_microstructure_importance(xgb_model, feature_cols)
    plot_four_tier_comparison(metrics)

    print("\n" + "=" * 65)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"  All figures saved to: {config.FIGURES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    run_pipeline()
