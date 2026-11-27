import os
import numpy as np
import pandas as pd
from src.baseline import price_options_dataframe
from src.data_loader import _apply_microstructure_filters, extract_clean_btc_snapshot
from src.model import train_stratified_split, train_microstructure_residual_model, compute_rmse, compute_mape
from src.svi_model import calibrate_svi_surface, svi_implied_vol
from src.sanity_checks import run_four_tier_sanity_checks


def _make_synthetic_chain() -> pd.DataFrame:
    """Creates a synthetic multi-expiry BTC options DataFrame for isolated testing."""
    rows = []
    F = 45000.0
    for exp, T, atm_iv in [("12JAN24", 11 / 365.25, 0.55), ("26JAN24", 25 / 365.25, 0.60), ("29MAR24", 88 / 365.25, 0.65)]:
        for strike in np.linspace(36000, 54000, 13):
            k = float(np.log(strike / F))
            iv = float(svi_implied_vol(k, T, atm_iv**2 * T * 0.8, 0.04, -0.25, 0.0, 0.12))
            otype = "put" if strike < F else "call"
            rows.append({
                "symbol": f"BTC-{exp}-{int(strike)}-{'P' if otype == 'put' else 'C'}",
                "option_type": otype,
                "strike": strike,
                "underlying_price": F,
                "T": T,
                "expiry_str": exp,
                "log_moneyness": k,
                "best_bid": 0.02,
                "best_ask": 0.022,
                "mark_price": 0.021,
                "mark_iv": iv * 100.0,
                "implied_vol": iv,
                "market_mid": 0.021 * F,
                "mark_usd": 0.021 * F,
                "relative_spread": 0.002 / 0.021,
                "orderbook_imbalance": 0.1,
                "open_interest": 100.0,
            })
    df = pd.DataFrame(rows)
    df["mark_usd"] = price_options_dataframe(df, df["implied_vol"].values, model="black_76")
    df["market_mid"] = df["mark_usd"] * 1.01
    return df


def test_microstructure_filters_reject_dummy_and_crossed():
    df = _make_synthetic_chain().iloc[:4].copy()
    df.loc[0, "best_ask"] = 1.00  # dummy ask >= 0.95
    df.loc[1, "best_ask"] = 0.01  # crossed market (ask < bid)
    cleaned = _apply_microstructure_filters(df)
    assert len(cleaned) == 2
    assert (cleaned["best_ask"] < 0.95).all()
    assert (cleaned["best_ask"] > cleaned["best_bid"]).all()


def test_price_options_dataframe_mixed_types():
    df = _make_synthetic_chain()
    prices_b76 = price_options_dataframe(df, df["implied_vol"].values, model="black_76")
    prices_bs = price_options_dataframe(df, 0.40, model="black_scholes", r=0.05)
    assert prices_b76.shape == (len(df),)
    assert prices_bs.shape == (len(df),)
    assert np.all(prices_b76 > 0.0)


def test_stratified_split_and_xgb_oos_flow():
    df = _make_synthetic_chain()
    _, df = calibrate_svi_surface(df)
    df["bs_legacy_price"] = price_options_dataframe(df, 0.40, model="black_scholes", r=0.05)
    df["atm_iv"] = 0.60
    df["b76_flat_price"] = price_options_dataframe(df, df["atm_iv"].values, model="black_76")
    df["iv_residual_target"] = df["implied_vol"] - df["svi_iv"]

    tr, va, te = train_stratified_split(df, group_col="expiry_str")
    assert len(tr) + len(va) + len(te) == len(df)
    # Every expiry present in test set
    assert set(te["expiry_str"].unique()) == set(df["expiry_str"].unique())

    fcols = ["log_moneyness", "T", "relative_spread", "orderbook_imbalance", "open_interest"]
    model = train_microstructure_residual_model(tr[fcols], tr["iv_residual_target"].values, va[fcols], va["iv_residual_target"].values)
    te["hybrid_iv"] = np.maximum(te["svi_iv"] + model.predict(te[fcols]), 0.05)
    te["hybrid_price"] = price_options_dataframe(te, te["hybrid_iv"].values, model="black_76")

    cal_models, _ = calibrate_svi_surface(df)
    metrics = run_four_tier_sanity_checks(df, te, cal_models)
    assert metrics["arbitrage"]["calendar_free"] is True
    assert metrics["arbitrage"]["butterfly_passed"] == 3


def test_snapshot_loader_btc_only_and_sample_fallback(tmp_path, monkeypatch):
    from src import config
    sample_csv = os.path.join(config.SAMPLE_DATA_DIR, config.SNAPSHOT_FILE)
    if os.path.exists(sample_csv):
        monkeypatch.setattr(config, "PROCESSED_DATA_DIR", str(tmp_path / "processed"))
        monkeypatch.setattr(config, "RAW_DATA_DIR", str(tmp_path / "missing_raw"))
        loaded = extract_clean_btc_snapshot()
        assert len(loaded) > 100
        assert loaded["symbol"].str.startswith("BTC-").all()
        assert (loaded["best_ask"] < 0.95).all()


def test_metrics_helpers():
    a = np.array([100.0, 200.0])
    b = np.array([110.0, 190.0])
    assert abs(compute_rmse(a, b) - 10.0) < 1e-9
    assert abs(compute_mape(a, b) - 7.5) < 1e-9
