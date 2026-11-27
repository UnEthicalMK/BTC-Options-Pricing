import numpy as np
from src.svi_model import (
    svi_raw,
    svi_implied_vol,
    check_svi_arbitrage_bounds,
    check_butterfly_arbitrage,
    check_calendar_arbitrage,
    calibrate_svi_slice,
)


def test_svi_raw_positivity():
    k = np.linspace(-0.5, 0.5, 50)
    a, b, rho, m, sigma = 0.04, 0.1, -0.25, 0.0, 0.1
    w = svi_raw(k, a, b, rho, m, sigma)
    assert np.all(w > 0.0), "Total variance must be strictly positive"


def test_svi_implied_vol_atm():
    T = 0.25
    a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
    iv = svi_implied_vol(np.array([0.0]), T, a, b, rho, m, sigma)[0]
    expected_iv = np.sqrt((a + b * sigma) / T)
    assert abs(iv - expected_iv) < 1e-6


def test_svi_implied_vol_vectorized_T():
    """Verifies svi_implied_vol supports array-like T inputs without TypeError."""
    k = np.array([0.0, 0.1, -0.1])
    T = np.array([0.1, 0.25, 0.5])
    ivs = svi_implied_vol(k, T, 0.04, 0.1, -0.2, 0.0, 0.1)
    assert ivs.shape == (3,)
    assert np.all(ivs > 0.0)


def test_svi_arbitrage_bounds_check():
    T = 0.1
    assert check_svi_arbitrage_bounds(0.05, 0.1, -0.2, 0.0, 0.1, T)[0]
    assert not check_svi_arbitrage_bounds(0.05, -0.1, -0.2, 0.0, 0.1, T)[0]
    assert not check_svi_arbitrage_bounds(0.05, 0.1, 1.2, 0.0, 0.1, T)[0]


def test_svi_butterfly_arbitrage_check():
    k_grid = np.linspace(-0.5, 0.5, 100)
    T = 0.25
    is_arb_free, min_g, _ = check_butterfly_arbitrage(k_grid, T, 0.04, 0.1, -0.2, 0.0, 0.1)
    assert is_arb_free
    assert min_g >= 0.0


def test_check_calendar_arbitrage():
    k_grid = np.linspace(-0.5, 0.5, 51)
    valid_models = {
        "T1": {"T": 0.1, "params": {"a": 0.02, "b": 0.05, "rho": -0.2, "m": 0.0, "sigma": 0.1}},
        "T2": {"T": 0.2, "params": {"a": 0.05, "b": 0.08, "rho": -0.2, "m": 0.0, "sigma": 0.1}},
    }
    ok, min_diff, viols = check_calendar_arbitrage(valid_models, k_grid=k_grid)
    assert ok and min_diff > 0.0 and len(viols) == 0

    crossing_models = {
        "T1": {"T": 0.1, "params": {"a": 0.06, "b": 0.08, "rho": -0.2, "m": 0.0, "sigma": 0.1}},
        "T2": {"T": 0.2, "params": {"a": 0.03, "b": 0.05, "rho": -0.2, "m": 0.0, "sigma": 0.1}},
    }
    ok_bad, min_diff_bad, viols_bad = check_calendar_arbitrage(crossing_models, k_grid=k_grid)
    assert not ok_bad and min_diff_bad < 0.0 and len(viols_bad) == 1


def test_calibrate_svi_slice():
    T = 0.1
    true_params = [0.03, 0.08, -0.3, 0.0, 0.12]
    k_grid = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    iv_noisy = svi_implied_vol(k_grid, T, *true_params) + np.array([0.001, -0.001, 0.002, 0.0, -0.002, 0.001, -0.001])

    result = calibrate_svi_slice(k_grid, iv_noisy, T)
    assert result["rmsve"] < 0.01
    assert result["is_arbitrage_free"]
