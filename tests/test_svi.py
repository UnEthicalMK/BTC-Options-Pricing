import numpy as np
import pytest
from src.svi_model import (
    svi_raw,
    svi_implied_vol,
    check_svi_arbitrage_bounds,
    check_butterfly_arbitrage,
    calibrate_svi_slice,
)


def test_svi_raw_positivity():
    """
    Validates that SVI total variance w(k) produces strictly positive variance
    under typical parameter sets.
    """
    k = np.linspace(-0.5, 0.5, 50)
    a, b, rho, m, sigma = 0.04, 0.1, -0.25, 0.0, 0.1
    w = svi_raw(k, a, b, rho, m, sigma)
    assert np.all(w > 0.0), "Total variance must be strictly positive"


def test_svi_implied_vol_atm():
    """
    At the money (k = 0, m = 0), w(0) = a + b * sigma.
    sigma_IV = sqrt((a + b * sigma) / T).
    """
    T = 0.25
    a, b, rho, m, sigma = 0.04, 0.1, 0.0, 0.0, 0.1
    iv = svi_implied_vol(np.array([0.0]), T, a, b, rho, m, sigma)[0]
    expected_w = a + b * sigma  # 0.04 + 0.01 = 0.05
    expected_iv = np.sqrt(expected_w / T)  # sqrt(0.05 / 0.25) = sqrt(0.20)
    assert abs(iv - expected_iv) < 1e-6


def test_svi_arbitrage_bounds_check():
    """
    Tests parameter boundary validator for invalid b, rho, and negative min variance.
    """
    T = 0.1
    # Valid
    is_valid, _ = check_svi_arbitrage_bounds(0.05, 0.1, -0.2, 0.0, 0.1, T)
    assert is_valid

    # Negative b (invalid)
    is_valid, msg = check_svi_arbitrage_bounds(0.05, -0.1, -0.2, 0.0, 0.1, T)
    assert not is_valid

    # |rho| >= 1 (invalid)
    is_valid, msg = check_svi_arbitrage_bounds(0.05, 0.1, 1.2, 0.0, 0.1, T)
    assert not is_valid


def test_svi_butterfly_arbitrage_check():
    """
    Validates the Breeden-Litzenberger density g(k) check function.
    """
    k_grid = np.linspace(-0.5, 0.5, 100)
    T = 0.25
    a, b, rho, m, sigma = 0.04, 0.1, -0.2, 0.0, 0.1
    is_arb_free, min_g, g_vals = check_butterfly_arbitrage(k_grid, T, a, b, rho, m, sigma)
    assert is_arb_free
    assert min_g >= 0.0


def test_calibrate_svi_slice():
    """
    Tests calibration of SVI on a synthetic smile with known parameters.
    """
    T = 0.1
    true_params = [0.03, 0.08, -0.3, 0.0, 0.12]
    k_grid = np.array([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3])
    true_iv = svi_implied_vol(k_grid, T, *true_params)

    # Add small perturbation
    iv_noisy = true_iv + np.array([0.001, -0.001, 0.002, 0.0, -0.002, 0.001, -0.001])

    result = calibrate_svi_slice(k_grid, iv_noisy, T)
    assert result["rmsve"] < 0.01  # Fit error < 1% vol
    assert result["is_arbitrage_free"]
