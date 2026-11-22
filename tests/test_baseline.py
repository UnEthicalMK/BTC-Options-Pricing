import numpy as np
import pytest
from src.baseline import (
    vectorized_black_76,
    calculate_black_76_greeks,
    vectorized_black_scholes,
)


def test_black_scholes_legacy_call_accuracy():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = 0.05
    sigma = np.array([0.20])
    calculated_call = vectorized_black_scholes(S, K, T, r, sigma, option_type="call")[0]
    expected_call = 10.450585
    assert abs(calculated_call - expected_call) < 1e-4


def test_black_scholes_legacy_put_accuracy():
    S = np.array([100.0])
    K = np.array([100.0])
    T = np.array([1.0])
    r = 0.05
    sigma = np.array([0.20])
    calculated_put = vectorized_black_scholes(S, K, T, r, sigma, option_type="put")[0]
    expected_put = 5.573505
    assert abs(calculated_put - expected_put) < 1e-4


def test_black_76_put_call_parity():
    """
    Validates Black-76 forward put-call parity (r = 0):
        C - P = F - K
    Parity error must be within numerical machine tolerance.
    """
    np.random.seed(42)
    N = 1000
    F = np.random.uniform(20000, 70000, N)
    K = np.random.uniform(20000, 70000, N)
    T = np.random.uniform(0.01, 1.5, N)
    sigma = np.random.uniform(0.30, 1.0, N)

    calls = vectorized_black_76(F, K, T, sigma, option_type="call")
    puts = vectorized_black_76(F, K, T, sigma, option_type="put")

    lhs = calls - puts
    rhs = F - K
    max_error = np.max(np.abs(lhs - rhs))
    assert max_error < 1e-9, f"Black-76 parity error too large: {max_error}"


def test_black_76_greeks():
    """
    Validates that Black-76 Greeks satisfy fundamental financial properties:
    1. 0 <= Delta_call <= 1
    2. -1 <= Delta_put <= 0
    3. Delta_call - Delta_put == 1
    4. Gamma > 0 and Vega > 0
    """
    F = np.array([45000.0])
    K = np.array([45000.0])
    T = np.array([0.25])
    sigma = np.array([0.60])

    d_call, g_call, v_call = calculate_black_76_greeks(F, K, T, sigma, option_type="call")
    d_put, g_put, v_put = calculate_black_76_greeks(F, K, T, sigma, option_type="put")

    assert 0.0 <= d_call[0] <= 1.0
    assert -1.0 <= d_put[0] <= 0.0
    assert abs((d_call[0] - d_put[0]) - 1.0) < 1e-9
    assert g_call[0] > 0.0
    assert v_call[0] > 0.0
    assert abs(g_call[0] - g_put[0]) < 1e-9
    assert abs(v_call[0] - v_put[0]) < 1e-9
