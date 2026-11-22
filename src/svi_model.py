import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from src import config


def svi_raw(k, a, b, rho, m, sigma):
    """
    Gatheral's Raw SVI (Stochastic Volatility Inspired) total implied variance:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Parameters
    ----------
    k : float or np.ndarray
        Log-moneyness: k = ln(K / F)
    a, b, rho, m, sigma : float
        SVI parameters:
        a: vertical shift (variance level)
        b: wing angle (curvature)
        rho: smile orientation / skew (-1 < rho < 1)
        m: horizontal translation of smile minimum
        sigma: smoothness of the ATM vertex (sigma > 0)

    Returns
    -------
    np.ndarray
        Total implied variance w(k) = sigma_IV^2 * T
    """
    k = np.asarray(k, dtype=float)
    disc = np.sqrt((k - m)**2 + sigma**2)
    return a + b * (rho * (k - m) + disc)


def svi_implied_vol(k, T, a, b, rho, m, sigma):
    """
    Converts SVI total variance w(k) to annualized implied volatility:
        sigma_IV = sqrt(max(w(k), 1e-6) / T)
    """
    w = svi_raw(k, a, b, rho, m, sigma)
    w_safe = np.maximum(w, 1e-6)
    T_safe = np.maximum(float(T), 1e-6)
    return np.sqrt(w_safe / T_safe)


def check_svi_arbitrage_bounds(a, b, rho, m, sigma, T):
    """
    Validates Gatheral's static no-arbitrage parameter bounds:
    1. b >= 0
    2. |rho| < 1
    3. sigma > 0
    4. a + b * sigma * sqrt(1 - rho^2) >= 0 (strictly positive total variance)
    5. b * (1 + |rho|) < 4 / T (Roger Lee's moment formula bound)
    """
    if b < 0:
        return False, "b must be non-negative"
    if abs(rho) >= 1.0:
        return False, "|rho| must be strictly less than 1"
    if sigma <= 0:
        return False, "sigma must be strictly positive"
    min_w = a + b * sigma * np.sqrt(max(1.0 - rho**2, 1e-6))
    if min_w < 0:
        return False, f"Minimum total variance is negative ({min_w:.4f} < 0)"
    if b * (1.0 + abs(rho)) >= 4.0 / max(float(T), 1e-4):
        return False, "Violates Roger Lee asymptotic moment bound"
    return True, "No arbitrage violations"


def check_butterfly_arbitrage(k_grid, T, a, b, rho, m, sigma):
    """
    Evaluates Breeden-Litzenberger risk-neutral density g(k):
        g(k) = (1 - k*w'/(2w))^2 - (w'^2 / 4)*(1/w + 1/4) + w''/2
    Arbitrage-free condition requires g(k) >= 0 across all k.
    """
    k = np.asarray(k_grid, dtype=float)
    w = np.maximum(svi_raw(k, a, b, rho, m, sigma), 1e-6)
    
    # First derivative w'(k)
    disc = np.sqrt((k - m)**2 + sigma**2)
    w_prime = b * (rho + (k - m) / disc)
    
    # Second derivative w''(k)
    w_prime2 = b * (sigma**2 / (disc**3))
    
    # Breeden-Litzenberger g(k)
    term1 = (1.0 - (k * w_prime) / (2.0 * w))**2
    term2 = (w_prime**2 / 4.0) * (1.0 / w + 0.25)
    term3 = 0.5 * w_prime2
    
    g = term1 - term2 + term3
    is_arbitrage_free = bool(np.all(g >= -1e-4))
    min_density = float(np.min(g))
    return is_arbitrage_free, min_density, g


def calibrate_svi_slice(k_array, iv_array, T):
    """
    Calibrates SVI parameters (a, b, rho, m, sigma) for a single maturity slice.
    Minimizes Root Mean Squared Volatility Error (RMSVE) directly in IV space.

    Parameters
    ----------
    k_array : np.ndarray
        Array of log-moneyness values: k = ln(K / F)
    iv_array : np.ndarray
        Market implied volatilities (decimal, e.g. 0.65)
    T : float
        Maturity in fractional years.

    Returns
    -------
    dict
        Fitted parameters, RMSVE, and arbitrage verification flag.
    """
    k = np.asarray(k_array, dtype=float)
    iv = np.asarray(iv_array, dtype=float)
    T = max(float(T), 1e-5)
    w_market = (iv**2) * T

    atm_idx = np.argmin(np.abs(k))
    atm_w = float(w_market[atm_idx])
    
    # Well-scaled initial parameter heuristic
    x0 = [atm_w * 0.8, 0.02, -0.2, 0.0, 0.05]
    # Bound a based on maturity T to prevent negative total variance
    bounds = (
        [-2.0 * T - 0.01, 0.0001, -0.999, -1.0, 0.0001],
        [5.0,            3.0,     0.999,  1.0, 2.0]
    )

    res = least_squares(
        lambda p: svi_implied_vol(k, T, *p) - iv,
        x0=x0,
        bounds=bounds,
        loss='soft_l1',
        f_scale=0.02,
        max_nfev=1500
    )

    a_opt, b_opt, rho_opt, m_opt, sigma_opt = res.x
    iv_fitted = svi_implied_vol(k, T, a_opt, b_opt, rho_opt, m_opt, sigma_opt)
    rmsve = float(np.sqrt(np.mean((iv - iv_fitted)**2)))

    fine_k = np.linspace(min(k.min() - 0.1, -0.5), max(k.max() + 0.1, 0.5), 150)
    is_arb_free, min_g, _ = check_butterfly_arbitrage(fine_k, T, a_opt, b_opt, rho_opt, m_opt, sigma_opt)

    return {
        "params": {
            "a": float(a_opt),
            "b": float(b_opt),
            "rho": float(rho_opt),
            "m": float(m_opt),
            "sigma": float(sigma_opt),
        },
        "rmsve": rmsve,
        "is_arbitrage_free": is_arb_free,
        "min_density": min_g,
        "T": T,
        "n_points": len(k)
    }


def calibrate_svi_surface(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Calibrates SVI smiles across all active maturity slices in the dataset.
    Adds 'svi_iv' and 'svi_price' to the input DataFrame.
    """
    df = df.copy()
    df['svi_iv'] = np.nan
    df['svi_price'] = np.nan

    calibrated_models = {}
    expiries = df['expiry_str'].unique() if 'expiry_str' in df.columns else df['T'].unique()

    from src.baseline import vectorized_black_76

    for exp in expiries:
        mask = (df['expiry_str'] == exp) if 'expiry_str' in df.columns else (df['T'] == exp)
        sub = df[mask]
        if len(sub) < 4:
            df.loc[mask, 'svi_iv'] = sub['implied_vol'].median()
            continue

        T_val = sub['T'].iloc[0]
        k_vals = sub['log_moneyness'].values
        iv_vals = sub['implied_vol'].values

        fit_result = calibrate_svi_slice(k_vals, iv_vals, T_val)
        p = fit_result['params']
        calibrated_models[str(exp)] = fit_result

        # Predict SVI IV
        fitted_iv = svi_implied_vol(k_vals, T_val, p['a'], p['b'], p['rho'], p['m'], p['sigma'])
        df.loc[mask, 'svi_iv'] = fitted_iv

        # Predict Black-76 SVI Option Price
        call_mask = mask & (df['option_type'].str.lower() == 'call')
        put_mask = mask & (df['option_type'].str.lower() == 'put')

        if call_mask.any():
            c_sub = df[call_mask]
            df.loc[call_mask, 'svi_price'] = vectorized_black_76(
                F=c_sub['underlying_price'].values,
                K=c_sub['strike'].values,
                T=c_sub['T'].values,
                sigma=df.loc[call_mask, 'svi_iv'].values,
                option_type='call'
            )
        if put_mask.any():
            p_sub = df[put_mask]
            df.loc[put_mask, 'svi_price'] = vectorized_black_76(
                F=p_sub['underlying_price'].values,
                K=p_sub['strike'].values,
                T=p_sub['T'].values,
                sigma=df.loc[put_mask, 'svi_iv'].values,
                option_type='put'
            )

    return calibrated_models, df
