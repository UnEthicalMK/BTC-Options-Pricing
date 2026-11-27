import numpy as np
import pandas as pd
from scipy.optimize import least_squares, minimize
from src import config
from src.baseline import price_options_dataframe


def svi_raw(k, a, b, rho, m, sigma):
    """
    Gatheral's Raw SVI total implied variance:
        w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    """
    k = np.asarray(k, dtype=float)
    disc = np.sqrt((k - m) ** 2 + sigma ** 2)
    return a + b * (rho * (k - m) + disc)


def svi_implied_vol(k, T, a, b, rho, m, sigma):
    """
    Converts SVI total variance w(k) to annualized implied volatility:
        sigma_IV = sqrt(max(w(k), 1e-6) / T)
    Supports both scalar and array-like inputs for k and T.
    """
    w = svi_raw(k, a, b, rho, m, sigma)
    w_safe = np.maximum(w, 1e-6)
    T_safe = np.maximum(np.asarray(T, dtype=float), 1e-6)
    return np.sqrt(w_safe / T_safe)


def check_svi_arbitrage_bounds(a, b, rho, m, sigma, T):
    """
    Validates Gatheral's static parameter bounds and Roger Lee's moment bound:
    1. b >= 0, |rho| < 1, sigma > 0
    2. a + b * sigma * sqrt(1 - rho^2) >= 0 (non-negative total variance)
    3. b * (1 + |rho|) < 4 / T (Roger Lee asymptotic slope bound)
    """
    if b < 0:
        return False, "b must be non-negative"
    if abs(rho) >= 1.0:
        return False, "|rho| must be strictly less than 1"
    if sigma <= 0:
        return False, "sigma must be strictly positive"
    min_w = a + b * sigma * np.sqrt(max(1.0 - rho ** 2, 1e-6))
    if min_w < 0:
        return False, f"Minimum total variance is negative ({min_w:.4f} < 0)"
    if b * (1.0 + abs(rho)) >= 4.0 / max(float(T), 1e-4):
        return False, "Violates Roger Lee asymptotic moment bound"
    return True, "No arbitrage violations"


def check_butterfly_arbitrage(k_grid, T, a, b, rho, m, sigma):
    """
    Evaluates Breeden-Litzenberger risk-neutral density factor g(k):
        g(k) = (1 - k*w'/(2w))^2 - (w'^2 / 4)*(1/w + 1/4) + w''/2
    Strict no-butterfly-arbitrage requires g(k) >= 0 across all k.
    """
    k = np.asarray(k_grid, dtype=float)
    w = np.maximum(svi_raw(k, a, b, rho, m, sigma), 1e-6)

    disc = np.sqrt((k - m) ** 2 + sigma ** 2)
    w_prime = b * (rho + (k - m) / disc)
    w_prime2 = b * (sigma ** 2 / (disc ** 3))

    term1 = (1.0 - (k * w_prime) / (2.0 * w)) ** 2
    term2 = (w_prime ** 2 / 4.0) * (1.0 / w + 0.25)
    term3 = 0.5 * w_prime2

    g = term1 - term2 + term3
    min_density = float(np.min(g))
    return bool(min_density >= 0.0), min_density, g


def check_calendar_arbitrage(calibrated_models: dict, k_grid: np.ndarray = None):
    """
    Checks calendar spread monotonicity across sorted expiries (T_1 < T_2 < ...):
        w(k, T_{i+1}) >= w(k, T_i)  for all k in k_grid.

    Returns
    -------
    tuple[bool, float, list[dict]]
        (is_calendar_free, min_variance_diff, violation_records)
    """
    if k_grid is None:
        k_grid = np.linspace(config.SVI_ARB_K_GRID_MIN, config.SVI_ARB_K_GRID_MAX, 121)

    ordered = sorted(calibrated_models.items(), key=lambda item: item[1]["T"])
    prev_exp = None
    prev_w = None
    min_diff = 0.0
    violations = []

    for exp, res in ordered:
        p = res["params"]
        cur_w = svi_raw(k_grid, p["a"], p["b"], p["rho"], p["m"], p["sigma"])
        if prev_w is not None:
            diff = float(np.min(cur_w - prev_w))
            min_diff = min(min_diff, diff) if violations or prev_exp != ordered[0][0] else diff
            if diff < 0.0:
                violations.append({
                    "prev_expiry": prev_exp,
                    "expiry": exp,
                    "min_diff": diff,
                })
        prev_exp = exp
        prev_w = cur_w

    return len(violations) == 0, float(min_diff), violations


def calibrate_svi_slice(k_array, iv_array, T, prev_w_grid=None, k_grid=None):
    """
    Calibrates SVI parameters (a, b, rho, m, sigma) for a single maturity slice.
    Uses robust least-squares initialization followed by SLSQP constraint polish
    when needed to enforce strict butterfly, Roger Lee, and calendar bounds.
    """
    k = np.asarray(k_array, dtype=float)
    iv = np.asarray(iv_array, dtype=float)
    T = max(float(T), 1e-5)
    w_market = (iv ** 2) * T

    if k_grid is None:
        k_grid = np.linspace(config.SVI_ARB_K_GRID_MIN, config.SVI_ARB_K_GRID_MAX, 121)

    atm_idx = np.argmin(np.abs(k))
    atm_w = float(w_market[atm_idx])

    x0 = [atm_w * 0.8, 0.02, -0.2, 0.0, 0.05]
    bounds = (
        [-2.0 * T - 0.01, 0.0001, -0.999, -1.0, 0.0001],
        [5.0,             3.0,     0.999,  1.0, 2.0],
    )

    res = least_squares(
        lambda p: svi_implied_vol(k, T, *p) - iv,
        x0=x0,
        bounds=bounds,
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=1500,
    )
    opt_p = res.x

    # Verify static and cross-slice arbitrage conditions across k_grid
    is_bfly, min_g, _ = check_butterfly_arbitrage(k_grid, T, *opt_p)
    rl_ok, _ = check_svi_arbitrage_bounds(*opt_p, T)
    cal_ok = True
    if prev_w_grid is not None:
        cal_ok = bool(np.min(svi_raw(k_grid, *opt_p) - prev_w_grid) >= 0.0)

    if not (is_bfly and rl_ok and cal_ok):
        cons = [
            {"type": "ineq", "fun": lambda p: check_butterfly_arbitrage(k_grid, T, *p)[1] - 1e-5},
            {"type": "ineq", "fun": lambda p: p[0] + p[1] * p[4] * np.sqrt(max(1.0 - p[2] ** 2, 1e-6)) - 1e-6},
            {"type": "ineq", "fun": lambda p: (4.0 / max(T, 1e-4)) - p[1] * (1.0 + abs(p[2])) - 1e-4},
        ]
        if prev_w_grid is not None:
            pw = prev_w_grid.copy()
            cons.append({"type": "ineq", "fun": lambda p, pw=pw: np.min(svi_raw(k_grid, *p) - pw) - 1e-6})

        res_c = minimize(
            lambda p: np.mean((svi_implied_vol(k, T, *p) - iv) ** 2),
            x0=opt_p,
            method="SLSQP",
            bounds=list(zip(bounds[0], bounds[1])),
            constraints=cons,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        opt_p = res_c.x
        is_bfly, min_g, _ = check_butterfly_arbitrage(k_grid, T, *opt_p)
        rl_ok, _ = check_svi_arbitrage_bounds(*opt_p, T)

    a_opt, b_opt, rho_opt, m_opt, sigma_opt = opt_p
    iv_fitted = svi_implied_vol(k, T, a_opt, b_opt, rho_opt, m_opt, sigma_opt)
    rmsve = float(np.sqrt(np.mean((iv - iv_fitted) ** 2)))

    return {
        "params": {
            "a": float(a_opt),
            "b": float(b_opt),
            "rho": float(rho_opt),
            "m": float(m_opt),
            "sigma": float(sigma_opt),
        },
        "rmsve": rmsve,
        "is_arbitrage_free": bool(is_bfly and rl_ok),
        "is_butterfly_free": is_bfly,
        "is_roger_lee_valid": rl_ok,
        "min_density": min_g,
        "T": T,
        "n_points": len(k),
    }


def calibrate_svi_surface(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """
    Calibrates SVI smiles sequentially from shortest to longest maturity,
    enforcing butterfly and calendar arbitrage constraints across the grid.
    """
    df = df.copy()
    df["svi_iv"] = np.nan

    calibrated_models = {}
    group_col = "expiry_str" if "expiry_str" in df.columns else "T"
    expiries = df.groupby(group_col)["T"].first().sort_values().index.tolist()

    k_grid = np.linspace(config.SVI_ARB_K_GRID_MIN, config.SVI_ARB_K_GRID_MAX, 121)
    prev_w_grid = None

    for exp in expiries:
        mask = df[group_col] == exp
        sub = df[mask]
        if len(sub) < 4:
            df.loc[mask, "svi_iv"] = sub["implied_vol"].median()
            continue

        T_val = float(sub["T"].iloc[0])
        k_vals = sub["log_moneyness"].values
        iv_vals = sub["implied_vol"].values

        fit_result = calibrate_svi_slice(k_vals, iv_vals, T_val, prev_w_grid=prev_w_grid, k_grid=k_grid)
        p = fit_result["params"]
        calibrated_models[str(exp)] = fit_result

        df.loc[mask, "svi_iv"] = svi_implied_vol(
            k_vals, T_val, p["a"], p["b"], p["rho"], p["m"], p["sigma"]
        )
        prev_w_grid = svi_raw(k_grid, p["a"], p["b"], p["rho"], p["m"], p["sigma"])

    df["svi_price"] = price_options_dataframe(df, df["svi_iv"].values, model="black_76")
    return calibrated_models, df
