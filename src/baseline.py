import numpy as np
import pandas as pd
from scipy.stats import norm


def vectorized_black_76(F, K, T, sigma, option_type="call"):
    """
    Computes analytical European option prices under the Black-76 model (r = 0),
    matching Deribit's inverse coin-margined forward pricing conventions.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-4)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    otype = option_type.lower()
    if otype == "call":
        price = F * norm.cdf(d1) - K * norm.cdf(d2)
    elif otype == "put":
        price = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return np.maximum(price, 0.0)


def calculate_black_76_greeks(F, K, T, sigma, option_type="call"):
    """
    Computes forward Delta, Gamma, and Vega under Black-76 (r = 0).
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-6)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-4)

    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))

    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (F * sigma * np.sqrt(T))
    vega = F * np.sqrt(T) * pdf_d1

    otype = option_type.lower()
    if otype == "call":
        delta = norm.cdf(d1)
    elif otype == "put":
        delta = norm.cdf(d1) - 1.0
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return delta, gamma, vega


def vectorized_black_scholes(S, K, T, r, sigma, option_type="call"):
    """
    Classical Black-Scholes spot formula with cash discounting (r > 0).
    Retained for Tier 0 naive benchmark comparison.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-5)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-5)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    otype = option_type.lower()
    if otype == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif otype == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return np.maximum(price, 0.0)


def price_options_dataframe(
    df: pd.DataFrame,
    sigma: np.ndarray | float,
    model: str = "black_76",
    r: float = 0.0,
) -> np.ndarray:
    """
    Prices mixed call/put rows in an options DataFrame using a unified code path.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'underlying_price', 'strike', 'T', and 'option_type'.
    sigma : float or array-like
        Annualized volatility matching `df` length (or scalar).
    model : str
        'black_76' (default, r=0) or 'black_scholes' (legacy cash discounting).
    r : float
        Risk-free rate used only when model == 'black_scholes'.
    """
    sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), len(df))
    prices = np.empty(len(df), dtype=float)
    otypes = df["option_type"].str.lower()

    for otype in ("call", "put"):
        mask = (otypes == otype).values
        if not np.any(mask):
            continue
        sub = df.iloc[mask]
        if model == "black_76":
            prices[mask] = vectorized_black_76(
                F=sub["underlying_price"].values,
                K=sub["strike"].values,
                T=sub["T"].values,
                sigma=sigma_arr[mask],
                option_type=otype,
            )
        elif model == "black_scholes":
            prices[mask] = vectorized_black_scholes(
                S=sub["underlying_price"].values,
                K=sub["strike"].values,
                T=sub["T"].values,
                r=r,
                sigma=sigma_arr[mask],
                option_type=otype,
            )
        else:
            raise ValueError(f"Unsupported pricing model: {model}")

    return prices
