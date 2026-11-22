import numpy as np
from scipy.stats import norm


def vectorized_black_76(F, K, T, sigma, option_type="call"):
    """
    Computes analytical European option prices under the Black-76 model (r = 0),
    matching Deribit's inverse coin-margined forward pricing conventions.

    Parameters
    ----------
    F : float or np.ndarray
        Underlying forward/futures price for the option expiry.
    K : float or np.ndarray
        Strike price of the option contract.
    T : float or np.ndarray
        Time to maturity in fractional years.
    sigma : float or np.ndarray
        Annualized volatility (decimal, e.g. 0.65 for 65%).
    option_type : str
        'call' or 'put'.

    Returns
    -------
    np.ndarray
        Option price in USD. (Divide by F to convert to BTC premium).
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
    Computes forward Delta, Gamma, and Vega under Black-76 (r=0).

    Returns
    -------
    delta, gamma, vega : tuple of np.ndarray
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


# -------------------------------------------------------------------------
# Legacy Black-Scholes implementation (retained for baseline comparisons)
# -------------------------------------------------------------------------
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


def calculate_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Classical spot Black-Scholes Greeks with cash interest rate.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), 1e-5)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-5)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    pdf_d1 = norm.pdf(d1)
    gamma = pdf_d1 / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * pdf_d1

    otype = option_type.lower()
    if otype == "call":
        delta = norm.cdf(d1)
    elif otype == "put":
        delta = norm.cdf(d1) - 1.0
    else:
        raise ValueError("option_type must be either 'call' or 'put'")

    return delta, gamma, vega
