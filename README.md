# BTC Options Pricing: Black-76, SVI Surface & Hybrid Engine

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Exchange](https://img.shields.io/badge/exchange-Deribit-orange.svg)](https://www.deribit.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A quantitative derivatives pricing and volatility surface engine for Bitcoin (BTC) options traded on Deribit.

---

## Overview

The engine addresses key challenges in crypto options pricing through a multi-tiered framework:
- **Deribit-Native Black-76 ($r = 0$):** Handles inverse coin-margining and forward-referenced payoffs without cash discounting errors.
- **Arbitrage-Free SVI Surface:** Sequential constrained calibration of Jim Gatheral's Raw SVI model enforcing butterfly convexity, Roger Lee moment bounds, and calendar monotonicity.
- **Microstructure Residual Regressor (XGBoost):** Evaluates order book features (depth imbalance, relative spread, open interest) on strict out-of-sample holdouts in implied volatility space.
- **4-Tier Quantitative Benchmark:** Compares naive Black-Scholes against Black-76, parametric SVI, and the hybrid ML engine.

---

## 1. Core Methodology

### Black-76 for Coin-Margined Options
Deribit options settle in Bitcoin against the forward price $F$. With $r = 0$:
$$C = F \cdot N(d_1) - K \cdot N(d_2), \quad P = K \cdot N(-d_2) - F \cdot N(-d_1)$$
$$d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2 T}{\sigma \sqrt{T}}, \quad d_2 = d_1 - \sigma \sqrt{T}$$
- Premium in BTC: $C_{\text{BTC}} = C / F$.
- Forward Put-Call Parity: $C - P = F - K$.

### SVI Volatility Smile Calibration
Total implied variance $w(k, \tau) = \sigma^2 \tau$ as a function of log-moneyness $k = \ln(K / F)$:
$$w(k) = a + b \left( \rho (k - m) + \sqrt{(k - m)^2 + \sigma^2} \right)$$
- **$a$:** Overall variance level
- **$b$:** Wing slope / curvature ($b \ge 0$)
- **$\rho$:** Skewness / spot-vol correlation ($|\rho| < 1$)
- **$m$:** Smile horizontal shift
- **$\sigma$:** Smoothness at the money ($\sigma > 0$)
- **No-Arbitrage Checks ($k \in [-1.0, +1.0]$):** Verified across all 11 slices for Breeden-Litzenberger density ($g(k) \ge 0$), Roger Lee bounds, and calendar monotonicity ($w_{i+1} \ge w_i$).

### Machine Learning Layer (XGBoost)
Predicts the implied volatility residual $\Delta \sigma = \sigma_{\text{market}} - \sigma_{\text{SVI}}$ using order book imbalance, relative bid-ask spread, open interest, log-moneyness $k$, and maturity $T$, evaluated on a stratified 70/15/15 holdout split.

---

## 2. Benchmark Results

Evaluated on **556 Deribit BTC options** across **11 expiration dates** (1-day to 1-year). Dollar pricing is benchmarked against Deribit's official mark price (`mark_usd`) since SVI calibrates to `mark_iv` (BBO `market_mid` RMSE is `$137.88` due to wide spreads on deep-ITM strikes).

### Strict Out-of-Sample Holdout ($N = 89$, Stratified Test Set)

| Tier | Model Architecture | Volatility RMSVE | Mark Price RMSE | Mark MAPE | No-Arbitrage Audit |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Tier 0** | Naive Black-Scholes ($r=0.05, \sigma=40\%$) | 29.29% | \$1,591.47 | 55.84% | Violates forward parity |
| **Tier 1** | Black-76 with Flat ATM Volatility | 13.38% | \$204.54 | 19.28% | No smile skew |
| **Tier 2** | **SVI Volatility Surface (Parametric)** | **0.82%** | **\$17.04** | **1.27%** | **11/11 Butterfly, 0 Calendar Viol.** |
| **Tier 3** | **Full Hybrid Engine (SVI + XGBoost OOS)** | **0.82%** | **\$20.75** | **1.36%** | **No formal guarantee** |

*(Full-surface $N = 556$ Tier 2 SVI calibration: RMSVE `1.41%`, Mark RMSE `$14.32`.)*

---

## 3. Key Finding: Occam's Razor in Pricing

- **The Strawman Baseline:** Benchmarking ML against naive Black-Scholes ($\sigma=40\%, r=0.05$) creates an illusion of massive error reduction where trees mostly approximate a Taylor expansion of volatility level.
- **SVI Dominance:** Gatheral's 5-parameter SVI model with Black-76 ($r=0$) captures the volatility surface natively (**\$14.32** full-surface mark RMSE, **\$17.04** OOS) with **0 calendar or butterfly violations**.
- **XGBoost Out-of-Sample Reality:** On a leakage-free stratified holdout, XGBoost early-stopping halts at iteration 0 and slightly degrades OOS pricing (**\$17.04** to **\$20.75** Mark RMSE).
- **Takeaway:** In cross-sectional derivatives pricing, enforcing domain physics (forward discounting and arbitrage-free parametric geometry) strictly outperforms unconstrained gradient boosting.

---

## 4. Repository Structure

```
BTC_Option_Pricing/
├── pyproject.toml              # Package & pytest configuration
├── requirements.txt            # Compatible Python 3.10+ dependencies
├── main.py                     # End-to-end execution pipeline
├── data/samples/               # Tracked 08:00 UTC BTC snapshot for fresh clones
├── src/
│   ├── baseline.py             # Black-76 (r=0) & Black-Scholes pricing
│   ├── config.py               # Paths, SVI bounds & ML hyperparameters
│   ├── data_fetcher.py         # Tardis.dev dataset downloader
│   ├── data_loader.py          # BTC-only snapshot extractor & sample fallback
│   ├── svi_model.py            # Constrained SVI calibration & arbitrage checks
│   ├── model.py                # Stratified split & XGBoost residual trainer
│   ├── sanity_checks.py        # Benchmark metrics & verified arbitrage audit
│   └── visualize.py            # Diagnostic plotting
├── tests/
│   ├── test_baseline.py        # Parity & Greeks unit tests
│   ├── test_svi.py             # SVI bounds, density & calendar tests
│   └── test_pipeline.py        # Loader, split, XGBoost & audit tests
└── figures/
    ├── 01_svi_volatility_smiles.png
    ├── 02_microstructure_feature_importance.png
    └── 03_four_tier_benchmark_comparison.png
```

---

## 5. Quickstart

### Installation
```powershell
pip install -r requirements.txt
```

### Run Unit Tests
```powershell
pytest
```
Expected: **16 passed** in under 5 seconds.

### Run Pipeline
```powershell
python main.py
```
Loads the clean 08:00 UTC BTC snapshot (automatically falling back to `data/samples/` on fresh clones), calibrates SVI across 11 expiries, runs strict OOS XGBoost evaluation, verifies arbitrage bounds, and exports figures to `figures/`.

---

## 6. Visual Diagnostics

1. **`01_svi_volatility_smiles.png`:** Arbitrage-free SVI smiles vs. Deribit mark IV across 11 expiries.
2. **`02_microstructure_feature_importance.png`:** XGBoost gain attribution across features.
3. **`03_four_tier_benchmark_comparison.png`:** Out-of-sample RMSVE and Mark-Price RMSE across all 4 tiers.

---

## References
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization*.
- Black, F. (1976). *The pricing of commodity contracts*. J. Financial Economics.
- Breeden, D. T., & Litzenberger, R. H. (1978). *State-contingent claims implicit in option prices*.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
