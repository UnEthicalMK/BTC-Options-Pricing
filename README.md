# BTC Options Pricing: Black-76, SVI Surface & Hybrid Engine

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Exchange](https://img.shields.io/badge/exchange-Deribit-orange.svg)](https://www.deribit.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A quantitative derivatives pricing and volatility surface engine for Bitcoin (BTC) options traded on Deribit.

---

## Overview

The engine addresses key challenges in crypto options pricing through a multi-tiered framework:
- **Deribit-Native Black-76 ($r = 0$):** Handles inverse coin-margining and forward-referenced payoffs without cash discounting errors.
- **SVI Volatility Surface:** Implements Jim Gatheral's Quasi-Explicit SVI model to guarantee static no-arbitrage across all expiries.
- **Microstructure Residual Regressor (XGBoost):** Evaluates high-frequency order book features (depth imbalance, relative spread, open interest) in implied volatility space.
- **4-Tier Quantitative Benchmark:** Compares textbook models against parametric surfaces and machine learning.

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
- **No-Arbitrage Guarantee:** Checked via Breeden-Litzenberger density ($g(k) \ge 0$).

### Machine Learning Layer (XGBoost)
Predicts the implied volatility residual: $\Delta \sigma = \sigma_{\text{market}} - \sigma_{\text{SVI}}$.
- **Features:** Order book imbalance, relative bid-ask spread, open interest, moneyness $k$, and maturity $T$.

---

## 2. Benchmark Results

Evaluated on **556 Deribit BTC options** across **11 expiration dates** (1-day to 1-year):

| Tier | Model Architecture | Volatility RMSVE | Pricing RMSE | Pricing MAPE | No-Arbitrage Guarantee |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tier 0** | Naive Black-Scholes ($r=0.05, \sigma=40\%$) | 30.99% | \$1,726.34 | 59.55% | No (Violates parity & calendar) |
| **Tier 1** | Black-76 with Flat ATM Volatility | 15.55% | \$221.49 | 24.70% | Partial (No skew modeling) |
| **Tier 2** | **SVI Volatility Surface (Parametric)** | **1.08%** | **\$137.72** | **2.72%** | **100% Guaranteed** |
| **Tier 3** | **Full Hybrid Engine (SVI + XGBoost)** | **1.07%** | **\$137.67** | **2.72%** | **100% Manifold Bound** |

---

## 3. Key Finding: Occam's Razor in Pricing

- **The Strawman Baseline:** Training ML against a flawed Black-Scholes baseline ($\sigma=40\%, r=0.05$) creates an apparent "92% error reduction". In reality, the ML model is merely acting as a numerical multiplier to correct the baseline volatility level.
- **SVI Dominance:** Once Gatheral's 5-parameter SVI model is calibrated with Black-76 ($r=0$), it captures **99% of the real market smile natively** in milliseconds (RMSVE: 1.08%, MAPE: 2.72%).
- **XGBoost's Marginal Gain:** Adding XGBoost on top of SVI improves pricing by only **\$0.05** (\$137.72 to \$137.67).
- **Takeaway:** In cross-sectional derivatives pricing, domain-specific financial physics dominate unconstrained machine learning. ML should complement, not substitute, sound financial modeling.

---

## 4. Repository Structure

```
BTC_Option_Pricing/
├── pyproject.toml              # Pytest configuration
├── requirements.txt            # Dependencies
├── main.py                     # Execution pipeline
├── src/
│   ├── baseline.py             # Black-76 (r=0) & Black-Scholes pricing
│   ├── svi_model.py            # SVI calibration & arbitrage checks
│   ├── data_loader.py          # Clean 08:00 UTC snapshot extraction
│   ├── model.py                # XGBoost microstructure model
│   ├── sanity_checks.py        # 4-tier benchmark metrics
│   └── visualize.py            # Diagnostic plotting
├── tests/
│   ├── test_baseline.py        # Parity & Greeks unit tests
│   └── test_svi.py             # SVI bounds & density unit tests
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
Expected: **9 passed** in under 4 seconds.

### Run Pipeline
```powershell
python main.py
```
This loads the clean 08:00 UTC settlement snapshot, calibrates SVI across 11 expiries, trains the microstructure regressor, prints the benchmark report, and exports diagnostic plots to `figures/`.

---

## 6. Visual Diagnostics

High-resolution plots are exported to `figures/`:
1. **`01_svi_volatility_smiles.png`:** SVI fitted smiles vs. market quotes across all 11 expiries.
2. **`02_microstructure_feature_importance.png`:** Order book depth and spread attribution.
3. **`03_four_tier_benchmark_comparison.png`:** Multi-tier RMSVE and Dollar RMSE comparison.

---

## References
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization*.
- Black, F. (1976). *The pricing of commodity contracts*. J. Financial Economics.
- Breeden, D. T., & Litzenberger, R. H. (1978). *State-contingent claims implicit in option prices*.

---

## License
MIT License. See [LICENSE](LICENSE) for details.
