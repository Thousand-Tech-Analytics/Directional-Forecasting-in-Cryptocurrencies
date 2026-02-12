# Directional Forecasting in Cryptocurrencies  
Minute-level price direction prediction using feature-engineered time-series machine learning

---

## Overview
This project develops a leakage-safe machine learning pipeline to predict whether the price of a cryptocurrency will move **up (1)** or **not up (0)** in the next minute using minute-level OHLCV and trading activity data.

The objective is to build a reproducible end-to-end modeling workflow including feature engineering, time-series validation, model training, threshold optimization, and final inference artifact generation.

---

## Dataset
The dataset contains minute-level historical trading information including:

- Open, high, low, close prices  
- Volume and quote asset volume  
- Number of trades  
- Taker buy base / quote volumes  
- Target: next-minute price direction

All features are constructed using **past and current timestamps only** to avoid look-ahead leakage.

---

## Methodology

### 1. Leakage-Safe Time-Series Pipeline
- Chronological ordering and validation splitting (80/20 time-based)
- Feature engineering using lagged and rolling statistics
- Reproducible inference pipeline for full-length test predictions

### 2. Feature Engineering
Three feature versions were developed:

- **v2:** multi-horizon returns, rolling volatility, volume momentum  
- **v3:** microstructure features (taker buy ratios, trade intensity, quote-per-trade)  
- **v3.1:** lagged microstructure signals for short-term order-flow persistence

### 3. Modeling
- LightGBM gradient boosting classifier
- Threshold optimization for Macro-F1 score
- Versioned model comparison for final selection

---

## Results

We evaluated multiple feature-engineered model versions using a leakage-safe time-based validation framework. Performance was measured using the Macro-F1 metric to balance predictive quality across both classes.

| Model Version | Feature Count | Best Threshold | Macro-F1 |
|---|---:|---:|---:|
| v2 | 15 | 0.48 | 0.5116 |
| v3 | 35 | 0.48 | 0.5141 |
| v3.1 | 39 | 0.48 | **0.5145** |

The introduction of microstructure-derived features (taker buy ratios, trade intensity, and their lagged variants) provided consistent improvements over the baseline feature set. The final selected configuration (v3.1) achieved the best validation performance and was used to generate the final submission artifacts.

---

## Final Artifacts
- `final_results_summary.csv` — model comparison table  
- `submission_final_v3_1_lgb_thr048_full.csv` — final reproducible predictions  

---

## How to Run

1. Load training and test datasets  
2. Run feature engineering pipelines (v2 → v3 → v3.1)  
3. Train the LightGBM model  
4. Perform threshold tuning  
5. Generate final submission file

All steps are implemented in the accompanying Kaggle notebook and can be executed sequentially to reproduce the results.

---

## Key Takeaways
- Microstructure-aware features improved short-horizon directional forecasting performance.
- Threshold optimization significantly affected Macro-F1 performance.
- A reproducible time-series validation pipeline is critical for reliable evaluation in high-frequency financial prediction tasks.
