# GSLC 2 — Bitcoin Price Forecasting with Recurrent Neural Networks

## Project Overview

This repository contains the submission for **GSLC (Graded Structured Learning Cycle) 2**, a Deep Learning assignment by **Adhika Gunawan (2802438205)**.

The core task (Question 2) builds and evaluates recurrent neural network architectures to forecast **Bitcoin (BTC-USD) daily log returns / closing price** from historical OHLCV data. Two architectures are trained and compared:

1. **Baseline — Stacked GRU**: a 2-layer stacked GRU (32 → 16 units) with dropout and L2 regularization.
2. **Modified — Bidirectional GRU with Self-Attention**: a deeper architecture using stacked `Bidirectional(GRU)` layers, Batch Normalization, Swish activations, and a temporal self-attention layer to reduce recency bias.

The notebook also answers conceptual Deep Learning questions on **Object Detection vs. Image Segmentation** and **Semantic vs. Instance vs. Panoptic Segmentation** (Questions 4–5).

### Key Findings

- BTC price action is **trend-dominated** (STL Strength of Trend ≈ 0.88) with very weak seasonality (≈ 0.13), which motivated the choice of a sequential/recurrent architecture over classical time-series models.
- Both GRU models struggle to beat a naive predictor on **log-return R²** (negative on the test set), confirming that BTC daily returns are close to market-efficient noise.
- The **BiGRU + Attention** model improved **Directional Accuracy** from 49.4% (baseline, worse than a coin flip) to 54.4%, but neither model can anticipate price reversals — both are lagging, reactive predictors.
- Recommendation for future work: incorporate external signals (sentiment/NLP, macroeconomic indicators, on-chain data) since price-only historical features appear to have hit a performance ceiling.

## Dataset Description

- **Source**: Yahoo Finance (`yfinance`), ticker `BTC-USD`, daily interval.
- **File**: `btc_daily.csv` — cached local copy of the OHLCV data used in the notebook (`Open`, `High`, `Low`, `Close`, `Volume`, indexed by `Date`).
- **Date range**: 2021-06-07 to 2026-05-07.
- **Engineered features** (computed from OHLCV in the notebook):

  | Feature | Description |
  |---|---|
  | `log_return` | Target variable — `ln(C_t / C_{t-1})` |
  | `rsi_14` | 14-day Wilder Relative Strength Index |
  | `bb_pos` | Position of close price within Bollinger Bands (0–1) |
  | `norm_range` | Normalized daily volatility `(High - Low) / Close` |
  | `lag1`, `lag2`, `lag3` | Lagged log returns (short-term momentum) |
  | `vol_pct` | Percentage change in trading volume |

## Methodology & Implementation

1. **EDA**: raw OHLCV visualization, STL decomposition (trend/seasonal/residual), monthly/weekly seasonality profiles, Pearson correlation matrix, Augmented Dickey-Fuller (ADF) stationarity test on log returns.
2. **Feature Engineering**: technical indicators (RSI, Bollinger Band position), normalized volatility, momentum lags, and volume change — all designed to be stationary and price-scale-invariant.
3. **Scaling**: `RobustScaler` (median/IQR-based), chosen because the engineered features are leptokurtic / fat-tailed with significant outliers.
4. **Windowing**: sliding window of `SEQ_LEN = 30` days used to predict the next-day log return.
5. **Splitting**: chronological (no shuffling) 70% train / 15% validation / 15% test split to avoid look-ahead bias.
6. **Modeling**:
   - Baseline: `Sequential` Stacked GRU (32 → 16 units, dropout 0.25, L2 regularization), Adam optimizer, MSE loss.
   - Modified: Functional-API Bidirectional GRU (128 → 64 units) + Batch Normalization + Dropout + a self-attention layer over the 30-day window + Swish-activated dense layers, trained with Huber loss.
7. **Callbacks**: `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` (best weights saved to `best_gru.weights.h5` and `bi_gru.weights.h5`).
8. **Evaluation**: R² (on both log return and reconstructed price), Directional Accuracy, RMSE, MAE, MAPE, and visual actual-vs-predicted plots on the test set.

## Project Structure

```
GSLC 2/
├── README.md                                   # This file
├── gslc.ipynb                                   # Main notebook (EDA, preprocessing, both models, evaluation, conceptual Q4-Q5)
├── gslc copy.ipynb                              # Working copy / backup of the main notebook
├── btc_gru_no_overfit.ipynb                     # Exploratory notebook — overfitting mitigation experiments
├── btc_daily.csv                                # Cached BTC-USD daily OHLCV dataset
├── best_gru.weights.h5                          # Saved weights — baseline Stacked GRU
├── bi_gru.weights.h5                             # Saved weights — modified BiGRU + Attention model
├── eda_raw.png, eda_stl.png, eda_pacf.png,
│   eda_seasonality.png, eda_corr.png             # EDA visualizations
├── feature_dist.png, feature_boxplot.png         # Feature engineering / scaling diagnostics
├── split_viz.png                                 # Train/val/test split visualization
├── training_curves.png                           # Loss/MAE training curves
├── eval_pred.png                                 # Actual vs. predicted price on test set
├── Deep Learning Report.pdf                      # Written report accompanying the assignment
├── gslc.zip                                      # Archived submission bundle
└── GSLC-2-Adhika-Gunawan-2802438205/             # Final packaged submission (notebook + report)
    ├── gslc.ipynb
    └── Deep Learning Report.pdf
```

## How to Run / Requirements

### Prerequisites

- Python 3.9+
- Jupyter Notebook / JupyterLab
- (Optional but recommended) a CUDA-capable GPU for faster training

### Install dependencies

```bash
pip install numpy pandas yfinance matplotlib seaborn statsmodels scikit-learn tensorflow
```

### Run

1. Open `gslc.ipynb` in Jupyter.
2. Run all cells top to bottom — the notebook will:
   - download/refresh BTC-USD data via `yfinance` (or reuse `btc_daily.csv`),
   - run the EDA and preprocessing pipeline,
   - train the baseline Stacked GRU and the modified BiGRU + Attention model,
   - save the best weights to `best_gru.weights.h5` / `bi_gru.weights.h5`,
   - produce the evaluation plots and metrics shown in the report.
3. To reload a trained model without retraining, rebuild the corresponding architecture function in the notebook and call `model.load_weights('best_gru.weights.h5')` (or `bi_gru.weights.h5`).
