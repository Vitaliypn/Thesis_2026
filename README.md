# Crypto Asset Ranking — Thesis 2026

A machine learning system that predicts which crypto assets will **outperform
the market** over the next 7–30 days, using a multi-source dataset of ~200 coins
collected across 5 major exchanges.

---

## Problem Statement

Rather than predicting absolute price direction, this project frames the task as
a **cross-sectional ranking problem**: for each coin at each week-end, predict its
relative forward return compared to the full cohort.

Each coin is labelled into one of 5 buckets based on its percentile rank within
the cohort for that period:

| Label | Percentile | Meaning |
|-------|-----------|---------|
| `Strong Buy` | Top 10% (above 90th) | Expected strong outperformer |
| `Buy` | 70th – 90th | Expected outperformer |
| `Neutral` | 30th – 70th | Expected market-neutral |
| `Avoid` | 10th – 30th | Expected underperformer |
| `Strong Avoid` | Bottom 10% (below 10th) | Expected strong underperformer |

Labels are computed as **risk-adjusted forward returns** (Sharpe-style) relative
to the cohort — the model is always picking winners and losers, not betting on
a rising or falling market.

---

## Project Structure

```
Thesis_2026/
│
├── Building dataset/            — ETL pipeline: raw data → model-ready dataset
│   ├── add_top_assets.py        — Step 1: build asset universe from CoinGecko
│   ├── add_ohlcv.py             — Step 2: collect OHLCV history via ccxt
│   ├── add_missing_ohlcv.py     — Step 3: clean & deduplicate price data
│   ├── add_technical_features.py— Step 4: RSI, MACD, Bollinger, ATR, OBV, etc.
│   ├── add_market_features.py   — Step 5: market microstructure features
│   ├── add_new_features.py      — Step 6: market cap, TVL, calculated features
│   ├── add_derivatives.py       — Step 7: funding rates, open interest (CoinGlass)
│   ├── add_market_flags.py      — Step 8: binary flags (stablecoin, DeFi, cap tier)
│   ├── add_social_metrics.py    — Step 9: social volume & sentiment (LunarCrush)
│   ├── add_external_features.py — Step 10: halving cycle, Google Trends
│   ├── cleaning_dataset.py      — Step 11: final cleanup
│   ├── build_labels_and_crosssectional.py — Step 12: generate labels
│   ├── etl_config.py            — central config (exchanges, timeframes, etc.)
│   ├── utils_io.py              — shared helpers
│   └── README.md                — full pipeline documentation
│
└── Models/
    ├── Data/                    — model-ready datasets (gitignored, not in repo)
    ├── Modeling/                — baseline models and model comparison
    ├── Experiments/             — feature and strategy experiments
    ├── Shap analysis/           — SHAP feature importance analysis
    ├── Figures/                 — all experiment output plots (76 PNGs)
    └── README.md                — modeling documentation
```

---

## Data Sources

| Source | Data collected | API key |
|--------|---------------|---------|
| CoinGecko | Asset universe, market cap, rankings, metadata | Free |
| CCXT — Binance, Kraken, Bybit, KuCoin, OKX | OHLCV price history (5–8 years daily) | Free |
| alternative.me | Fear & Greed Index (full history since 2018) | Free |
| DefiLlama | TVL for DeFi protocols | Free |
| Google Trends (pytrends) | BTC & ETH search interest (weekly) | Free |
| CoinGlass V4 | Funding rates, open interest, liquidations | **Required** |
| LunarCrush | Social volume, sentiment, galaxy score, alt rank | **Required** |

---

## Features

~90 features across 6 categories:

| Category | Examples |
|----------|---------|
| **Technical** | RSI, MACD, Bollinger Bands, ATR, OBV, EMA 7/21/50/200, ADX, Stochastic |
| **Price-derived** | Returns 1d/7d/30d, volatility 30d, drawdown from 90d peak, ATH/ATL distance |
| **Market structure** | Market cap, coin age, cap tier flags, circulating supply ratio |
| **Derivatives** | Funding rate, 7d avg funding, 30d cumulative cost, open interest USD |
| **Social** | Galaxy score, alt rank, social volume, sentiment (LunarCrush) |
| **Macro / external** | Fear & Greed index, BTC dominance, global market cap, halving cycle phase, Google Trends |

Cross-sectional rank and z-score variants are computed for key features at each time step.

---

## Models & Results

Four tree-based models trained on weekly snapshots with a temporal train/test split (cutoff: 2024-01):

| Model | Accuracy | Macro F1 | Lift | Avg Spread | Win Rate | Ann. Spread |
|-------|---------|---------|------|-----------|---------|------------|
| **XGBoost** | **38.15%** | 18.94% | **+18.15pp** | **3.20%** | 66.3% | **166.5%** |
| Gradient Boosting | 36.13% | 19.71% | +16.13pp | 1.24% | 58.5% | 64.6% |
| Random Forest | 34.49% | **24.66%** | +14.49pp | 2.45% | **67.0%** | 127.3% |
| LightGBM | 29.74% | 23.71% | +9.74pp | 2.34% | 64.3% | 121.8% |

- **Accuracy** — 5-class classification accuracy (random baseline = 20%)
- **Lift** — accuracy improvement over random baseline
- **Avg Spread** — median weekly return of Strong Buy minus Strong Avoid
- **Win Rate** — % of weeks where the long-short spread was positive
- **Ann. Spread** — annualised long-short spread

---

## Top Features (by importance)

```
atr_pct              coin_age_days        price_vs_atl
price_vs_ath         volatility_30d       return_1d
atr_14               bb_width             rsi_14
return_7d            volume_vs_30d_avg    obv_divergence
price_vs_ema50       macd_hist            stoch_d
minus_di             drawdown_from_90d_peak  return_30d
price_vs_ema200      adx                  plus_di
ema_50_vs_200        galaxy_score_zscore  bb_pct
volume               coin_mcap_share_recalc  market_cap_usd_zscore
stoch_k              obv                  coin_mcap_share_recalc_rank
```

---

## Setup

```bash
# 1. Install dependencies
pip install ccxt pandas tqdm requests pytrends python-dotenv lunarcrush \
            scikit-learn xgboost lightgbm shap matplotlib

# 2. Configure API keys
cp .env.example .env
# Edit .env and fill in COINGLASS_API_KEY and LUNARCRUSH_API_KEY

# 3. Run the ETL pipeline
# See Building dataset/README.md for the full step-by-step guide

# 4. Run models
# Open notebooks in Models/Modeling/ and Models/Experiments/
```

---

## Repository Notes

- Large data files (`*.csv`, `*.pkl`, `*.h5`, `*.parquet`) are gitignored — not included in this repo
- To reproduce results, run the full ETL pipeline first to generate `data_for_experiments.csv`
- All scripts use relative paths and work from any machine without configuration
