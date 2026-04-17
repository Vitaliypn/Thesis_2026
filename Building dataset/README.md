# Building dataset — README

Constructs a model-ready crypto dataset: OHLCV price history for the top-200
coins across 5 exchanges, enriched with technical indicators, market structure,
social metrics, derivatives, and external signals.

---

## Pipeline Overview

Run scripts in this order:

```
Step 1 — Asset list         add_top_assets.py
Step 2 — OHLCV collection   add_ohlcv.py
Step 3 — OHLCV cleaning     add_missing_ohlcv.py
Step 4 — Technical features add_technical_features.py
Step 5 — Market features    add_market_features.py
Step 6 — New features       add_new_features.py
Step 7 — Derivatives        add_derivatives.py
Step 8 — Market flags       add_market_flags.py
Step 9 — Social metrics     add_social_metrics.py
Step 10 — External features add_external_features.py
Step 11 — Dataset cleaning  cleaning_dataset.py
Step 12 — Labels            build_labels_and_crosssectional.py
```

---

## Files

### Data Collection

| File | Input | Output | Description |
|------|-------|--------|-------------|
| `add_top_assets.py` | CoinGecko API | `assets.csv` | Builds asset list from top-150 by market cap, historically from 2020 to today |
| `add_ohlcv.py` | `assets.csv` + exchanges | `ohlcv.csv`, `market_pairs.csv` | Fetches 5–8 years of daily OHLCV for USDT pairs across Binance, Kraken, Bybit, KuCoin, OKX |
| `add_missing_ohlcv.py` | `ohlcv.csv` | `ohlcv_clean.csv` | Removes delisted coins, deduplicates by exchange priority |

### Feature Engineering

| File | Input | Output | Description |
|------|-------|--------|-------------|
| `add_technical_features.py` | `ohlcv_clean.csv` | `technical_indicators.csv`, `fear_greed.csv`, `global_market.csv` | RSI, MACD, Bollinger Bands, EMA, ATR, OBV, Fear & Greed index, BTC dominance |
| `add_market_features.py` | `ohlcv_clean.csv` | `market_features.csv` | Market microstructure features via ccxt (no API key needed) |
| `add_new_features.py` | `dataset_final.csv` | `dataset_enriched.csv` | Calculated features, market cap/rank (CoinGecko), TVL for DeFi coins (DefiLlama) |
| `add_derivatives.py` | `dataset_enriched.csv` | `dataset_enriched.csv` | Funding rates, open interest, liquidations via CoinGlass V4 API |
| `add_market_flags.py` | `dataset_enriched.csv` | `dataset_flagged.csv` | Binary flags: is_stable, is_active, is_defi, cap tier, has_futures |
| `add_social_metrics.py` | `dataset_model.csv` | `dataset_with_social.csv` | Social volume, sentiment, galaxy score via LunarCrush API |
| `add_external_features.py` | `dataset_mcap_fixed.csv` | `dataset_features.csv` | Bitcoin halving cycle features, Google Trends (BTC/ETH search interest) |

### Dataset Assembly

| File | Input | Output | Description |
|------|-------|--------|-------------|
| `cleaning_dataset.py` | `dataset_clean.csv` | `dataset_final.csv` | Drops redundant columns, handles infinities and final cleanup |
| `build_labels_and_crosssectional.py` | `dataset_features.csv` | `model_dataset.csv` | Generates relative 30-day forward return labels (Strong Buy / Buy / Hold / Avoid) |

### Support

| File | Purpose |
|------|---------|
| `etl_config.py` | Central config (exchanges, timeframes, quote filter, lookback days) |
| `utils_io.py` | Shared helpers (CSV append, HTTP retries, symbol sanitization) |

---

## Quick Start

```bash
pip install ccxt pandas tqdm requests pytrends python-dotenv lunarcrush

# Step 1 — asset list
python add_top_assets.py

# Step 2 — OHLCV (7 years)
python add_ohlcv.py --days 2555

# Step 3 — clean OHLCV
python add_missing_ohlcv.py

# Steps 4–10 — feature engineering (run in order above)
python add_technical_features.py
python add_market_features.py
# ... etc.

# Step 12 — labels
python build_labels_and_crosssectional.py
```

---

## Configuration (`etl_config.py`)

| Setting | Default | Description |
|---------|---------|-------------|
| `CCXT_EXCHANGES` | binance, kraken, bybit, kucoin, okx | Exchanges to query |
| `DEFAULT_DAYS` | 2555 (7 years) | Default lookback |
| `MAX_DAYS` | 2920 (8 years) | Hard ceiling |
| `ALLOWED_QUOTES` | `{"USDT"}` | Only collect USDT pairs |
| `CANDLES_PER_REQUEST` | 1000 | Pagination page size |
| `OHLCV_TIMEFRAME` | `1d` | Candle interval |

---

## API Keys Required

Set these in a `.env` file (never commit it):

```
LUNARCRUSH_API_KEY=...    # add_social_metrics.py
COINGLASS_API_KEY=...     # add_derivatives.py
```

CoinGecko, DefiLlama, alternative.me (Fear & Greed), and pytrends are **free — no key needed**.

---

## Output: `model_dataset.csv`

Final model-ready dataset with one row per `(asset_id, date)`.

**Labels** (`label` column): `Strong Buy`, `Buy`, `Hold`, `Avoid` — relative to the
cross-sectional cohort each month-end based on 30-day forward risk-adjusted return.

**Expected size**: ~200 coins × 5 exchanges × 2555 days ≈ 2.5M rows before deduplication.

---

## Notes

- All scripts are **resumable** — they skip rows already on disk.
- Exchange history limits: Binance from ~2017, Kraken BTC from ~2016, Bybit from ~2019.
- For large-scale querying of the output CSVs, consider converting to Parquet or DuckDB.
