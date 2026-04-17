"""
add_technical_features.py
───────────────────────────
Collects all Priority 1 features in one run:

  [1] Technical indicators — computed from ohlcv_clean.csv (no API)
      RSI, MACD, Bollinger Bands, EMA 7/21/50/200, ATR, OBV,
      Stochastic, ADX, returns 1d/7d/30d, volatility 30d

  [2] Fear & Greed Index — alternative.me free API, full history since 2018
      fear_greed_value (0–100), fear_greed_label

  [3] BTC Dominance + Global Market Cap — CoinGecko /global (free, no key)
      btc_dominance, total_market_cap_usd, total_volume_24h,
      defi_market_cap, defi_to_total_ratio

Outputs (all written to the same directory as this script):
  technical_indicators.csv  — one row per (asset_id, timestamp)
  fear_greed.csv            — one row per day (global signal)
  global_market.csv         — one row per day (global signal)

Usage:
    python add_technical_features.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

DYPLOM     = Path(__file__).parent
OHLCV_FILE = DYPLOM / "ohlcv_clean.csv"

OUT_TECHNICAL = DYPLOM / "technical_indicators.csv"
OUT_FNG       = DYPLOM / "fear_greed.csv"
OUT_GLOBAL    = DYPLOM / "global_market.csv"

COINGECKO_BASE  = "https://api.coingecko.com/api/v3"
FNG_API         = "https://api.alternative.me/fng/?limit=0&format=json"

# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 5) -> dict | list:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = max(int(r.headers.get("Retry-After", 15)), 15)
                print(f"  ⏳ Rate-limited — waiting {wait}s …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1))
    return {}



def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs  = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input:  DataFrame with columns [asset_id, timestamp, open, high, low, close, volume]
            sorted by (asset_id, timestamp) ascending.
    Output: Same rows with ~20 indicator columns added.
    """
    results = []

    for asset_id, group in df.groupby("asset_id"):
        g = group.sort_values("timestamp").copy()
        c = g["close"]
        h = g["high"]
        l = g["low"]
        v = g["volume"]

        g["return_1d"]     = c.pct_change(1)
        g["return_7d"]     = c.pct_change(7)
        g["return_30d"]    = c.pct_change(30)
        g["volatility_30d"]= g["return_1d"].rolling(30).std()

        g["ema_7"]   = c.ewm(span=7,   adjust=False).mean()
        g["ema_21"]  = c.ewm(span=21,  adjust=False).mean()
        g["ema_50"]  = c.ewm(span=50,  adjust=False).mean()
        g["ema_200"] = c.ewm(span=200, adjust=False).mean()

        g["price_vs_ema50"]  = (c - g["ema_50"])  / g["ema_50"]
        g["price_vs_ema200"] = (c - g["ema_200"]) / g["ema_200"]
        g["ema_50_vs_200"]   = (g["ema_50"] - g["ema_200"]) / g["ema_200"]

        g["rsi_14"] = compute_rsi(c, 14)
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        g["macd"]        = ema12 - ema26
        g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
        g["macd_hist"]   = g["macd"] - g["macd_signal"]

        sma20          = c.rolling(20).mean()
        std20          = c.rolling(20).std()
        g["bb_upper"]  = sma20 + 2 * std20
        g["bb_lower"]  = sma20 - 2 * std20
        g["bb_width"]  = (g["bb_upper"] - g["bb_lower"]) / sma20
        g["bb_pct"]    = (c - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"])
        prev_close  = c.shift(1)
        tr          = pd.concat([
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs()
        ], axis=1).max(axis=1)
        g["atr_14"] = tr.ewm(com=13, min_periods=14).mean()
        g["atr_pct"]= g["atr_14"] / c  # normalised ATR

        direction   = c.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        g["obv"]    = (v * direction).cumsum()
        g["obv_ema"]= g["obv"].ewm(span=20, adjust=False).mean()
        g["obv_divergence"] = (g["obv"] - g["obv_ema"]) / g["obv_ema"].abs().replace(0, float("nan"))
        low14  = l.rolling(14).min()
        high14 = h.rolling(14).max()
        g["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, float("nan"))
        g["stoch_d"] = g["stoch_k"].rolling(3).mean()

        tr14        = tr.ewm(com=13, min_periods=14).mean()
        plus_dm     = (h.diff()).clip(lower=0)
        minus_dm    = (-l.diff()).clip(lower=0)
        plus_dm2    = plus_dm.where(plus_dm > minus_dm, 0)
        minus_dm2   = minus_dm.where(minus_dm > plus_dm, 0)
        plus_di     = 100 * plus_dm2.ewm(com=13, min_periods=14).mean() / tr14
        minus_di    = 100 * minus_dm2.ewm(com=13, min_periods=14).mean() / tr14
        dx          = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
        g["adx"]    = dx.ewm(com=13, min_periods=14).mean()
        g["plus_di"]  = plus_di
        g["minus_di"] = minus_di
        results.append(g)
        print(f"  ✓ {asset_id:<35} {len(g)} rows", end="\r")

    print()
    return pd.concat(results, ignore_index=True)


def run_technical(ohlcv_path: Path, out_path: Path):
    print("\n" + "="*60)
    print("  PART 1 — Technical Indicators")
    print("="*60)
    print(f"  Loading {ohlcv_path} …")
    df = pd.read_csv(ohlcv_path)
    print(f"  {len(df):,} rows, {df['asset_id'].nunique()} coins")

    print("  Computing indicators …")
    result = compute_technical_indicators(df)
    indicator_cols = [
        "asset_id", "exchange", "pair_symbol", "granularity", "timestamp",
        "return_1d", "return_7d", "return_30d", "volatility_30d",
        "ema_7", "ema_21", "ema_50", "ema_200",
        "price_vs_ema50", "price_vs_ema200", "ema_50_vs_200",
        "rsi_14",
        "macd", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "bb_pct",
        "atr_14", "atr_pct",
        "obv", "obv_ema", "obv_divergence",
        "stoch_k", "stoch_d",
        "adx", "plus_di", "minus_di",
    ]
    result = result[indicator_cols]
    result.to_csv(out_path, index=False)

    print(f"\n  ✅ {len(result):,} rows → {out_path}")
    print(f"     Indicators: {len(indicator_cols) - 5} features per row")



def run_fear_greed(out_path: Path):
    print("\n" + "="*60)
    print("  PART 2 — Fear & Greed Index")
    print("="*60)

    print("  Fetching from alternative.me …")
    data = _get(FNG_API)

    if not data or "data" not in data:
        print("  ✗ Failed to fetch Fear & Greed data")
        return

    rows = []
    for entry in data["data"]:
        ts = int(entry["timestamp"])
        rows.append({
            "timestamp":         ts * 1000,   # convert to ms to match ohlcv
            "date":              pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d"),
            "fear_greed_value":  int(entry["value"]),
            "fear_greed_label":  entry["value_classification"],
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    print(f"  ✅ {len(df):,} days of data → {out_path}")
    print(f"     Range: {df['date'].min()} → {df['date'].max()}")
    print(f"     Distribution:")
    print(df["fear_greed_label"].value_counts().to_string())