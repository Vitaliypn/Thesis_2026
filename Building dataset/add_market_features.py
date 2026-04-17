"""
add_market_features.py
────────────────────────
Fetches additional market microstructure features for price prediction,
joined to your existing ohlcv_clean.csv by (asset_id, timestamp).

Features collected (all free via ccxt, no API key needed):

  FROM DAILY OHLCV AGGREGATION (derived, no extra calls):
  ┌─────────────────────────────────────────────────────┐
  │  Already in ohlcv_clean: open, high, low, close,    │
  │  volume                                             │
  └─────────────────────────────────────────────────────┘

  NEW — TICKER SNAPSHOT (fetchTicker, 1 call/coin/exchange):
  • quote_volume      – volume in USDT terms (volume × price)
  • vwap              – volume-weighted average price for the day
  • price_change_pct  – 24h % price change
  • bid               – best bid price
  • ask               – best ask price
  • bid_ask_spread    – ask - bid (liquidity proxy)
  • bid_ask_spread_pct– spread as % of mid price

  NEW — ORDER BOOK SNAPSHOT (fetchOrderBook, 1 call/coin/exchange):
  • ob_bid_depth_1pct  – total bid volume within 1% of mid price
  • ob_ask_depth_1pct  – total ask volume within 1% of mid price
  • ob_imbalance       – (bid_depth - ask_depth) / (bid_depth + ask_depth)
                         > 0 = more buyers, < 0 = more sellers
  • ob_bid_levels      – number of bid price levels in top 20
  • ob_ask_levels      – number of ask price levels in top 20

  NEW — FUNDING RATE (fetchFundingRateHistory, perp swap only, Binance):
  • funding_rate       – 8h funding rate (positive = longs pay shorts)
  • funding_rate_avg7d – 7-day rolling average funding rate

  NEW — OPEN INTEREST (fetchOpenInterestHistory, Binance perp):
  • open_interest_usdt – total open interest in USDT (perp contracts)

Notes:
  • Ticker + order book = CURRENT snapshot only (not historical).
    These will only enrich today's / recent rows.
  • Funding rate history = historical, going back ~1 year on Binance.
  • Open interest history = historical, going back ~30 days on Binance.
  • Order book is the most valuable feature for short-term prediction
    (shows real buying/selling pressure) but is snapshot-only.

Reads  : ohlcv_clean.csv
         assets.csv
Writes : market_features.csv

Usage:
    python add_market_features.py
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd

DYPLOM      = Path(__file__).parent
OHLCV_FILE  = DYPLOM / "ohlcv_clean.csv"
ASSETS_FILE = DYPLOM / "assets.csv"
OUT_FILE    = DYPLOM / "market_features.csv"

PERP_EXCHANGE = "binance"
SPOT_EXCHANGE = "binance"

ORDER_BOOK_DEPTH = 20
OB_DEPTH_PCT     = 0.01
FUNDING_DAYS     = 365
OI_DAYS          = 30

FEATURES_COLS = [
    "asset_id", "pair_symbol", "timestamp",
    "quote_volume", "vwap", "price_change_pct",
    "bid", "ask", "bid_ask_spread", "bid_ask_spread_pct",
    "ob_bid_depth_1pct", "ob_ask_depth_1pct",
    "ob_imbalance", "ob_bid_levels", "ob_ask_levels",
    "funding_rate", "funding_rate_avg7d",
    "open_interest_usdt",
    "source",
]

MS_PER_DAY = 86_400_000


def now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def safe_call(fn, *args, retries=4, **kwargs):
    """Call an exchange method with retry + rate limit handling."""
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except ccxt.RateLimitExceeded:
            wait = 15 * (attempt + 1)
            print(f"    ⏳ Rate limit — waiting {wait}s …")
            time.sleep(wait)
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            wait = 10 * (attempt + 1)
            print(f"    ⚠  Network error: {e} — retry in {wait}s …")
            time.sleep(wait)
        except Exception as e:
            print(f"    ✗  {e}")
            return None
    return None


def fetch_ticker_features(ex: ccxt.Exchange, pair: str) -> dict:
    t = safe_call(ex.fetch_ticker, pair)
    if not t:
        return {}
    mid = ((t.get("bid") or 0) + (t.get("ask") or 0)) / 2
    spread = (t.get("ask") or 0) - (t.get("bid") or 0)
    spread_pct = (spread / mid * 100) if mid else None
    return {
        "quote_volume":      t.get("quoteVolume"),
        "vwap":              t.get("vwap"),
        "price_change_pct":  t.get("percentage"),
        "bid":               t.get("bid"),
        "ask":               t.get("ask"),
        "bid_ask_spread":    round(spread, 8) if spread else None,
        "bid_ask_spread_pct":round(spread_pct, 6) if spread_pct else None,
    }


def fetch_orderbook_features(ex: ccxt.Exchange, pair: str) -> dict:
    ob = safe_call(ex.fetch_order_book, pair, ORDER_BOOK_DEPTH)
    if not ob:
        return {}

    bids = ob.get("bids", [])
    asks = ob.get("asks", [])

    if not bids or not asks:
        return {}

    mid = (bids[0][0] + asks[0][0]) / 2
    threshold_bid = mid * (1 - OB_DEPTH_PCT)
    threshold_ask = mid * (1 + OB_DEPTH_PCT)

    bid_depth = sum(size for price, size in bids if price >= threshold_bid)
    ask_depth = sum(size for price, size in asks if price <= threshold_ask)

    total = bid_depth + ask_depth
    imbalance = (bid_depth - ask_depth) / total if total else 0

    return {
        "ob_bid_depth_1pct": round(bid_depth, 4),
        "ob_ask_depth_1pct": round(ask_depth, 4),
        "ob_imbalance":      round(imbalance, 6),
        "ob_bid_levels":     len(bids),
        "ob_ask_levels":     len(asks),
    }


def fetch_funding_history(ex: ccxt.Exchange, perp_pair: str, days: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns: timestamp, funding_rate, funding_rate_avg7d
    timestamp is in ms (daily aligned — funding is 8h so we resample to daily mean).
    """
    since = now_ms() - days * MS_PER_DAY
    rows = []
    cursor = since

    while True:
        batch = safe_call(
            ex.fetch_funding_rate_history,
            perp_pair, cursor, limit=500
        )
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1].get("timestamp", 0)
        if last_ts <= cursor or len(batch) < 500:
            break
        cursor = last_ts + 1
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)[["timestamp", "fundingRate"]].rename(
        columns={"fundingRate": "funding_rate"}
    )
    df["funding_rate"] = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.normalize()

    daily = df.groupby("date")["funding_rate"].mean().reset_index()
    daily["timestamp"] = daily["date"].astype("int64") // 1_000_000
    daily["funding_rate_avg7d"] = daily["funding_rate"].rolling(7, min_periods=1).mean()
    return daily[["timestamp", "funding_rate", "funding_rate_avg7d"]]


def fetch_oi_history(ex: ccxt.Exchange, perp_pair: str, days: int) -> pd.DataFrame:
    """
    Returns DataFrame with columns: timestamp, open_interest_usdt
    """
    since = now_ms() - days * MS_PER_DAY
    rows = []
    cursor = since

    while True:
        batch = safe_call(
            ex.fetch_open_interest_history,
            perp_pair, "1d", cursor, limit=500
        )
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1].get("timestamp", 0) if isinstance(batch[-1], dict) else 0
        if last_ts <= cursor or len(batch) < 500:
            break
        cursor = last_ts + 1
        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        if isinstance(r, dict):
            records.append({
                "timestamp": r.get("timestamp"),
                "open_interest_usdt": r.get("openInterestValue") or r.get("openInterest"),
            })

    df = pd.DataFrame(records).dropna()
    df["open_interest_usdt"] = pd.to_numeric(df["open_interest_usdt"], errors="coerce")
    return df[["timestamp", "open_interest_usdt"]]


def run(days_funding: int = FUNDING_DAYS, days_oi: int = OI_DAYS):
    print("=" * 60)
    print("  fetch_market_features.py")
    print("=" * 60)

    assets = pd.read_csv(ASSETS_FILE)
    asset_to_symbol = dict(zip(assets["asset_id"], assets["symbol"].str.upper()))

    ohlcv = pd.read_csv(OHLCV_FILE, usecols=["asset_id", "timestamp"])
    coin_ids = ohlcv["asset_id"].unique().tolist()
    print(f"  Coins to enrich: {len(coin_ids)}")

    spot_ex = getattr(ccxt, SPOT_EXCHANGE)()
    spot_ex.enableRateLimit = True
    spot_ex.load_markets()

    perp_ex = ccxt.binance({"options": {"defaultType": "future"}})
    perp_ex.enableRateLimit = True
    perp_ex.load_markets()

    all_rows = []
    today_ts = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).timestamp() * 1000)

    for i, asset_id in enumerate(coin_ids, 1):
        symbol = asset_to_symbol.get(asset_id)
        if not symbol:
            continue

        spot_pair = f"{symbol}/USDT"
        perp_pair = f"{symbol}/USDT:USDT"

        print(f"\n[{i:>3}/{len(coin_ids)}] {asset_id} ({spot_pair})")
        row = {
            "asset_id":   asset_id,
            "pair_symbol": spot_pair,
            "timestamp":  today_ts,
            "source":     "ccxt",
        }

        print(f"  → ticker …", end=" ")
        t_features = fetch_ticker_features(spot_ex, spot_pair)
        row.update(t_features)
        print("✓" if t_features else "✗")
        time.sleep(0.3)

        print(f"  → order book …", end=" ")
        ob_features = fetch_orderbook_features(spot_ex, spot_pair)
        row.update(ob_features)
        print("✓" if ob_features else "✗")
        time.sleep(0.3)

        all_rows.append(row)

        print(f"  → funding rate history …", end=" ")
        if perp_pair in perp_ex.markets:
            funding_df = fetch_funding_history(perp_ex, perp_pair, days_funding)
            if not funding_df.empty:
                for _, fr_row in funding_df.iterrows():
                    all_rows.append({
                        "asset_id":            asset_id,
                        "pair_symbol":         perp_pair,
                        "timestamp":           int(fr_row["timestamp"]),
                        "funding_rate":        fr_row["funding_rate"],
                        "funding_rate_avg7d":  fr_row["funding_rate_avg7d"],
                        "source":              "ccxt_funding",
                    })
                print(f"✓ ({len(funding_df)} days)")
            else:
                print("✗ (no data)")
        else:
            print("✗ (no perp pair)")
        time.sleep(0.3)

        print(f"  → open interest history …", end=" ")
        if perp_pair in perp_ex.markets:
            oi_df = fetch_oi_history(perp_ex, perp_pair, days_oi)
            if not oi_df.empty:
                for _, oi_row in oi_df.iterrows():
                    all_rows.append({
                        "asset_id":           asset_id,
                        "pair_symbol":        perp_pair,
                        "timestamp":          int(oi_row["timestamp"]),
                        "open_interest_usdt": oi_row["open_interest_usdt"],
                        "source":             "ccxt_oi",
                    })
                print(f"✓ ({len(oi_df)} days)")
            else:
                print("✗ (no data)")
        else:
            print("✗ (no perp pair)")
        time.sleep(0.5)

    print(f"\n▶  Writing {len(all_rows):,} rows to {OUT_FILE} …")
    out_df = pd.DataFrame(all_rows)

    for col in FEATURES_COLS:
        if col not in out_df.columns:
            out_df[col] = None

    out_df = out_df[FEATURES_COLS]
    out_df.to_csv(OUT_FILE, index=False)

    print(f"\n{'='*60}")
    print(f"✅  Done!  {len(out_df):,} rows written to:")
    print(f"   {OUT_FILE}")
    print(f"{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Fetch market microstructure features")
    p.add_argument("--days-funding", type=int, default=FUNDING_DAYS,
                   help=f"Days of funding rate history (default {FUNDING_DAYS})")
    p.add_argument("--days-oi", type=int, default=OI_DAYS,
                   help=f"Days of open interest history (default {OI_DAYS})")
    args = p.parse_args()
    run(days_funding=args.days_funding, days_oi=args.days_oi)
