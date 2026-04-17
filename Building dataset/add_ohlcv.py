"""
markets_ccxt_top200.py
──────────────────────
Collects market-pair metadata and OHLCV (price/volume) history for the
top-200 crypto coins — USDT pairs only — across multiple exchanges.

Key improvements over the original markets_ccxt.py:
  ✦  USDT-only filtering  (no BTC, ETH, or fiat quote pairs)
  ✦  Full pagination      (bypasses the 1500-candle ccxt limit → 5-8 years of daily data)
  ✦  Deduplication        (skips candles already on disk via a simple date-set check)
  ✦  Robust error handling with per-exchange & per-pair retry logic
  ✦  Progress persistence (appends incrementally; safe to resume after crash)
  ✦  Verbose logging      (shows date ranges being fetched, total candles saved)

Prerequisites:
    pip install ccxt pandas tqdm requests

Step 1 — build assets.csv:
    python build_top200_assets.py --templates-dir ./data

Step 2 — collect OHLCV:
    python markets_ccxt_top200.py --templates-dir ./data --days 2555

    --days  default 2555 (7 years). Use 1825 for 5 years, 2920 for 8 years.
    --timeframe  default 1d (daily). Use 1h for hourly (much larger output).
    --exchanges  optional override, e.g. --exchanges binance kraken
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
from tqdm import tqdm

from etl_config import (
    ALLOWED_QUOTES,
    CANDLES_PER_REQUEST,
    CCXT_EXCHANGES,
    DEFAULT_DAYS,
    MAX_DAYS,
    OHLCV_TIMEFRAME,
    SPOT_ONLY,
)
from utils_io import append_rows_csv, ensure_csv, sanitize_symbol

PAIRS_COLS = [
    "asset_id", "exchange", "market_type", "base", "quote",
    "pair_symbol", "is_spot", "is_perp", "is_option", "fee_tier",
]
OHLCV_COLS = [
    "asset_id", "exchange", "pair_symbol", "granularity",
    "timestamp", "open", "high", "low", "close", "volume", "source",
]

MS_PER_DAY = 24 * 60 * 60 * 1000


def load_asset_index(templates_dir: Path) -> tuple[set[str], dict[str, str]]:
    """
    Returns:
        symbols  – set of uppercase ticker symbols  e.g. {'BTC', 'ETH', …}
        sym2id   – dict mapping symbol → asset_id   e.g. {'BTC': 'bitcoin'}
    """
    path = templates_dir / "assets.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"assets.csv not found at {path}\n"
            "Run build_top200_assets.py first."
        )
    df = pd.read_csv(path)
    df["symbol"] = df["symbol"].dropna().str.upper()
    symbols = set(df["symbol"].tolist())
    sym2id = dict(zip(df["symbol"], df["asset_id"]))
    return symbols, sym2id


def load_existing_timestamps(ohlcv_path: Path) -> dict[tuple[str, str], set[int]]:
    """
    Build a dict of (exchange, pair_symbol) → set of already-stored timestamps (ms).
    Used to skip candles we already have and avoid duplicates on resume.
    """
    if not ohlcv_path.exists():
        return {}
    try:
        df = pd.read_csv(ohlcv_path, usecols=["exchange", "pair_symbol", "timestamp"])
        index: dict[tuple, set] = {}
        for _, row in df.iterrows():
            key = (row["exchange"], row["pair_symbol"])
            index.setdefault(key, set()).add(int(row["timestamp"]))
        return index
    except Exception:
        return {}


def fetch_ohlcv_paginated(
    ex: ccxt.Exchange,
    pair: str,
    timeframe: str,
    since_ms: int,
    limit: int = CANDLES_PER_REQUEST,
    existing_ts: set[int] | None = None,
) -> list[list]:
    """
    Fetch all candles from `since_ms` to now by paginating through pages of
    `limit` candles.  Returns a flat list of raw OHLCV rows
    [timestamp_ms, open, high, low, close, volume].

    Deduplicates against `existing_ts` if provided.
    """
    all_candles: list[list] = []
    cursor = since_ms

    while True:
        try:
            batch = ex.fetch_ohlcv(pair, timeframe=timeframe, since=cursor, limit=limit)
        except ccxt.BadSymbol:
            break
        except (ccxt.NetworkError, ccxt.RequestTimeout) as e:
            print(f"      ⚠  Network error for {pair}: {e}. Retrying in 10s …")
            time.sleep(10)
            continue
        except Exception as e:
            print(f"      ✗  {pair}: {e}")
            break

        if not batch:
            break

        if existing_ts:
            batch = [c for c in batch if c[0] not in existing_ts]

        all_candles.extend(batch)

        last_ts = batch[-1][0] if batch else None
        if last_ts is None:
            break

        tf_ms = ex.parse_timeframe(timeframe) * 1000
        next_cursor = last_ts + tf_ms

        now_ms = ex.milliseconds()
        if next_cursor >= now_ms:
            break

        if len(batch) < limit:
            break

        cursor = next_cursor
        time.sleep(ex.rateLimit / 1000 if ex.rateLimit else 0.2)

    return all_candles


def discover_usdt_pairs(
    ex: ccxt.Exchange,
    ex_id: str,
    asset_symbols: set[str],
    sym2id: dict[str, str],
) -> list[list]:
    """
    Return market-pair rows for asset_symbols, USDT quote only.
    """
    markets = ex.load_markets()
    rows = []
    for m in markets.values():
        base  = sanitize_symbol(m.get("base"))
        quote = sanitize_symbol(m.get("quote"))
        if base not in asset_symbols:
            continue
        if quote not in ALLOWED_QUOTES:
            continue
        if SPOT_ONLY and not m.get("spot"):
            continue
        pair        = m.get("symbol")
        market_type = (
            "spot"   if m.get("spot")   else
            "swap"   if m.get("swap")   else
            "option" if m.get("option") else
            "other"
        )
        rows.append([
            sym2id.get(base),
            ex_id,
            market_type,
            base,
            quote,
            pair,
            bool(m.get("spot")),
            bool(m.get("swap")),
            bool(m.get("option")),
            None,
        ])
    return rows


def run(
    templates_dir: Path,
    days: int = DEFAULT_DAYS,
    timeframe: str = OHLCV_TIMEFRAME,
    exchanges: list[str] | None = None,
    output_dir: Path | None = None,
):
    from etl_config import RUN_FOLDER_NAME

    templates_dir = Path(templates_dir)

    if output_dir is None:
        output_dir = templates_dir.parent / RUN_FOLDER_NAME
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✔  Output folder : {output_dir.resolve()}")
    print(f"   (Your existing data in '{templates_dir}' is untouched.)")

    pairs_path = output_dir / "market_pairs.csv"
    ohlcv_path = output_dir / "ohlcv.csv"

    ensure_csv(pairs_path, PAIRS_COLS)
    ensure_csv(ohlcv_path, OHLCV_COLS)

    assets_source = output_dir if (output_dir / "assets.csv").exists() else templates_dir
    asset_symbols, sym2id = load_asset_index(assets_source)
    print(f"✔  Loaded {len(asset_symbols)} asset symbols from assets.csv")

    exchange_list = exchanges or CCXT_EXCHANGES
    since_ms = int(datetime.now(timezone.utc).timestamp() * 1000) - days * MS_PER_DAY
    since_dt = datetime.fromtimestamp(since_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

    print(f"✔  Collecting {timeframe} candles from {since_dt} → now  ({days} days)")
    print(f"✔  Quote filter: {ALLOWED_QUOTES}")
    print(f"✔  Exchanges: {exchange_list}\n")

    print("▶  Indexing existing OHLCV data for deduplication …")
    existing_index = load_existing_timestamps(ohlcv_path)
    print(f"   Found {sum(len(v) for v in existing_index.values()):,} existing candles.\n")

    total_new_candles = 0

    for ex_id in exchange_list:
        print(f"{'─'*60}")
        print(f"▶  Exchange: {ex_id.upper()}")

        try:
            ex: ccxt.Exchange = getattr(ccxt, ex_id)()
        except AttributeError:
            print(f"   ✗  ccxt has no exchange named '{ex_id}'. Skipping.")
            continue

        ex.enableRateLimit = True

        print(f"   Loading markets …")
        try:
            pair_rows = discover_usdt_pairs(ex, ex_id, asset_symbols, sym2id)
        except Exception as e:
            print(f"   ✗  Failed to load markets for {ex_id}: {e}")
            continue

        print(f"   Found {len(pair_rows)} USDT pairs for tracked assets.")
        if not pair_rows:
            continue

        append_rows_csv(pairs_path, pair_rows, PAIRS_COLS)

        if not ex.has.get("fetchOHLCV"):
            print(f"   ⚠  {ex_id} does not support fetchOHLCV. Skipping OHLCV.")
            continue

        exchange_new = 0
        for row in tqdm(pair_rows, desc=f"   {ex_id} OHLCV", unit="pair", ncols=80):
            asset_id, exchange, _, base, quote, pair, *_ = row

            existing_ts = existing_index.get((exchange, pair))
            candles = fetch_ohlcv_paginated(
                ex, pair, timeframe, since_ms,
                limit=CANDLES_PER_REQUEST,
                existing_ts=existing_ts,
            )

            if candles:
                ohlcv_rows = [
                    [asset_id, exchange, pair, timeframe, int(ts), o, h, l, c, v, "ccxt"]
                    for ts, o, h, l, c, v in candles
                ]
                append_rows_csv(ohlcv_path, ohlcv_rows, OHLCV_COLS)
                exchange_new += len(candles)

                if existing_ts is None:
                    existing_index[(exchange, pair)] = {c[0] for c in candles}
                else:
                    existing_ts.update(c[0] for c in candles)

            time.sleep(ex.rateLimit / 1000 if ex.rateLimit else 0.2)

        total_new_candles += exchange_new
        print(f"   ✅  {ex_id}: {exchange_new:,} new candles written.\n")

    print(f"{'═'*60}")
    print(f"✅  Done.  Total new candles written: {total_new_candles:,}")
    print(f"   Pairs file : {pairs_path}")
    print(f"   OHLCV file : {ohlcv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Collect OHLCV price data for top-200 crypto coins (USDT only)."
    )
    p.add_argument(
        "--templates-dir", default=".",
        help="Path to the template directory containing assets.csv",
    )
    p.add_argument(
        "--days", type=int, default=DEFAULT_DAYS,
        help=f"How many days back to collect (default {DEFAULT_DAYS} = 7 years, max {MAX_DAYS})",
    )
    p.add_argument(
        "--timeframe", default=OHLCV_TIMEFRAME,
        help="Candle timeframe: 1d (default), 4h, 1h, 15m, etc.",
    )
    p.add_argument(
        "--exchanges", nargs="+", default=None,
        help="Override exchange list, e.g. --exchanges binance kraken",
    )
    p.add_argument(
        "--output-dir", default=None,
        help=(
            "Folder to write market_pairs.csv and ohlcv.csv into. "
            "Defaults to a new timestamped subfolder next to --templates-dir. "
            "Your existing data is never modified."
        ),
    )
    args = p.parse_args()

    days = min(args.days, MAX_DAYS)
    run(
        templates_dir=Path(args.templates_dir),
        days=days,
        timeframe=args.timeframe,
        exchanges=args.exchanges,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
