"""
add_missing_ohlcv.py
──────────────
Cleans ass_missing_aohlcv.py in two steps:

  1. Remove any coins not in the top of assets.csv
  2. For duplicate (asset_id, timestamp) rows across different exchanges,
     keep only ONE row — priority: binance > kraken > bybit > kucoin > okx
"""

from pathlib import Path
import pandas as pd

DYPLOM      = Path(__file__).parent
OHLCV_IN    = DYPLOM / "ohlcv.csv"
ASSETS_FILE = DYPLOM / "assets.csv"
OHLCV_OUT   = DYPLOM / "ohlcv_clean.csv"

EXCHANGE_PRIORITY = ["binance", "kraken", "bybit", "kucoin", "okx"]

TOP_N = 250


def main():
    print("=" * 55)
    print("  clean_ohlcv.py")
    print("=" * 55)
    print(f"\n▶  Loading assets from {ASSETS_FILE} …")
    assets_df = pd.read_csv(ASSETS_FILE)
    top250_ids = set(assets_df["asset_id"].head(TOP_N).tolist())
    print(f"   Top-{TOP_N} asset IDs loaded: {len(top250_ids)}")

    print(f"\n▶  Loading OHLCV from {OHLCV_IN} …")
    df = pd.read_csv(OHLCV_IN)
    print(f"   Rows before cleaning : {len(df):,}")
    print(f"   Unique coins         : {df['asset_id'].nunique()}")
    print(f"   Unique exchanges     : {df['exchange'].unique().tolist()}")
    print(f"\n▶  Step 1 — Removing coins outside top-{TOP_N} …")
    df_filtered = df[df["asset_id"].isin(top250_ids)].copy()
    removed_coins = df["asset_id"].nunique() - df_filtered["asset_id"].nunique()
    removed_rows  = len(df) - len(df_filtered)
    print(f"   Coins removed : {removed_coins}")
    print(f"   Rows removed  : {removed_rows:,}")
    print(f"   Rows remaining: {len(df_filtered):,}")
    print(f"\n▶  Step 2 — Deduplicating same coin+timestamp across exchanges …")

    priority_map = {ex: i for i, ex in enumerate(EXCHANGE_PRIORITY)}
    df_filtered["_priority"] = df_filtered["exchange"].map(priority_map).fillna(99).astype(int)

    before = len(df_filtered)
    df_filtered = df_filtered.sort_values("_priority")
    df_deduped = df_filtered.drop_duplicates(subset=["asset_id", "timestamp"], keep="first")

    df_deduped = df_deduped.drop(columns=["_priority"])
    df_deduped = df_deduped.sort_values(["asset_id", "timestamp"]).reset_index(drop=True)
    dupes_removed = before - len(df_deduped)
    print(f"   Duplicate rows removed : {dupes_removed:,}")
    print(f"   Rows remaining         : {len(df_deduped):,}")
    print(f"\n▶  Final dataset:")
    print(f"   Unique coins     : {df_deduped['asset_id'].nunique()}")
    print(f"   Unique exchanges : {df_deduped['exchange'].unique().tolist()}")
    print(f"   Date range       : {df_deduped['timestamp'].min()} → {df_deduped['timestamp'].max()}")
    print(f"   Total rows       : {len(df_deduped):,}")
    df_deduped.to_csv(OHLCV_OUT, index=False)
    print(f"\n✅  Saved to {OHLCV_OUT}")


if __name__ == "__main__":
    main()
