"""
add_market_flags.py
────────────────────
Reads dataset_enriched.csv, adds new features, saves as dataset_flagged.csv.

New features:
  BINARY FLAGS
    is_stable       — 1 if coin is a stablecoin (USDT, USDC, DAI, etc.)
    is_active       — 1 if coin has traded in the last 30 days with non-trivial volume
    is_defi         — 1 if coin has TVL data (DeFi protocol)
    is_large_cap    — 1 if cap_tier == 'large'  (>$10B)
    is_mid_cap      — 1 if cap_tier == 'mid'    ($1B–$10B)
    is_small_cap    — 1 if cap_tier == 'small'  ($100M–$1B)
    is_micro_cap    — 1 if cap_tier == 'micro'  (<$100M)
    has_futures     — 1 if coin has derivatives data (oi_usd not null)
    is_above_ema50  — 1 if close > ema_50
    is_above_ema200 — 1 if close > ema_200
    is_new_ath      — 1 if price within 5% of ATH
    is_bear_market  — 1 if price >50% below ATH

"""

from pathlib import Path
import pandas as pd
import numpy as np

DYPLOM   = Path(__file__).parent
IN_FILE  = DYPLOM / "dataset_enriched.csv"
OUT_FILE = DYPLOM / "dataset_flagged.csv"

STABLECOINS = {
    "tether", "usd-coin", "binance-usd", "dai", "true-usd", "frax",
    "pax-gold", "paypal-usd", "first-digital-usd", "usd1-wlfi",
    "ripple-usd", "bfusd", "stable-2", "falcon-finance-ff",
}

print("=" * 62)
print("  add_market_flags.py")
print("=" * 62)

print(f"\n  Loading {IN_FILE.name} …")
df = pd.read_csv(IN_FILE, low_memory=False)
print(f"  {len(df):,} rows  |  {df['asset_id'].nunique()} coins  |  {len(df.columns)} cols")

df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)

print("\n  Computing binary flags …")
df["is_stable"] = df["asset_id"].isin(STABLECOINS).astype(int)
df["is_active"] = (
    df.groupby("asset_id")["volume"]
      .transform(lambda x: (x.rolling(30, min_periods=1).mean() > 0).astype(int))
)
if "tvl_usd" in df.columns:
    df["is_defi"] = df["tvl_usd"].notna().astype(int)
else:
    df["is_defi"] = 0
if "cap_tier" in df.columns:
    df["is_large_cap"] = (df["cap_tier"] == "large").astype(int)
    df["is_mid_cap"]   = (df["cap_tier"] == "mid").astype(int)
    df["is_small_cap"] = (df["cap_tier"] == "small").astype(int)
    df["is_micro_cap"] = (df["cap_tier"] == "micro").astype(int)
else:
    for c in ["is_large_cap","is_mid_cap","is_small_cap","is_micro_cap"]:
        df[c] = np.nan


if "oi_usd" in df.columns:
    df["has_futures"] = df["oi_usd"].notna().astype(int)
else:
    df["has_futures"] = 0

if "ema_50" in df.columns and "close" in df.columns:
    df["is_above_ema50"] = (df["close"] > df["ema_50"]).astype(int)
else:
    df["is_above_ema50"] = np.nan

if "ema_200" in df.columns and "close" in df.columns:
    df["is_above_ema200"] = (df["close"] > df["ema_200"]).astype(int)
else:
    df["is_above_ema200"] = np.nan


if "ath_change_pct" in df.columns:
    df["is_new_ath"] = (df["ath_change_pct"] >= -5).astype(int)
else:
    df["is_new_ath"] = np.nan


if "ath_change_pct" in df.columns:
    df["is_bear_market"] = (df["ath_change_pct"] <= -50).astype(int)
else:
    df["is_bear_market"] = np.nan
print("  Computing market share features …")


df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
df.to_csv(OUT_FILE, index=False)

new_cols = [
    "is_stable", "is_active", "is_defi", "is_large_cap", "is_mid_cap",
    "is_small_cap", "is_micro_cap", "has_futures", "is_above_ema50",
    "is_above_ema200", "is_new_ath", "is_bear_market",
    "coin_mcap_share", "stablecoin_mcap_share",
    "btc_mcap_share", "eth_mcap_share", "defi_mcap_share",
]

print(f"\n{'='*62}")
print(f"✅  Saved: {OUT_FILE.name}")
print(f"   Rows    : {len(df):,}")
print(f"   Columns : {len(df.columns)}  (+{len([c for c in new_cols if c in df.columns])} new)")

print(f"\n  ── Binary flags ─────────────────────────────────────────")
binary = ["is_stable","is_active","is_defi","is_large_cap","is_mid_cap",
          "is_small_cap","is_micro_cap","has_futures","is_above_ema50",
          "is_above_ema200","is_new_ath","is_bear_market"]
for col in binary:
    if col not in df.columns:
        continue
    pct_true = df[col].mean() * 100
    print(f"  {col:<25} {pct_true:>6.1f}% rows = 1")