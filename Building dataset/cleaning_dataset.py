import pandas as pd
import numpy as np

print("Loading dataset...")
df = pd.read_csv("dataset_clean.csv")
print(f"  Shape: {df.shape}")

drop_cols = [
    "is_active",          # always 1
    "source",             # metadata
    "granularity",        # always 1d
    "fear_greed_label",   # redundant with fear_greed_value
    "cap_tier",           # redundant with is_large/mid/small/micro flags
    "ath_change_pct",     # near-duplicate of price_vs_ath
]
drop_cols = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=drop_cols)
print(f"\nDropped {len(drop_cols)} redundant columns")

inf_count = np.isinf(df["bb_pct"]).sum()
print(f"\nbb_pct: {inf_count} inf values → clipping to [−2, 3]")
df["bb_pct"] = df["bb_pct"].replace([np.inf, -np.inf], np.nan)
p1  = df["bb_pct"].quantile(0.01)
p99 = df["bb_pct"].quantile(0.99)
df["bb_pct"] = df["bb_pct"].clip(p1, p99)


winsorise_cols = [
    "atr_pct",
    "funding_rate_30d_cum",
    "obv_divergence",
    "price_vs_atl",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr_14",
    "liq_long_usd",
    "liq_short_usd",
    "liq_total_usd",
    "oi_usd",
    "volume",
    "market_cap_usd",
]
winsorise_cols = [c for c in winsorise_cols if c in df.columns]

print(f"\nWinsorising {len(winsorise_cols)} columns at 1st/99th percentile:")
for col in winsorise_cols:
    p1  = df[col].quantile(0.01)
    p99 = df[col].quantile(0.99)
    before_min = df[col].min()
    before_max = df[col].max()
    df[col] = df[col].clip(p1, p99)
    print(f"  {col}: [{before_min:.2g}, {before_max:.2g}] → [{p1:.2g}, {p99:.2g}]")


per_coin_normalise = [
    "macd", "macd_signal", "macd_hist","atr_14","obv",
]
per_coin_normalise = [c for c in per_coin_normalise if c in df.columns]

print(f"\nPer-coin z-score normalising {len(per_coin_normalise)} price-scale columns:")
df = df.sort_values(["asset_id", "timestamp"])
for col in per_coin_normalise:
    mean = df.groupby("asset_id")[col].transform("mean")
    std  = df.groupby("asset_id")[col].transform("std").replace(0, 1)
    df[col] = (df[col] - mean) / std
    print(f"  {col} → z-scored per coin")

print("\n=== NULL CHECK AFTER FIXES ===")
nulls = df.isnull().sum()
nulls = nulls[nulls > 0].sort_values(ascending=False)
if len(nulls):
    for col, n in nulls.items():
        print(f"  {col}: {n:,} ({n/len(df)*100:.1f}%)")
else:
    print("  No nulls.")

df.to_csv("dataset_final.csv", index=False)

print(f"\n=== FINAL SHAPE ===")
print(f"  Rows:    {len(df):,}")
print(f"  Columns: {len(df.columns)}")
print("\nSaved → dataset_final.csv")