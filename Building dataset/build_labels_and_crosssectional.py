"""
build_labels_and_crosssectional.py
───────────────────────────────────
Transforms dataset_features.csv into a model-ready dataset.

Two things happen here:

  [1] LABEL GENERATION
      For each coin × month-end date, compute the forward 30-day
      risk-adjusted return (Sharpe-style), then bucket into quartiles
      relative to the full cohort that month:

        Strong Buy  — top 25% of coins (best risk-adjusted return)
        Buy         — 25–50%
        Hold        — 50–75%
        Avoid       — bottom 25% (worst risk-adjusted return)

      Label is always RELATIVE — you're predicting which coins will
      outperform the cohort, not absolute price direction.

  [2] CROSS-SECTIONAL FEATURE ENGINEERING
      For each feature on each month-end snapshot, compute where this
      coin sits RELATIVE to all other coins that month:

        {feature}_rank      — percentile rank 0–1 among all coins
        {feature}_zscore    — z-score vs cohort (mean=0, std=1)

      This transforms absolute indicators into relative ones — exactly
      what a ranking model needs. A RSI of 65 means nothing alone;
      RSI in the top 10% of all coins that month means a lot.

Output:
  model_dataset.csv     — monthly snapshots with labels + all features
  label_distribution.csv — label counts per month (sanity check)

Usage:
    python build_labels_and_crosssectional.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd

DYPLOM    = Path(__file__).parent
IN_FILE   = DYPLOM / "dataset_features.csv"
OUT_FILE  = DYPLOM / "model_dataset.csv"
OUT_DIST  = DYPLOM / "label_distribution.csv"
FORWARD_DAYS = 30
MIN_COINS_PER_MONTH = 20

CROSSSECTIONAL_FEATURES = [
    "return_1d",
    "return_7d",
    "return_30d",
    "volatility_30d",
    "rsi_14",
    "macd_hist",
    "bb_pct",
    "atr_pct",
    "obv_divergence",
    "stoch_k",
    "adx",
    "volume_vs_30d_avg",
    "drawdown_from_90d_peak",
    "price_vs_ath",
    "range_position_30d",
    "consecutive_up_days",
    "consecutive_down_days",
    "coin_age_days",
]


def sharpe_30d(returns_series: pd.Series) -> float:
    """
    Annualised Sharpe-style score over a 30-day window.
    Using daily returns: mean / std * sqrt(365)
    Returns NaN if insufficient data or zero volatility.
    """
    if len(returns_series) < 15:
        return float("nan")
    mean = returns_series.mean()
    std  = returns_series.std()
    if std == 0 or pd.isna(std):
        return float("nan")
    return (mean / std) * np.sqrt(365)


def percentile_rank(series: pd.Series) -> pd.Series:
    """Rank each value as a percentile 0–1 within the series."""
    return series.rank(pct=True, na_option="keep")


def zscore(series: pd.Series) -> pd.Series:
    """Z-score normalise a series."""
    mean = series.mean()
    std  = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(float("nan"), index=series.index)
    return (series - mean) / std


def label_from_rank(rank_pct: float) -> str | None:
    """Convert percentile rank to quadrant label."""
    if pd.isna(rank_pct):
        return None
    if rank_pct >= 0.75:
        return "Strong Buy"
    if rank_pct >= 0.50:
        return "Buy"
    if rank_pct >= 0.25:
        return "Hold"
    return "Avoid"



def get_month_end_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the daily data, extract month-end rows only.
    Month-end = last available trading day in each calendar month per coin.
    """
    df = df.copy()
    df["date"]       = pd.to_datetime(df["date"])
    df["year_month"] = df["date"].dt.to_period("M")

    # Last available date per coin per month
    snapshots = (
        df.sort_values("date")
          .groupby(["asset_id", "year_month"])
          .last()
          .reset_index()
    )
    return snapshots



def compute_forward_sharpe(df_daily: pd.DataFrame,
                            snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    For each (asset_id, month-end date), look forward FORWARD_DAYS calendar days
    in the daily data and compute the Sharpe-style score.

    Attaches columns:
        forward_return_30d   — simple total return over next 30 days
        forward_sharpe_30d   — risk-adjusted version (our ranking signal)
    """
    df_daily = df_daily.copy()
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    daily_by_asset = {
        asset: grp.sort_values("date").set_index("date")
        for asset, grp in df_daily.groupby("asset_id")
    }

    forward_returns = []
    forward_sharpes = []

    for _, row in snapshots.iterrows():
        asset   = row["asset_id"]
        snap_dt = pd.to_datetime(row["date"])

        if asset not in daily_by_asset:
            forward_returns.append(float("nan"))
            forward_sharpes.append(float("nan"))
            continue

        asset_daily = daily_by_asset[asset]
        end_dt      = snap_dt + pd.Timedelta(days=FORWARD_DAYS)

        window = asset_daily.loc[
            (asset_daily.index > snap_dt) & (asset_daily.index <= end_dt),
            "close"
        ]

        if len(window) < 5:
            forward_returns.append(float("nan"))
            forward_sharpes.append(float("nan"))
            continue

        fwd_return = (window.iloc[-1] - window.iloc[0]) / window.iloc[0]
        daily_rets = window.pct_change().dropna()
        fwd_sharpe = sharpe_30d(daily_rets)

        forward_returns.append(fwd_return)
        forward_sharpes.append(fwd_sharpe)

    snapshots = snapshots.copy()
    snapshots["forward_return_30d"] = forward_returns
    snapshots["forward_sharpe_30d"] = forward_sharpes
    return snapshots



def assign_labels(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    For each month, rank all coins by forward_sharpe_30d and assign labels.
    Label is purely relative — top 25% of coins that month = Strong Buy.
    """
    labels       = []
    sharpe_ranks = []

    for month, grp in snapshots.groupby("year_month"):
        valid = grp["forward_sharpe_30d"].notna()
        if valid.sum() < MIN_COINS_PER_MONTH:
            labels.extend([None] * len(grp))
            sharpe_ranks.extend([float("nan")] * len(grp))
            continue

        ranks = percentile_rank(grp["forward_sharpe_30d"])
        for rank in ranks:
            labels.append(label_from_rank(rank))
            sharpe_ranks.append(rank)

    snapshots = snapshots.copy()
    snapshots["label"]              = labels
    snapshots["forward_sharpe_rank"] = sharpe_ranks
    return snapshots



def add_crosssectional_features(snapshots: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature in CROSSSECTIONAL_FEATURES, add two new columns
    per month-end snapshot:
        {feature}_rank   — percentile rank 0–1 among all coins that month
        {feature}_zscore — z-score vs cohort
    """
    result_parts = []
    available_features = [f for f in CROSSSECTIONAL_FEATURES if f in snapshots.columns]
    missing = set(CROSSSECTIONAL_FEATURES) - set(available_features)
    if missing:
        print(f"  ⚠  Features not found, skipping: {', '.join(sorted(missing))}")

    for month, grp in snapshots.groupby("year_month"):
        grp = grp.copy()
        for feat in available_features:
            grp[f"{feat}_rank"]   = percentile_rank(grp[feat])
            grp[f"{feat}_zscore"] = zscore(grp[feat])
        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)

def add_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds three simple composite scores — not used as model input directly,
    but useful for sanity-checking the labels and for baseline comparison.

    momentum_score     — average rank of return_7d and return_30d
    mean_reversion_score — inverted RSI rank (oversold = high score)
    trend_score        — average of price_vs_ema50_rank and adx_rank
    """
    df = df.copy()

    # Momentum: high rank = strong recent performance
    mom_cols = [c for c in ["return_7d_rank", "return_30d_rank"] if c in df.columns]
    if mom_cols:
        df["momentum_score"] = df[mom_cols].mean(axis=1)

    # Mean reversion: low RSI rank = oversold = high score
    if "rsi_14_rank" in df.columns:
        df["mean_reversion_score"] = 1 - df["rsi_14_rank"]

    # Trend strength
    trend_cols = [c for c in ["adx_rank", "price_vs_ema50"] if c in df.columns]
    if "price_vs_ema50_rank" in df.columns:
        trend_cols = [c for c in ["adx_rank", "price_vs_ema50_rank"] if c in df.columns]
    if trend_cols:
        df["trend_score"] = df[trend_cols].mean(axis=1)

    return df



def main():
    print("=" * 60)
    print("  build_labels_and_crosssectional.py")
    print("=" * 60)

    print(f"\n  Loading {IN_FILE.name} …")
    df = pd.read_csv(IN_FILE, low_memory=False)
    print(f"  {len(df):,} rows × {len(df.columns)} cols")

    if "is_stable" in df.columns:
        before = df["asset_id"].nunique()
        df = df[df["is_stable"] != 1].copy()
        after = df["asset_id"].nunique()
        print(f"  Excluded stablecoins: {before - after} coins removed")
    else:
        STABLE_IDS = {"tether","usd-coin","dai","binance-usd","true-usd",
                      "frax","first-digital-usd","paypal-usd","bfusd","ripple-usd","usd1-wlfi"}
        before = df["asset_id"].nunique()
        df = df[~df["asset_id"].isin(STABLE_IDS)].copy()
        after = df["asset_id"].nunique()
        print(f"  Excluded stablecoins: {before - after} coins removed")

    print(f"  {df['asset_id'].nunique()} coins  |  "
          f"{df['date'].nunique()} days  |  "
          f"{pd.to_datetime(df['date']).dt.to_period('M').nunique()} months")

    print("\n  Step 1: Extracting month-end snapshots …")
    snapshots = get_month_end_snapshots(df)
    print(f"  {len(snapshots):,} month-end rows  "
          f"({snapshots['year_month'].nunique()} months × "
          f"{snapshots['asset_id'].nunique()} coins avg)")

    print("\n  Step 2: Computing forward 30-day risk-adjusted return …")
    print(f"  (looking {FORWARD_DAYS} days ahead for each month-end snapshot)")
    snapshots = compute_forward_sharpe(df, snapshots)
    valid = snapshots["forward_sharpe_30d"].notna().sum()
    total = len(snapshots)
    print(f"  Forward Sharpe computed: {valid}/{total} rows "
          f"({valid/total*100:.1f}%)")
    print(f"  Note: last {FORWARD_DAYS//30} month(s) will have NaN labels "
          f"(no future data yet — correct)")

    print("\n  Step 3: Assigning quadrant labels …")
    snapshots = assign_labels(snapshots)

    label_counts = snapshots["label"].value_counts()
    print(f"\n  Label distribution (all months combined):")
    for lbl in ["Strong Buy", "Buy", "Hold", "Avoid"]:
        n = label_counts.get(lbl, 0)
        pct = n / label_counts.sum() * 100
        bar = "█" * int(pct / 2)
        print(f"    {lbl:<12} {n:>5}  {pct:>5.1f}%  {bar}")

    monthly_dist = (
        snapshots.groupby(["year_month", "label"])
                 .size()
                 .unstack(fill_value=0)
                 .reset_index()
    )
    monthly_dist.to_csv(OUT_DIST, index=False)
    print(f"\n  Monthly label distribution saved → {OUT_DIST.name}")

    print("\n  Step 4: Adding cross-sectional rank + zscore features …")
    snapshots = add_crosssectional_features(snapshots)
    rank_cols  = [c for c in snapshots.columns if c.endswith("_rank") and
                  not c.startswith("market_cap") and c != "forward_sharpe_rank"]
    zscore_cols = [c for c in snapshots.columns if c.endswith("_zscore")]
    print(f"  Added {len(rank_cols)} rank columns + {len(zscore_cols)} zscore columns")
    print("\n  Step 5: Adding composite scores …")
    snapshots = add_composite_scores(snapshots)
    labeled   = snapshots[snapshots["label"].notna()].copy()
    unlabeled = snapshots[snapshots["label"].isna()].copy()
    print(f"\n  Labeled rows  : {len(labeled):,}")
    print(f"  Unlabeled rows: {len(unlabeled):,}  "
          f"(last {FORWARD_DAYS//30} month — no future data, excluded)")

    labeled["year_month"] = labeled["year_month"].astype(str)
    labeled = labeled.sort_values(["year_month", "asset_id"]).reset_index(drop=True)
    labeled.to_csv(OUT_FILE, index=False)
    print(f"\n{'='*60}")
    print(f"✅  Saved: {OUT_FILE.name}")
    print(f"   Rows      : {len(labeled):,}")
    print(f"   Columns   : {len(labeled.columns)}")
    print(f"   Coins     : {labeled['asset_id'].nunique()}")
    print(f"   Months    : {labeled['year_month'].nunique()}")
    print(f"   Range     : {labeled['year_month'].min()} → {labeled['year_month'].max()}")
    print(f"\n   Column breakdown:")
    print(f"   Original features      : ~{len(df.columns)}")
    print(f"   Cross-sectional ranks  : {len(rank_cols)}")
    print(f"   Cross-sectional zscores: {len(zscore_cols)}")
    print(f"   Composite scores       : 3  (momentum, mean_reversion, trend)")
    print(f"   Labels                 : label, forward_return_30d, forward_sharpe_30d")
    print(f"\n   Label encoding for model:")
    print(f"   Strong Buy = 3  |  Buy = 2  |  Hold = 1  |  Avoid = 0")
    print(f"\n   Next step: train XGBoost / LightGBM on model_dataset.csv")
    print(f"   Target column: 'label'")
    print(f"   Time-based split: train on months before 2024, test on 2024+")
    print("="*60)


if __name__ == "__main__":
    main()