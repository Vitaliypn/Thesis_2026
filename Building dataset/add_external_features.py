"""
add_external_features.py
─────────────────────────
Adds to dataset_mcap_fixed.csv:

  1. Bitcoin halving features (offline):
       days_since_halving   — days since last halving
       days_to_halving      — days until next halving
       halving_cycle        — which cycle (1/2/3/4)
       halving_cycle_phase  — 0.0→1.0 progress through current cycle

  2. Google Trends (pytrends, weekly → daily interpolated):
       btc_search_trend     — 0–100 relative search interest
       eth_search_trend     — 0–100 relative search interest

  3. Macro markets (yfinance daily):
       sp500_close          — S&P 500 closing price
       sp500_return_1d      — daily return
       sp500_return_7d      — 7-day return
       nasdaq_close         — NASDAQ closing price
       nasdaq_return_1d
       nasdaq_return_7d
       dxy_close            — DXY (USD index)
       dxy_return_1d
       dxy_return_7d

Saves: dataset_features.csv
"""

from pathlib import Path
import time
import pandas as pd
import numpy as np
import yfinance as yf
from pytrends.request import TrendReq

DYPLOM   = Path(__file__).parent
IN_FILE  = DYPLOM / "dataset_mcap_fixed.csv"
OUT_FILE = DYPLOM / "dataset_features.csv"

HALVINGS = [
    pd.Timestamp("2012-11-28"),
    pd.Timestamp("2016-07-09"),
    pd.Timestamp("2020-05-11"),
    pd.Timestamp("2024-04-20"),
    pd.Timestamp("2028-03-27"),
]


def build_halving_features(dates: pd.Series) -> pd.DataFrame:
    ts = pd.to_datetime(dates)
    rows = []
    for t in ts:
        past    = [h for h in HALVINGS if h <= t]
        future  = [h for h in HALVINGS if h > t]
        last    = past[-1]  if past   else HALVINGS[0]
        nxt     = future[0] if future else HALVINGS[-1]
        cycle   = past.index(last) + 1 if past else 1
        cycle_len = (nxt - last).days
        phase   = (t - last).days / cycle_len if cycle_len > 0 else 0.0
        rows.append({
            "date":               t.strftime("%Y-%m-%d"),
            "days_since_halving": (t - last).days,
            "days_to_halving":    (nxt - t).days,
            "halving_cycle":      cycle,
            "halving_cycle_phase": round(phase, 4),
        })
    return pd.DataFrame(rows)


def fetch_trends(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch weekly Google Trends for bitcoin and ethereum.
    Pytrends returns weekly data for long ranges — we interpolate to daily.
    Fetches in 6-month chunks to stay within pytrends limits.
    """
    pt = TrendReq(hl="en-US", tz=0, timeout=(10, 45))

    all_rows = []
    chunk_start = pd.Timestamp(start_date)
    chunk_end   = pd.Timestamp(end_date)
    step        = pd.DateOffset(months=6)

    current = chunk_start
    while current < chunk_end:
        nxt = min(current + step, chunk_end)
        tf  = f"{current.strftime('%Y-%m-%d')} {nxt.strftime('%Y-%m-%d')}"
        print(f"    Trends chunk: {tf}", end="  ")
        for attempt in range(3):
            try:
                pt.build_payload(["bitcoin", "ethereum"], cat=0,
                                 timeframe=tf, geo="", gprop="")
                df = pt.interest_over_time()
                if not df.empty:
                    df = df.drop(columns=["isPartial"], errors="ignore")
                    df.index = pd.to_datetime(df.index)
                    all_rows.append(df)
                    print(f"✓ {len(df)} weeks")
                else:
                    print("empty")
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg and attempt < 2:
                    wait = 60 * (attempt + 1)
                    print(f"429 — waiting {wait}s ...", end=" ")
                    time.sleep(wait)
                else:
                    print(f"✗ {e}")
                    break
        time.sleep(8)
        current = nxt

    if not all_rows:
        return pd.DataFrame()

    trends = pd.concat(all_rows).reset_index()
    trends = trends[~trends.index.duplicated(keep="last")].reset_index(drop=True)
    trends = trends.rename(columns={
        "date":     "week_date",
        "bitcoin":  "btc_search_trend",
        "ethereum": "eth_search_trend",
    })
    trends["week_date"] = pd.to_datetime(trends["week_date"])
    trends = trends.drop_duplicates(subset=["week_date"], keep="last").reset_index(drop=True)

    for col in ["btc_search_trend", "eth_search_trend"]:
        mx = trends[col].max()
        if mx > 0:
            trends[col] = (trends[col] / mx * 100).round(1)

    date_range = pd.date_range(
        trends["week_date"].min(),
        trends["week_date"].max(),
        freq="D"
    )
    daily = pd.DataFrame({"week_date": date_range})
    daily = daily.merge(trends, on="week_date", how="left")
    daily["btc_search_trend"] = daily["btc_search_trend"].interpolate(method="linear").round(1)
    daily["eth_search_trend"] = daily["eth_search_trend"].interpolate(method="linear").round(1)
    daily = daily.rename(columns={"week_date": "date"})
    daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")

    return daily


def fetch_macro(start_date: str, end_date: str) -> pd.DataFrame:
    tickers = {
        "sp500":  "^GSPC",
        "nasdaq": "^IXIC",
        "dxy":    "DX-Y.NYB",
    }
    frames = []
    for name, ticker in tickers.items():
        print(f"    {name} ({ticker})", end="  ")
        try:
            raw = yf.download(ticker, start=start_date, end=end_date,
                              progress=False, auto_adjust=True)
            if raw.empty:
                print("✗ empty")
                continue
            # Handle MultiIndex columns from yfinance
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            s = raw["Close"].copy()
            s.index = pd.to_datetime(s.index)
            df = pd.DataFrame({
                "date":           s.index.strftime("%Y-%m-%d"),
                f"{name}_close":  s.values,
            })
            df[f"{name}_return_1d"] = s.pct_change(1).values
            df[f"{name}_return_7d"] = s.pct_change(7).values
            frames.append(df)
            print(f"✓ {len(df)} days  latest={s.iloc[-1]:,.2f}")
        except Exception as e:
            print(f"✗ {e}")

    if not frames:
        return pd.DataFrame()

    result = frames[0]
    for f in frames[1:]:
        result = result.merge(f, on="date", how="outer")
    return result.sort_values("date").reset_index(drop=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  add_external_features.py")
    print("=" * 62)
    print(f"\n  Loading {IN_FILE.name} ...")
    df = pd.read_csv(IN_FILE, low_memory=False)
    print(f"  {len(df):,} rows | {df['asset_id'].nunique()} coins | {len(df.columns)} cols")
    original_rows  = len(df)
    original_dates = set(df["date"].unique())

    date_min = df["date"].min()
    date_max = df["date"].max()
    print(f"  Date range: {date_min} → {date_max}")

    print("\n  Step 1: Computing halving features ...")
    all_dates = df["date"].drop_duplicates().sort_values()
    halving_df = build_halving_features(all_dates)
    df = df.merge(halving_df, on="date", how="left")
    sample = halving_df[halving_df["date"].isin(
        ["2020-05-11","2020-05-12","2024-04-20","2024-04-21","2019-06-01"]
    )]
    for _, r in sample.iterrows():
        print(f"    {r['date']}  since={r['days_since_halving']:>5}d  "
              f"to={r['days_to_halving']:>5}d  "
              f"cycle={r['halving_cycle']}  phase={r['halving_cycle_phase']:.3f}")

    print("\n  Step 2: Fetching Google Trends ...")
    trends_df = fetch_trends(date_min, date_max)
    if not trends_df.empty:
        df = df.merge(trends_df, on="date", how="left")
        cov = df["btc_search_trend"].notna().mean() * 100
        print(f"  Merged trends  coverage={cov:.1f}%")
        for d in ["2021-01-01","2021-11-01","2022-06-01","2024-01-01"]:
            row = trends_df[trends_df["date"]==d]
            if not row.empty:
                r = row.iloc[0]
                print(f"    {d}  btc={r['btc_search_trend']:.0f}  eth={r['eth_search_trend']:.0f}")
    else:
        print("  ✗ No trends data")
        df["btc_search_trend"] = np.nan
        df["eth_search_trend"] = np.nan

    print("\n  Step 3: Fetching macro data ...")
    macro_df = fetch_macro(date_min, date_max)
    if not macro_df.empty:
        existing_dates = set(df["date"].unique())
        macro_df = macro_df[macro_df["date"].isin(existing_dates)]
        df = df.merge(macro_df, on="date", how="left")
        macro_cols = [c for c in macro_df.columns if c != "date"]
        date_sorted = df.sort_values("date")
        for col in macro_cols:
            df[col] = (date_sorted.groupby("asset_id")[col]
                       .transform(lambda s: s.ffill()))

        for col in ["sp500_close","nasdaq_close","dxy_close"]:
            cov = df[col].notna().mean() * 100
            print(f"  {col:<20} coverage={cov:.1f}%")
    else:
        print("  ✗ No macro data")

    df = df[df["date"].isin(original_dates)].copy()
    df = df.sort_values(["asset_id","date"]).reset_index(drop=True)
    assert len(df) == original_rows, f"Row count changed: {original_rows} -> {len(df)}"
    df.to_csv(OUT_FILE, index=False)

    new_cols = [
        "days_since_halving","days_to_halving","halving_cycle","halving_cycle_phase",
        "btc_search_trend","eth_search_trend",
        "sp500_close","sp500_return_1d","sp500_return_7d",
        "nasdaq_close","nasdaq_return_1d","nasdaq_return_7d",
        "dxy_close","dxy_return_1d","dxy_return_7d",
    ]
    print(f"\n{'='*62}")
    print(f"  Saved: {OUT_FILE.name}  |  {len(df):,} rows  |  {len(df.columns)} cols")
    print(f"\n  New columns:")
    for col in new_cols:
        if col not in df.columns:
            continue
        cov = df[col].notna().mean() * 100
        mn, mx = df[col].min(), df[col].max()
        print(f"    {col:<30} {cov:>5.1f}%  [{mn:.4g} - {mx:.4g}]")
    print("=" * 62)


if __name__ == "__main__":
    main()
