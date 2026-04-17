"""
add_new_features.py
─────────────────
Adds three layers of new features to dataset_final.csv:

  [A] Calculated features — no API, derived from existing data
      coin_age_days, price_vs_ath, price_vs_atl, drawdown_from_90d_peak,
      volume_vs_30d_avg, volume_spike, consecutive_up_days,
      consecutive_down_days, range_position_7d, range_position_30d

  [B] Market cap + coin rank — CoinGecko /coins/markets (free, no key)
      market_cap_usd, market_cap_rank, circulating_supply, total_supply,
      max_supply, fdv_usd, circulating_to_max_ratio

  [C] TVL for DeFi coins — DefiLlama (free, full history, no key)
      tvl_usd, tvl_change_7d, tvl_change_30d, mcap_to_tvl

Output: dataset_enriched.csv (drop-in replacement for dataset_final.csv)

Usage:
    python add_new_features.py
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests


DYPLOM    = Path(__file__).parent
IN_FILE   = DYPLOM / "dataset_final.csv"
OUT_FILE  = DYPLOM / "dataset_enriched.csv"

COINGECKO = "https://api.coingecko.com/api/v3"
DEFILLAMA = "https://api.llama.fi"


def _get(url: str, params: dict = None, retries: int = 5, delay: float = 1.5) -> dict | list | None:
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 30))
                print(f"    ⏳ Rate limited — waiting {wait}s …")
                time.sleep(wait)
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            time.sleep(delay)
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                print(f"    ✗ Failed: {e}")
                return None
            time.sleep(10 * (attempt + 1))
    return None


def add_calculated_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*60)
    print("  PART A — Calculated Features")
    print("="*60)

    results = []

    for asset_id, grp in df.groupby("asset_id"):
        g = grp.sort_values("date").copy()
        c = g["close"]
        v = g["volume"]
        first_date = pd.to_datetime(g["date"].iloc[0])
        g["coin_age_days"] = (pd.to_datetime(g["date"]) - first_date).dt.days
        g["ath"] = c.cummax()
        g["atl"] = c.cummin()
        g["price_vs_ath"] = (c - g["ath"]) / g["ath"]
        g["price_vs_atl"] = (c - g["atl"]) / g["atl"]

        peak_90d = c.rolling(90, min_periods=1).max()
        g["drawdown_from_90d_peak"] = (c - peak_90d) / peak_90d
        vol_30d_avg = v.rolling(30, min_periods=5).mean()
        g["volume_vs_30d_avg"] = (v - vol_30d_avg) / vol_30d_avg.replace(0, float("nan"))
        g["volume_spike"] = (g["volume_vs_30d_avg"] > 2.0).astype(int)
        direction = (c.diff() > 0).astype(int)
        consec_up   = []
        consec_down = []
        up_count = down_count = 0
        for d in direction:
            if d == 1:
                up_count += 1
                down_count = 0
            else:
                down_count += 1
                up_count = 0
            consec_up.append(up_count)
            consec_down.append(down_count)
        g["consecutive_up_days"]   = consec_up
        g["consecutive_down_days"] = consec_down
        for window in [7, 30]:
            lo = c.rolling(window, min_periods=2).min()
            hi = c.rolling(window, min_periods=2).max()
            span = (hi - lo).replace(0, float("nan"))
            g[f"range_position_{window}d"] = (c - lo) / span

        g = g.drop(columns=["ath", "atl"])
        results.append(g)
        print(f"  ✓ {asset_id:<35}", end="\r")

    print()
    df_out = pd.concat(results, ignore_index=True)

    new_cols = ["coin_age_days", "price_vs_ath", "price_vs_atl",
                "drawdown_from_90d_peak", "volume_vs_30d_avg", "volume_spike",
                "consecutive_up_days", "consecutive_down_days",
                "range_position_7d", "range_position_30d"]
    print(f"  ✅ Added {len(new_cols)} calculated features")
    return df_out



def fetch_coingecko_market_data(asset_ids: list[str]) -> pd.DataFrame:
    """
    Fetches current market cap, rank, supply from CoinGecko /coins/markets.
    Free, no API key. Returns one row per asset_id.
    CoinGecko limit: 250 per page. We page through to cover all coins.
    """
    print("\n" + "="*60)
    print("  PART B — Market Cap + Rank (CoinGecko)")
    print("="*60)

    rows = []
    page = 1
    per_page = 250

    while True:
        print(f"  Fetching page {page} …")
        data = _get(f"{COINGECKO}/coins/markets", params={
            "vs_currency": "usd",
            "order":       "market_cap_desc",
            "per_page":    per_page,
            "page":        page,
            "sparkline":   "false",
        }, delay=2.0)

        if not data:
            break

        for coin in data:
            rows.append({
                "asset_id":               coin.get("id"),
                "market_cap_usd":         coin.get("market_cap"),
                "market_cap_rank":        coin.get("market_cap_rank"),
                "fully_diluted_valuation":coin.get("fully_diluted_valuation"),
                "circulating_supply":     coin.get("circulating_supply"),
                "total_supply":           coin.get("total_supply"),
                "max_supply":             coin.get("max_supply"),
                "ath_usd":                coin.get("ath"),
                "ath_change_pct":         coin.get("ath_change_percentage"),
            })

        # Check if any of our target coins are still missing
        fetched_ids = {r["asset_id"] for r in rows}
        still_missing = set(asset_ids) - fetched_ids
        if not still_missing or len(data) < per_page:
            break
        page += 1

    df = pd.DataFrame(rows)
    df = df[df["asset_id"].isin(asset_ids)].copy()

    df["circulating_to_max_ratio"] = (
        df["circulating_supply"] / df["max_supply"].replace(0, float("nan"))
    )

    def cap_tier(mcap):
        if pd.isna(mcap): return None
        if mcap > 10e9:   return "large"
        if mcap > 1e9:    return "mid"
        if mcap > 100e6:  return "small"
        return "micro"
    df["cap_tier"] = df["market_cap_usd"].apply(cap_tier)

    covered = len(df)
    print(f"  ✅ Got market cap data for {covered}/{len(asset_ids)} coins")
    if covered < len(asset_ids):
        missing = set(asset_ids) - set(df["asset_id"])
        print(f"     Missing: {', '.join(sorted(missing)[:10])}{'...' if len(missing)>10 else ''}")

    return df


DEFILLAMA_SLUGS = {
    "aave":              "AAVE",
    "uniswap":           "Uniswap",
    "curve-dao-token":   "Curve",
    "compound-governance-token": "Compound",
    "maker":             "MakerDAO",
    "lido-dao":          "Lido",
    "pancakeswap-token": "PancakeSwap",
    "sushiswap":         "Sushiswap",
    "yearn-finance":     "Yearn Finance",
    "synthetix-network-token": "Synthetix",
    "balancer":          "Balancer",
    "1inch":             "1inch Network",
    "thorchain":         "THORChain",
    "convex-finance":    "Convex Finance",
    "frax-share":        "Frax",
    "gmx":               "GMX",
    "dydx":              "dYdX",
    "rocket-pool":       "Rocket Pool",
    "chainlink":         "Chainlink",
    "the-graph":         "The Graph",
    "ren":               "Ren",
    "band-protocol":     "Band Protocol",
    "uma":               "UMA",
    "ankr":              "Ankr",
    "perpetual-protocol":"Perpetual Protocol",
    "osmosis":           "Osmosis",
    "jito-governance-token": "Jito",
}


def fetch_defillama_tvl(asset_ids: list[str]) -> pd.DataFrame:
    """
    Fetches historical daily TVL for DeFi protocol tokens from DefiLlama.
    Returns a DataFrame with columns: asset_id, date, tvl_usd
    """
    print("\n" + "="*60)
    print("  PART C — TVL from DefiLlama")
    print("="*60)

    # First, get the full protocol list to find correct slugs
    print("  Fetching DefiLlama protocol list …")
    protocols = _get(f"{DEFILLAMA}/protocols", delay=1.0)
    if not protocols:
        print("  ✗ Could not fetch protocol list")
        return pd.DataFrame()

    # Build a name→slug lookup from the live list
    name_to_slug = {}
    for p in protocols:
        name_to_slug[p.get("name", "").lower()] = p.get("slug") or p.get("name")

    all_tvl_rows = []
    target_coins = [aid for aid in asset_ids if aid in DEFILLAMA_SLUGS]
    print(f"  Fetching TVL for {len(target_coins)} DeFi protocol coins …")

    for asset_id in target_coins:
        friendly_name = DEFILLAMA_SLUGS[asset_id]
        # Try to find the right slug
        slug = name_to_slug.get(friendly_name.lower())
        if not slug:
            # Try partial match
            for name, s in name_to_slug.items():
                if friendly_name.lower() in name:
                    slug = s
                    break

        if not slug:
            print(f"    ⚠  {asset_id}: slug not found in DefiLlama")
            continue

        data = _get(f"{DEFILLAMA}/protocol/{slug}", delay=1.5)
        if not data:
            print(f"    ⚠  {asset_id}: no data returned")
            continue

        tvl_data = data.get("tvl", [])
        if not tvl_data:
            print(f"    ⚠  {asset_id}: empty TVL series")
            continue

        for entry in tvl_data:
            ts   = entry.get("date")
            tvl  = entry.get("totalLiquidityUSD") or entry.get("tvl")
            if ts is None or tvl is None:
                continue
            all_tvl_rows.append({
                "asset_id": asset_id,
                "date":     pd.to_datetime(ts, unit="s", utc=True).strftime("%Y-%m-%d"),
                "tvl_usd":  float(tvl),
            })

        print(f"    ✓ {asset_id:<35} {len(tvl_data)} days", end="\r")

    print()

    if not all_tvl_rows:
        print("  ✗ No TVL data collected")
        return pd.DataFrame()

    tvl_df = pd.DataFrame(all_tvl_rows)
    tvl_df = tvl_df.sort_values(["asset_id", "date"]).reset_index(drop=True)

    # Compute TVL changes and rolling features per coin
    enriched = []
    for asset_id, grp in tvl_df.groupby("asset_id"):
        grp = grp.sort_values("date").copy()
        grp["tvl_change_7d"]  = grp["tvl_usd"].pct_change(7)
        grp["tvl_change_30d"] = grp["tvl_usd"].pct_change(30)
        enriched.append(grp)

    tvl_df = pd.concat(enriched, ignore_index=True)
    print(f"  ✅ TVL data: {len(tvl_df):,} rows for {tvl_df['asset_id'].nunique()} protocols")
    return tvl_df



def main():
    print("=" * 60)
    print("  enrich_dataset.py")
    print("  Adding calculated features + market cap + TVL")
    print("=" * 60)

    # ── Load base dataset ─────────────────────────────────────────────
    print(f"\n  Loading {IN_FILE.name} …")
    df = pd.read_csv(IN_FILE, low_memory=False)
    print(f"  {len(df):,} rows × {len(df.columns)} columns")
    print(f"  {df['asset_id'].nunique()} coins, {df['date'].nunique()} dates")

    asset_ids = df["asset_id"].unique().tolist()
    df = add_calculated_features(df)
    mcap_df = fetch_coingecko_market_data(asset_ids)
    if not mcap_df.empty:
        mcap_cols = [c for c in mcap_df.columns if c != "asset_id"]
        df = df.merge(mcap_df[["asset_id"] + mcap_cols], on="asset_id", how="left")
        print(f"\n  Market cap columns added: {mcap_cols}")
    tvl_df = fetch_defillama_tvl(asset_ids)

    if not tvl_df.empty:
        df = df.merge(
            tvl_df[["asset_id", "date", "tvl_usd", "tvl_change_7d", "tvl_change_30d"]],
            on=["asset_id", "date"],
            how="left"
        )
        # mcap/tvl ratio — fundamental valuation metric
        # (only meaningful for coins that have both)
        if "market_cap_usd" in df.columns:
            df["mcap_to_tvl"] = df["market_cap_usd"] / df["tvl_usd"].replace(0, float("nan"))
        print(f"\n  TVL columns added: tvl_usd, tvl_change_7d, tvl_change_30d, mcap_to_tvl")

    # ── Save ──────────────────────────────────────────────────────────
    df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)

    # ── Summary ───────────────────────────────────────────────────────
    null_rates = df.isnull().mean().sort_values(ascending=False)
    high_null  = null_rates[null_rates > 0.5]

    print("\n" + "="*60)
    print(f"✅  Saved: {OUT_FILE.name}")
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {len(df.columns)}  (+{len(df.columns) - df.shape[1] + len(df.columns)} new)")
    print(f"   Coins   : {df['asset_id'].nunique()}")
    print(f"   Range   : {df['date'].min()} → {df['date'].max()}")
    print(f"\n   New features added:")
    print(f"   Part A (calculated) : coin_age_days, price_vs_ath, drawdown_from_90d_peak,")
    print(f"                         volume_vs_30d_avg, volume_spike, consecutive_up/down_days,")
    print(f"                         range_position_7d/30d, price_vs_atl")
    print(f"   Part B (CoinGecko)  : market_cap_usd, market_cap_rank, circulating_supply,")
    print(f"                         max_supply, fdv_usd, circulating_to_max_ratio, cap_tier")
    print(f"   Part C (DefiLlama)  : tvl_usd, tvl_change_7d, tvl_change_30d, mcap_to_tvl")
    if not high_null.empty:
        print(f"\n   Columns >50% null (expected for DeFi-only features):")
        for col, rate in high_null.items():
            print(f"     {col:<30} {rate*100:.0f}% null")
    print("="*60)


if __name__ == "__main__":
    main()
