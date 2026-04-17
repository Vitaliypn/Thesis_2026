"""
add_top_assets.py
──────────────────────
Builds a comprehensive assets.csv containing every coin that has been
in the CoinGecko top 150 by market cap at ANY point from 2020 to today.

Strategy:
  1. Fetch the current top coinns (live API).
  2. For each year 2020–(current year), fetch a Jan-1 snapshot via
     CoinGecko's /coins/markets?date= endpoint to get the historical top-200.
  3. Union all results, deduplicate on asset_id.
  4. Enrich each coin with metadata (website, launch date, category, etc.)
     via /coins/{id}.
  5. Write to assets.csv

Usage:
    python add_top_assets.py
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR   = Path(__file__).parent
ASSETS_FILE  = OUTPUT_DIR / "assets.csv"
CGECKO_BASE  = "https://api.coingecko.com/api/v3"
START_YEAR   = 2020
TOP_N        = 250

ASSETS_COLS = [
    "asset_id", "name", "symbol", "category", "project_slug",
    "website", "whitepaper_url", "launch_date",
    "is_token", "parent_chain", "layer", "status", "description",
]

# ── HTTP helper ───────────────────────────────────────────────────────────────

def _get(url: str, params: dict = None, retries: int = 6) -> dict | list:
    """GET with rate-limit handling and exponential backoff."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"    ⏳ Rate-limited — waiting {wait}s …")
                time.sleep(wait)
                continue
            if r.status_code in (502, 503, 504):
                wait = 20 * (attempt + 1)
                print(f"    ⚠  {r.status_code} — retrying in {wait}s …")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"    ✗ Request error: {e} — retrying in {wait}s …")
            time.sleep(wait)
    return {}



def fetch_current_top200() -> list[dict]:
    """Fetch current top coins by market cap (2 pages × 100)."""
    coins = []
    for page in (1, 2):
        print(f"  📡 Fetching current top coins, page {page}/2 …")
        data = _get(f"{CGECKO_BASE}/coins/markets", params={
            "vs_currency": "usd",
            "order":       "market_cap_desc",
            "per_page":    100,
            "page":        page,
            "sparkline":   False,
        })
        if isinstance(data, list):
            coins.extend(data)
        time.sleep(2)
    return coins[:TOP_N]


def fetch_historical_top200(date_str: str) -> list[dict]:
    """
    Fetch top coins as of a specific date using the date param.
    date_str format: 'dd-mm-yyyy'  e.g. '01-01-2022'
    """
    coins = []
    for page in (1, 2):
        print(f"  📡 Fetching historical top-200 for {date_str}, page {page}/2 …")
        data = _get(f"{CGECKO_BASE}/coins/markets", params={
            "vs_currency": "usd",
            "order":       "market_cap_desc",
            "per_page":    100,
            "page":        page,
            "sparkline":   False,
            "date":        date_str,   # CoinGecko accepts dd-mm-yyyy for snapshots
        })
        if isinstance(data, list) and data:
            coins.extend(data)
        elif not isinstance(data, list) or not data:
            # Some dates return empty — that's fine
            print(f"    ℹ  No data returned for {date_str} page {page}")
        time.sleep(2.5)
    return coins[:TOP_N]


def fetch_coin_metadata(coin_id: str) -> dict:
    """Fetch detailed metadata for a single coin."""
    try:
        data = _get(f"{CGECKO_BASE}/coins/{coin_id}", params={
            "localization":   False,
            "tickers":        False,
            "market_data":    False,
            "community_data": False,
            "developer_data": False,
            "sparkline":      False,
        })
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def parse_basic(coins: list[dict]) -> dict[str, dict]:
    """Turn a /coins/markets list into a dict keyed by coin id."""
    result = {}
    for c in coins:
        cid = c.get("id", "").strip()
        if cid:
            result[cid] = {
                "asset_id":    cid,
                "name":        c.get("name", ""),
                "symbol":      (c.get("symbol") or "").upper(),
                "project_slug": cid,
                "status":      "active",
            }
    return result


def enrich_with_metadata(coin_id: str, base: dict, meta: dict) -> list:
    """Merge /coins/{id} metadata into a row for assets.csv."""
    links    = meta.get("links", {})
    homepage = (links.get("homepage") or [""])[0] or ""
    whitepaper = ""
    wlinks = links.get("whitepaper") or []
    if isinstance(wlinks, list) and wlinks:
        whitepaper = wlinks[0]
    elif isinstance(wlinks, str):
        whitepaper = wlinks

    genesis = meta.get("genesis_date") or ""
    categories = meta.get("categories") or []
    category = categories[0] if categories else ""

    platforms = meta.get("platforms") or {}
    is_token  = "yes" if len(platforms) > 0 else "no"
    parent_chain = ""
    for chain, addr in platforms.items():
        if chain and addr:
            parent_chain = chain
            break

    description_raw = meta.get("description", {})
    description = ""
    if isinstance(description_raw, dict):
        description = (description_raw.get("en") or "")[:300]
    elif isinstance(description_raw, str):
        description = description_raw[:300]

    return [
        coin_id,
        base.get("name", ""),
        base.get("symbol", ""),
        category,
        coin_id, 
        homepage,
        whitepaper,
        genesis,
        is_token,
        parent_chain,
        "",
        "active",
        description,
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def run(output_dir: Path = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_path = output_dir / "assets.csv"

    print("=" * 60)
    print("  build_top200_assets.py")
    print(f"  Output → {assets_path}")
    print("=" * 60)

    all_coins: dict[str, dict] = {}

    print("\n▶  [1/3] Fetching current top coins …")
    current = fetch_current_top200()
    all_coins.update(parse_basic(current))
    print(f"   Current top-coins: {len(current)} coins  |  running total: {len(all_coins)}")

    current_year = datetime.now(timezone.utc).year
    years = list(range(START_YEAR, current_year))

    snapshot_dates = []
    for y in years:
        snapshot_dates.append(f"01-01-{y}")
        snapshot_dates.append(f"01-07-{y}")

    print(f"\n▶  [1/3] Fetching historical snapshots ({len(snapshot_dates)} dates) …")
    for date_str in snapshot_dates:
        hist = fetch_historical_top200(date_str)
        before = len(all_coins)
        all_coins.update(parse_basic(hist))
        new_found = len(all_coins) - before
        print(f"   {date_str}: {len(hist)} coins returned  |  {new_found} new  |  total: {len(all_coins)}")

    print(f"\n✔  Total unique coins ever in top-200 since {START_YEAR}: {len(all_coins)}")

    print(f"\n▶  [2/3] Enriching {len(all_coins)} coins with metadata …")
    print("   (This calls /coins/{{id}} once per coin — may take a few minutes)")

    rows = []
    coin_ids = sorted(all_coins.keys())

    for i, cid in enumerate(coin_ids, 1):
        print(f"   [{i:>3}/{len(coin_ids)}] {cid} …", end=" ")
        meta = fetch_coin_metadata(cid)
        if meta:
            row = enrich_with_metadata(cid, all_coins[cid], meta)
            print("✓")
        else:
            base = all_coins[cid]
            row = [
                cid, base.get("name",""), base.get("symbol",""),
                "","",  "","","","","","","active","",
            ]
            print("⚠  (used basic info)")
        rows.append(row)
        time.sleep(1.2)

    print(f"\n▶  [3/3] Writing assets.csv …")
    new_df = pd.DataFrame(rows, columns=ASSETS_COLS)
    new_df["symbol"] = new_df["symbol"].str.upper()

    if assets_path.exists():
        existing_df = pd.read_csv(assets_path)
        combined = (
            pd.concat([existing_df, new_df], ignore_index=True)
            .drop_duplicates(subset=["asset_id"], keep="last")
            .sort_values("symbol")
            .reset_index(drop=True)
        )
        print(f"   Merged with existing file ({len(existing_df)} rows → {len(combined)} total)")
    else:
        combined = new_df.sort_values("symbol").reset_index(drop=True)

    combined.to_csv(assets_path, index=False)

    print(f"\n{'='*60}")
    print(f"✅  Done!  {len(combined)} assets written to:")
    print(f"   {assets_path}")
    print(f"{'='*60}")
    return output_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Build assets.csv from top-200 coins per year since 2020"
    )
    p.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help=f"Where to write assets.csv (default: {OUTPUT_DIR})",
    )
    args = p.parse_args()
    run(Path(args.output_dir))
