import os
import pandas as pd
import requests
import time
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["LUNARCRUSH_API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

print("Loading dataset...")
df = pd.read_csv("dataset_model.csv")
symbols = list(df["asset_id"].unique())
print(f"Found {len(symbols)} unique assets")

print("Fetching LunarCrush coin list...")
r = requests.get("https://lunarcrush.com/api4/public/coins/list/v1", headers=HEADERS)
if r.status_code != 200:
    raise RuntimeError(f"Coin list failed: {r.status_code} {r.text[:200]}")

coin_list = r.json().get("data", [])
slug_map   = {str(c.get("id", "")).lower(): c for c in coin_list}
symbol_map = {str(c.get("symbol", "")).upper(): c for c in coin_list}
name_map   = {str(c.get("name", "")).lower(): c for c in coin_list}

print("Fetching CoinGecko slug → ticker map...")
cg_r = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=15)
cg_slug_to_ticker = {
    c["id"]: c["symbol"].upper()
    for c in (cg_r.json() if cg_r.status_code == 200 else [])
    if c.get("id") and c.get("symbol")
}

def resolve(asset_id):
    if asset_id in slug_map:
        return slug_map[asset_id]
    ticker = cg_slug_to_ticker.get(asset_id, "")
    if ticker and ticker in symbol_map:
        return symbol_map[ticker]
    if asset_id.upper() in symbol_map:
        return symbol_map[asset_id.upper()]
    if asset_id.lower() in name_map:
        return name_map[asset_id.lower()]
    base = asset_id.split("-")[0].lower()
    if base in slug_map:
        return slug_map[base]
    if base.upper() in symbol_map:
        return symbol_map[base.upper()]
    return None

lc_rows = []
not_found = []

for i, asset_id in enumerate(symbols):
    print(f"[{i+1}/{len(symbols)}] {asset_id}", end=" ... ")

    coin_meta = resolve(asset_id)
    if coin_meta is None:
        print("NOT FOUND")
        not_found.append(asset_id)
        continue

    lc_id = coin_meta.get("id")
    url = f"https://lunarcrush.com/api4/public/coins/{lc_id}/v1"

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 429:
            print("rate limited, sleeping 60s...", end=" ")
            time.sleep(60)
            r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code != 200:
            print(f"HTTP {r.status_code}")
            continue

        data = r.json().get("data", {})
        coin = data[0] if isinstance(data, list) else data
        if not coin:
            print("empty")
            continue

        lc_rows.append({
            "asset_id":            asset_id,
            "galaxy_score":        coin.get("galaxy_score"),
            "alt_rank":            coin.get("alt_rank"),
            "lc_circulating_supply": coin.get("circulating_supply"),
            "lc_market_cap":       coin.get("market_cap"),      # LC's own mcap for cross-check
        })
        print(
            f"OK | galaxy={coin.get('galaxy_score')} "
            f"alt_rank={coin.get('alt_rank')} "
            f"circ_supply={coin.get('circulating_supply')}"
        )

    except Exception as e:
        print(f"ERROR: {e}")

    time.sleep(0.5)

print(f"\nCollected: {len(lc_rows)}/{len(symbols)}")
lc_df = pd.DataFrame(lc_rows)
df = df.merge(lc_df, on="asset_id", how="left")

print("Overwriting circulating_supply from LunarCrush where available...")
lc_supply_coverage = df["lc_circulating_supply"].notna().sum()
print(f"  LC supply available for: {lc_supply_coverage}/{len(df)} rows")

df["circulating_supply_original"] = df["circulating_supply"]
df["circulating_supply"] = df["lc_circulating_supply"].combine_first(df["circulating_supply"])

print("Recalculating market_cap_usd = circulating_supply * close...")
df["market_cap_usd_original"] = df["market_cap_usd"]
df["market_cap_usd"] = df["circulating_supply"] * df["close"]

mask = df["market_cap_usd_original"].notna() & df["market_cap_usd"].notna()
diff_pct = (
    (df.loc[mask, "market_cap_usd"] - df.loc[mask, "market_cap_usd_original"]).abs()
    / df.loc[mask, "market_cap_usd_original"].replace(0, float("nan"))
)
print(f"  Median mcap change:  {diff_pct.median()*100:.2f}%")
print(f"  Rows with >10% change: {(diff_pct > 0.10).sum()}")
print(f"  Rows with >50% change: {(diff_pct > 0.50).sum()}")

df = df.drop(columns=["lc_circulating_supply", "lc_market_cap",
                       "circulating_supply_original", "market_cap_usd_original"])

df.to_csv("dataset_with_social.csv", index=False)

print("\n=== DATA COVERAGE ===")
for col in ["galaxy_score", "alt_rank", "circulating_supply", "market_cap_usd"]:
    n = df[col].notna().sum()
    print(f"  {col}: {n}/{len(df)} ({n/len(df)*100:.1f}%)")

if not_found:
    print(f"\nUnmatched ({len(not_found)}): {', '.join(not_found)}")

print("\nSaved → dataset_with_social.csv")