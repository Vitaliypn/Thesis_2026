"""
add_derivatives.py
─────────────────────
Fetches derivatives & market data from CoinGlass V4 API and merges
into dataset_enriched.csv.

Base URL : https://open-api-v4.coinglass.com
Prefix   : /api/   (NOT /public/ — that was wrong)
Auth     : CG-API-KEY header

COIN-LEVEL features (per asset × date):
    funding_rate              OI-weighted funding rate
    funding_rate_7d_avg       7-day rolling mean
    funding_rate_30d_cum      30-day cumulative cost to hold long
    oi_usd                    aggregated open interest USD
    oi_change_1d / oi_change_7d
    liq_long_usd / liq_short_usd / liq_total_usd
    liq_ratio                 >0.5 = longs getting squeezed
    ls_ratio                  global long/short account ratio
    ls_ratio_7d_avg
    top_ls_account_ratio      top trader account ratio
    top_ls_position_ratio     top trader position ratio
    taker_buy_ratio           >0.5 = aggressive buying
    taker_buy_ratio_7d_avg

GLOBAL features (date-level, broadcast to all coins):
    altcoin_season_index
    stablecoin_mcap / stablecoin_mcap_change_7d
    btc_nupl
    btc_active_addresses
    coinbase_premium
    btc_dominance_cg
    etf_net_flow_usd / etf_total_aum

Usage:
    python add_derivatives.py
"""

from __future__ import annotations
import os
import time
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()


DYPLOM   = Path(__file__).parent
IN_FILE  = DYPLOM / "dataset_enriched.csv"
OUT_FILE = DYPLOM / "dataset_enriched.csv"

API_KEY = os.environ["COINGLASS_API_KEY"]
BASE    = "https://open-api-v4.coinglass.com/api"
HEADERS = {"CG-API-KEY": API_KEY, "Accept": "application/json"}

DELAY = 2.1
LIMIT = 2000

SYMBOL_MAP: dict[str, str] = {
    "bitcoin":                    "BTC",
    "ethereum":                   "ETH",
    "binancecoin":                "BNB",
    "ripple":                     "XRP",
    "cardano":                    "ADA",
    "solana":                     "SOL",
    "polkadot":                   "DOT",
    "dogecoin":                   "DOGE",
    "avalanche-2":                "AVAX",
    "shiba-inu":                  "SHIB",
    "polygon":                    "MATIC",
    "litecoin":                   "LTC",
    "cosmos":                     "ATOM",
    "chainlink":                  "LINK",
    "uniswap":                    "UNI",
    "stellar":                    "XLM",
    "near":                       "NEAR",
    "algorand":                   "ALGO",
    "tron":                       "TRX",
    "vechain":                    "VET",
    "filecoin":                   "FIL",
    "internet-computer":          "ICP",
    "fantom":                     "FTM",
    "axie-infinity":              "AXS",
    "theta-token":                "THETA",
    "elrond-erd-2":               "EGLD",
    "hedera-hashgraph":           "HBAR",
    "the-sandbox":                "SAND",
    "decentraland":               "MANA",
    "aave":                       "AAVE",
    "maker":                      "MKR",
    "compound-governance-token":  "COMP",
    "curve-dao-token":            "CRV",
    "synthetix-network-token":    "SNX",
    "yearn-finance":              "YFI",
    "sushiswap":                  "SUSHI",
    "1inch":                      "1INCH",
    "pancakeswap-token":          "CAKE",
    "the-graph":                  "GRT",
    "basic-attention-token":      "BAT",
    "chiliz":                     "CHZ",
    "enjincoin":                  "ENJ",
    "flow":                       "FLOW",
    "waves":                      "WAVES",
    "neo":                        "NEO",
    "monero":                     "XMR",
    "zcash":                      "ZEC",
    "dash":                       "DASH",
    "ethereum-classic":           "ETC",
    "bitcoin-cash":               "BCH",
    "tezos":                      "XTZ",
    "thorchain":                  "RUNE",
    "terra-luna":                 "LUNA",
    "terra-luna-classic":         "LUNC",
    "crypto-com-chain":           "CRO",
    "kava":                       "KAVA",
    "zilliqa":                    "ZIL",
    "harmony":                    "ONE",
    "icon":                       "ICX",
    "band-protocol":              "BAND",
    "uma":                        "UMA",
    "ocean-protocol":             "OCEAN",
    "ren":                        "REN",
    "loopring":                   "LRC",
    "balancer":                   "BAL",
    "ankr":                       "ANKR",
    "storj":                      "STORJ",
    "arweave":                    "AR",
    "helium":                     "HNT",
    "celo":                       "CELO",
    "convex-finance":             "CVX",
    "lido-dao":                   "LDO",
    "rocket-pool":                "RPL",
    "frax-share":                 "FXS",
    "gmx":                        "GMX",
    "dydx":                       "DYDX",
    "osmosis":                    "OSMO",
    "injective-protocol":         "INJ",
    "apecoin":                    "APE",
    "stepn":                      "GMT",
    "gala":                       "GALA",
    "immutable-x":                "IMX",
    "fetch-ai":                   "FET",
    "singularitynet":             "AGIX",
    "pepe":                       "PEPE",
    "floki":                      "FLOKI",
    "the-open-network":           "TON",
    "decred":                     "DCR",
    "nano":                       "XNO",
    "iota":                       "MIOTA",
    "arbitrum":                   "ARB",
    "optimism":                   "OP",
    "sui":                        "SUI",
    "aptos":                      "APT",
}


def _get(path: str, params: dict = None) -> list | dict | None:
    url = f"{BASE}/{path}"
    for attempt in range(4):
        try:
            r = requests.get(url, headers=HEADERS, params=params or {}, timeout=30)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 60))
                print(f"\n    ⏳ Rate limited — waiting {wait}s")
                time.sleep(wait)
                continue
            if r.status_code in (401, 403):
                print(f"\n    ✗ Auth error on {path}")
                return None
            if r.status_code in (400, 404):
                return None
            r.raise_for_status()
            body = r.json()
            if isinstance(body, dict):
                if body.get("code") not in (0, "0", "200", 200, None):
                    return None
                return body.get("data")
            return body
        except Exception:
            if attempt == 3:
                return None
            time.sleep(10 * (attempt + 1))
    return None


def _ts(ts) -> str:
    return pd.to_datetime(int(ts), unit="ms", utc=True).strftime("%Y-%m-%d")



def fetch_funding_rate(symbol: str) -> pd.DataFrame:
    data = _get("futures/fundingRate/oi-weight-ohlc-history",
                {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        data = _get("futures/fundingRate/ohlc-history",
                    {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()

    rows = []
    for d in data:
        t = d.get("t") or d.get("time") or d.get("createTime")
        c = d.get("c") or d.get("close") or d.get("value")
        if t and c is not None:
            rows.append({"date": _ts(t), "funding_rate": float(c)})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["funding_rate_7d_avg"]  = df["funding_rate"].rolling(7,  min_periods=2).mean()
    df["funding_rate_30d_cum"] = df["funding_rate"].rolling(30, min_periods=5).sum()
    return df


def fetch_open_interest(symbol: str) -> pd.DataFrame:
    data = _get("futures/openInterest/aggregated-history",
                {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        data = _get("futures/openInterest/ohlc-history",
                    {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()

    rows = []
    for d in data:
        t = d.get("t") or d.get("time")
        c = d.get("c") or d.get("close") or d.get("openInterest") or d.get("h")
        if t and c is not None:
            rows.append({"date": _ts(t), "oi_usd": float(c)})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["oi_change_1d"] = df["oi_usd"].pct_change(1)
    df["oi_change_7d"] = df["oi_usd"].pct_change(7)
    return df


def fetch_liquidations(symbol: str) -> pd.DataFrame:
    data = _get("futures/liquidation/aggregated-history",
                {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        data = _get("futures/liquidation/history",
                    {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()

    rows = []
    for d in data:
        t     = d.get("t") or d.get("time")
        long  = float(d.get("longLiquidationUsd")  or d.get("buyUsd")  or d.get("longUsd")  or 0)
        short = float(d.get("shortLiquidationUsd") or d.get("sellUsd") or d.get("shortUsd") or 0)
        if t:
            rows.append({"date": _ts(t), "liq_long_usd": long, "liq_short_usd": short})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["liq_total_usd"] = df["liq_long_usd"] + df["liq_short_usd"]
    total = df["liq_total_usd"].replace(0, float("nan"))
    df["liq_ratio"] = (df["liq_long_usd"] / total).fillna(0.5)
    return df


def fetch_longshort(symbol: str) -> pd.DataFrame:
    results = {}
    for path, col in [
        ("futures/global-long-short-account-ratio/history", "ls_ratio"),
        ("futures/top-long-short-account-ratio/history",    "top_ls_account_ratio"),
        ("futures/top-long-short-position-ratio/history",   "top_ls_position_ratio"),
    ]:
        data = _get(path, {"symbol": symbol, "interval": "1d", "limit": LIMIT})
        time.sleep(DELAY)
        if not data:
            continue
        rows = []
        for d in data:
            t = d.get("t") or d.get("time")
            v = (d.get("longShortRatio") or d.get("longAccount")
                 or d.get("longPosition") or d.get("value"))
            if t and v is not None:
                rows.append({"date": _ts(t), col: float(v)})
        if rows:
            results[col] = pd.DataFrame(rows).sort_values("date").drop_duplicates("date")

    if not results:
        return pd.DataFrame()

    df = list(results.values())[0]
    for frame in list(results.values())[1:]:
        df = df.merge(frame, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)
    if "ls_ratio" in df.columns:
        df["ls_ratio_7d_avg"] = df["ls_ratio"].rolling(7, min_periods=2).mean()
    return df


def fetch_taker_buysell(symbol: str) -> pd.DataFrame:
    data = _get("futures/taker-buy-sell-volume/history",
                {"symbol": symbol, "interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()

    rows = []
    for d in data:
        t    = d.get("t") or d.get("time")
        buy  = float(d.get("buyVolume")  or d.get("buy")  or 0)
        sell = float(d.get("sellVolume") or d.get("sell") or 0)
        total = buy + sell
        if t and total > 0:
            rows.append({"date": _ts(t), "taker_buy_ratio": buy / total})
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["taker_buy_ratio_7d_avg"] = df["taker_buy_ratio"].rolling(7, min_periods=2).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════

def _simple_global(path: str, col: str, val_key: str = "value") -> pd.DataFrame:
    data = _get(path, {"interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        t = d.get("t") or d.get("time")
        v = d.get(val_key) or d.get("value") or d.get("v")
        if t and v is not None:
            rows.append({"date": _ts(t), col: float(v)})
    return pd.DataFrame(rows).drop_duplicates("date") if rows else pd.DataFrame()


def fetch_altcoin_season() -> pd.DataFrame:
    return _simple_global("indicator/altcoin-season-index", "altcoin_season_index")


def fetch_stablecoin_mcap() -> pd.DataFrame:
    data = _get("indicator/stablecoin-market-cap/history", {"interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        t = d.get("t") or d.get("time")
        v = d.get("value") or d.get("marketCap") or d.get("v")
        if t and v:
            rows.append({"date": _ts(t), "stablecoin_mcap": float(v)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["stablecoin_mcap_change_7d"] = df["stablecoin_mcap"].pct_change(7)
    return df


def fetch_btc_nupl() -> pd.DataFrame:
    return _simple_global("indicator/bitcoin-nupl", "btc_nupl")


def fetch_btc_active_addr() -> pd.DataFrame:
    return _simple_global("indicator/bitcoin-active-addresses", "btc_active_addresses")


def fetch_coinbase_premium() -> pd.DataFrame:
    data = _get("indicator/coinbase-premium-index",
                {"symbol": "BTC", "interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        t = d.get("t") or d.get("time")
        v = d.get("coinbasePremium") or d.get("premium") or d.get("value")
        if t and v is not None:
            rows.append({"date": _ts(t), "coinbase_premium": float(v)})
    return pd.DataFrame(rows).drop_duplicates("date") if rows else pd.DataFrame()


def fetch_btc_dominance() -> pd.DataFrame:
    return _simple_global("indicator/bitcoin-dominance", "btc_dominance_cg", "dominance")


def fetch_etf_flows() -> pd.DataFrame:
    data = _get("bitcoin/etf/flow-history", {"interval": "1d", "limit": LIMIT})
    if not data:
        return pd.DataFrame()
    rows = []
    for d in data:
        t    = d.get("t") or d.get("time") or d.get("date")
        flow = d.get("netFlow") or d.get("flow") or d.get("value")
        aum  = d.get("totalNetAssets") or d.get("aum")
        if t and flow is not None:
            date_str = _ts(t) if str(t).isdigit() else str(t)[:10]
            row = {"date": date_str, "etf_net_flow_usd": float(flow)}
            if aum is not None:
                row["etf_total_aum"] = float(aum)
            rows.append(row)
    return pd.DataFrame(rows).drop_duplicates("date") if rows else pd.DataFrame()



ALL_NEW_COLS = [
    "funding_rate", "funding_rate_7d_avg", "funding_rate_30d_cum",
    "oi_usd", "oi_change_1d", "oi_change_7d",
    "liq_long_usd", "liq_short_usd", "liq_total_usd", "liq_ratio",
    "ls_ratio", "ls_ratio_7d_avg", "top_ls_account_ratio", "top_ls_position_ratio",
    "taker_buy_ratio", "taker_buy_ratio_7d_avg",
    "altcoin_season_index", "stablecoin_mcap", "stablecoin_mcap_change_7d",
    "btc_nupl", "btc_active_addresses", "coinbase_premium",
    "btc_dominance_cg", "etf_net_flow_usd", "etf_total_aum",
]


def main():
    print("=" * 60)
    print("  fetch_derivatives.py  —  CoinGlass V4")
    print(f"  Base: {BASE}")
    print("=" * 60)

    print("\n  Testing API connectivity …", end="  ")
    test = _get("futures/openInterest/ohlc-history",
                {"symbol": "BTC", "interval": "1d", "limit": 2})
    if test is None:
        print("✗ FAILED")
        print("  Could not reach API. Check your internet connection and key.")
        return
    print(f"✓  ({len(test) if isinstance(test, list) else 'ok'} rows returned)")

    print(f"\n  Loading {IN_FILE.name} …")
    df = pd.read_csv(IN_FILE, low_memory=False)
    print(f"  {len(df):,} rows  |  {df['asset_id'].nunique()} coins")

    asset_ids = df["asset_id"].unique().tolist()
    fetchable = [(aid, SYMBOL_MAP[aid]) for aid in asset_ids if aid in SYMBOL_MAP]
    no_perp   = [aid for aid in asset_ids if aid not in SYMBOL_MAP]
    print(f"  Coins with perp data : {len(fetchable)}")
    print(f"  Spot-only (no perp)  : {len(no_perp)}")

    df = df.drop(columns=[c for c in ALL_NEW_COLS if c in df.columns])

    # ── STEP 1: Global features ───────────────────────────────────────
    print("\n  ── STEP 1: Global features ──────────────────────────────")
    global_fetchers = [
        ("Altcoin Season Index",     fetch_altcoin_season),
        ("Stablecoin Market Cap",    fetch_stablecoin_mcap),
        ("Bitcoin NUPL",             fetch_btc_nupl),
        ("Bitcoin Active Addresses", fetch_btc_active_addr),
        ("Coinbase Premium Index",   fetch_coinbase_premium),
        ("Bitcoin Dominance",        fetch_btc_dominance),
        ("BTC ETF Flows",            fetch_etf_flows),
    ]

    global_df = None
    for name, fn in global_fetchers:
        print(f"  {name:<30} …", end="  ")
        result = fn()
        time.sleep(DELAY)
        if result.empty:
            print("✗ no data")
            continue
        cols = [c for c in result.columns if c != "date"]
        print(f"✓ {len(result)} days  cols={cols}")
        global_df = result if global_df is None else global_df.merge(result, on="date", how="outer")

    if global_df is not None:
        global_df = global_df.sort_values("date").reset_index(drop=True)
        df = df.merge(global_df, on="date", how="left")
        print(f"  → {len([c for c in global_df.columns if c != 'date'])} global features merged")

    # ── STEP 2: Coin-level features ───────────────────────────────────
    print(f"\n  ── STEP 2: Coin-level ({len(fetchable)} coins) ───────────────")

    all_fr, all_oi, all_liq, all_ls, all_tkr = [], [], [], [], []

    for i, (asset_id, symbol) in enumerate(fetchable):
        print(f"  [{i+1:>3}/{len(fetchable)}] {asset_id:<33}", end="  ")

        fr  = fetch_funding_rate(symbol);  time.sleep(DELAY)
        oi  = fetch_open_interest(symbol); time.sleep(DELAY)
        liq = fetch_liquidations(symbol);  time.sleep(DELAY)
        ls  = fetch_longshort(symbol)      # has internal sleeps
        tkr = fetch_taker_buysell(symbol); time.sleep(DELAY)

        for frames, result in [(all_fr, fr), (all_oi, oi), (all_liq, liq),
                                (all_ls, ls), (all_tkr, tkr)]:
            if not result.empty:
                result.insert(0, "asset_id", asset_id)
                frames.append(result)

        statuses = [
            f"FR:{'✓' if not fr.empty else '—'}",
            f"OI:{'✓' if not oi.empty else '—'}",
            f"LIQ:{'✓' if not liq.empty else '—'}",
            f"L/S:{'✓' if not ls.empty else '—'}",
            f"TKR:{'✓' if not tkr.empty else '—'}",
        ]
        print("  ".join(statuses))

    # ── Merge ─────────────────────────────────────────────────────────
    print("\n  Merging coin-level data …")
    for frames, label in [
        (all_fr,  "Funding rate"),
        (all_oi,  "Open interest"),
        (all_liq, "Liquidations"),
        (all_ls,  "Long/short ratio"),
        (all_tkr, "Taker buy/sell"),
    ]:
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            df = df.merge(combined, on=["asset_id", "date"], how="left")
            print(f"    {label:<25} {len(combined):>8,} rows")

    # ── Save ──────────────────────────────────────────────────────────
    df = df.sort_values(["asset_id", "date"]).reset_index(drop=True)
    df.to_csv(OUT_FILE, index=False)

    added = [c for c in ALL_NEW_COLS if c in df.columns]
    print(f"\n{'='*60}")
    print(f"✅  Saved: {OUT_FILE.name}")
    print(f"   Rows    : {len(df):,}")
    print(f"   Columns : {len(df.columns)}")
    print(f"\n   Coverage (% rows with data):")
    for col in added:
        pct = df[col].notna().mean() * 100
        bar = "█" * int(pct / 5)
        print(f"    {col:<42} {pct:>5.1f}%  {bar}")
    print(f"\n  Next: python3 build_labels_and_crosssectional.py")
    print("="*60)


if __name__ == "__main__":
    main()