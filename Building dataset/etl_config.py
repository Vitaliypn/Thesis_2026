"""
etl_config.py
─────────────
Central configuration for the crypto megadataset ETL pipeline.
Edit the values here to change behaviour across all scripts.
"""

CCXT_EXCHANGES = [
    "binance",
    "kraken",
    "bybit",
    "kucoin",
    "okx",
]

DEFAULT_DAYS   = 365 * 7
MAX_DAYS       = 365 * 8

OHLCV_TIMEFRAME = "1d"
CANDLES_PER_REQUEST = 1000


ALLOWED_QUOTES = {"USDT"}

SPOT_ONLY = True
DEFILLAMA_BASE = "https://api.llama.fi"
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
TOP_N_ASSETS   = 200   # how many top-by-mcap coins to include
import datetime as _dt
RUN_FOLDER_NAME = "run_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")