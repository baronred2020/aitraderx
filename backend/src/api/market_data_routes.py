from fastapi import APIRouter, Query
import httpx
import time

router = APIRouter(prefix="/api")

TWELVE_DATA_API_KEY = "701dbaae483c49ad8feaa37f72d290c6"

# Mapeo de símbolos amigables a Twelve Data
SYMBOL_MAP = {
    "EURUSD": ("EUR/USD", "forex"),
    "USDJPY": ("USD/JPY", "forex"),
    "GBPUSD": ("GBP/USD", "forex"),
    "AAPL": ("AAPL", "stock"),
    "MSFT": ("MSFT", "stock"),
    "TSLA": ("TSLA", "stock"),
    "BTCUSD": ("BTC/USD", "crypto"),
    "ETHUSD": ("ETH/USD", "crypto"),
    "XAUUSD": ("XAU/USD", "forex"),
    "OIL": ("WTICO/USD", "forex"),
    "SPX": ("SPX", "index"),
    "US10Y": ("US10Y", "bond")
}

CACHE = {}
CACHE_TTL = 15  # segundos

async def fetch_price(symbol: str):
    url = f"https://api.twelvedata.com/price?symbol={symbol}&apikey={TWELVE_DATA_API_KEY}"
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

@router.get("/market-data")
async def get_market_data(symbols: list[str] = Query(...)):
    results = {}
    now = time.time()
    for user_symbol in symbols:
        mapped = SYMBOL_MAP.get(user_symbol)
        if not mapped:
            results[user_symbol] = {"error": "Símbolo no soportado"}
            continue
        td_symbol, _ = mapped
        cache_entry = CACHE.get(user_symbol)
        if cache_entry and now - cache_entry[0] < CACHE_TTL:
            results[user_symbol] = cache_entry[1]
            continue
        try:
            data = await fetch_price(td_symbol)
            results[user_symbol] = data
            CACHE[user_symbol] = (now, data)
        except Exception as e:
            results[user_symbol] = {"error": str(e)}
    return results

CANDLE_CACHE = {}
CANDLE_TTL = 30  # segundos

async def fetch_candles(symbol: str, interval: str, outputsize: int):
    url = (
        f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}&outputsize={outputsize}&apikey={TWELVE_DATA_API_KEY}"
    )
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()

# Mapeo de resoluciones a Twelve Data
RESOLUTION_MAP = {
    "1": "1min",
    "5": "5min",
    "15": "15min",
    "30": "30min",
    "60": "1h",
    "D": "1day",
    "W": "1week",
    "M": "1month"
}

@router.get("/candles")
async def get_market_candles(
    symbol: str,
    interval: str = "15",
    count: int = 100
):
    mapped = SYMBOL_MAP.get(symbol)
    if not mapped:
        return {"error": "Símbolo no soportado"}
    td_symbol, _ = mapped
    timeInterval = RESOLUTION_MAP.get(interval, "15min")
    cache_key = f"{symbol}_{timeInterval}_{count}"
    now = int(time.time())
    cache_entry = CANDLE_CACHE.get(cache_key)
    if cache_entry and now - cache_entry[0] < CANDLE_TTL:
        return cache_entry[1]
    try:
        data = await fetch_candles(td_symbol, timeInterval, count)
        CANDLE_CACHE[cache_key] = (now, data)
        return data
    except Exception as e:
        return {"error": str(e)} 