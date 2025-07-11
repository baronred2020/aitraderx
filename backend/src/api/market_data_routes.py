from fastapi import APIRouter, Query
import httpx
import time
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd

router = APIRouter(prefix="/api")

# Mapeo de símbolos amigables a Yahoo Finance
SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "USDJPY": "USDJPY=X", 
    "GBPUSD": "GBPUSD=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "TSLA": "TSLA",
    "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",
    "XAUUSD": "GC=F",  # Gold futures
    "OIL": "CL=F",     # Crude oil futures
    "SPX": "^GSPC",    # S&P 500
    "US10Y": "^TNX"    # 10-year Treasury
}



CACHE = {}
CACHE_TTL = 15  # segundos



# Cache para velas de Twelve Data
CANDLE_CACHE = {}
CANDLE_TTL = 30  # 30 segundos para velas de Twelve Data

async def fetch_price(symbol: str):
    """Obtiene precio actual usando Yahoo Finance"""
    try:
        print(f"[Backend] Fetching price for symbol: {symbol}")
        yahoo_symbol = SYMBOL_MAP.get(symbol, symbol)
        print(f"[Backend] Mapped to Yahoo symbol: {yahoo_symbol}")
        
        ticker = yf.Ticker(yahoo_symbol)
        
        # Obtener datos históricos recientes para calcular cambio
        hist = ticker.history(period="2d")
        if len(hist) >= 2:
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[-2]
            change = current_price - previous_price
            change_percent = (change / previous_price) * 100
        else:
            current_price = ticker.info.get('regularMarketPrice', 0)
            change = 0
            change_percent = 0
            
        result = {
            "price": str(current_price),
            "change": str(change),
            "changePercent": str(change_percent),
            "volume": str(ticker.info.get('volume', 0)),
            "high": str(ticker.info.get('dayHigh', current_price)),
            "low": str(ticker.info.get('dayLow', current_price)),
            "open": str(ticker.info.get('regularMarketOpen', current_price)),
            "previousClose": str(ticker.info.get('regularMarketPreviousClose', current_price))
        }
        
        print(f"[Backend] Successfully fetched price for {symbol}: {result}")
        return result
    except Exception as e:
        print(f"[Backend] Error fetching price for {symbol}: {str(e)}")
        return {"error": str(e)}





@router.get("/market-data")
async def get_market_data(symbols: list[str] = Query(...)):
    """Endpoint principal para datos de mercado usando Yahoo Finance"""
    print(f"[Backend] Market data request for symbols: {symbols}")
    results = {}
    now = time.time()
    
    for symbol in symbols:
        cache_entry = CACHE.get(symbol)
        if cache_entry and now - cache_entry[0] < CACHE_TTL:
            print(f"[Backend] Using cached data for {symbol}")
            results[symbol] = cache_entry[1]
            continue
            
        print(f"[Backend] Fetching fresh data for {symbol}")
        try:
            data = await fetch_price(symbol)
            results[symbol] = data
            CACHE[symbol] = (now, data)
            print(f"[Backend] Successfully cached data for {symbol}")
        except Exception as e:
            print(f"[Backend] Error for {symbol}: {str(e)}")
            results[symbol] = {"error": str(e)}
    
    print(f"[Backend] Returning results: {results}")
    return results



async def fetch_candles(symbol: str, interval: str, outputsize: int):
    """Obtiene datos de velas usando Yahoo Finance"""
    try:
        print(f"[Backend] Fetching candles for symbol: {symbol}, interval: {interval}, count: {outputsize}")
        yahoo_symbol = SYMBOL_MAP.get(symbol, symbol)
        print(f"[Backend] Mapped to Yahoo symbol: {yahoo_symbol}")
        
        ticker = yf.Ticker(yahoo_symbol)
        
        # Mapear intervalos de Yahoo Finance
        interval_map = {
            "1min": "1m",
            "5min": "5m", 
            "15min": "15m",
            "30min": "30m",
            "1h": "1h",
            "1day": "1d",
            "1week": "1wk",
            "1month": "1mo"
        }
        
        yahoo_interval = interval_map.get(interval, "15m")
        print(f"[Backend] Using Yahoo interval: {yahoo_interval}")
        
        # Ajustar periodo compatible con Yahoo Finance
        if yahoo_interval in ["1m", "5m", "15m", "30m", "1h"]:
            period = "5d"  # Yahoo solo permite hasta 7 días para intervalos de minutos/horas
        elif yahoo_interval in ["1d", "1wk", "1mo"]:
            period = "1y"  # Para diarios o superiores, usar 1 año
        else:
            period = "1mo"
        
        print(f"[Backend] Using period: {period}")
        hist = ticker.history(period=period, interval=yahoo_interval)
        print(f"[Backend] Retrieved {len(hist)} candles")
        
        # Convertir a formato compatible con Twelve Data
        values = []
        for date, row in hist.iterrows():
            values.append({
                "datetime": date.strftime('%Y-%m-%d %H:%M:%S'),
                "open": str(row['Open']),
                "high": str(row['High']),
                "low": str(row['Low']),
                "close": str(row['Close']),
                "volume": str(row['Volume'])
            })
            
        result = {
            "values": values,
            "meta": {
                "symbol": symbol,
                "interval": interval,
                "currency_base": "USD",
                "currency_quote": "USD"
            }
        }
        
        print(f"[Backend] Successfully fetched candles for {symbol}: {len(values)} candles")
        return result
    except Exception as e:
        print(f"[Backend] Error fetching candles for {symbol}: {str(e)}")
        return {"error": str(e)}



@router.get("/candles")
async def get_market_candles(
    symbol: str,
    interval: str = "15",
    count: int = 100
):
    """Endpoint principal para velas usando Yahoo Finance"""
    print(f"[Backend] Candles request for symbol: {symbol}, interval: {interval}, count: {count}")
    
    # Mapear intervalos de Twelve Data a Yahoo Finance
    interval_map = {
        "1": "1min",
        "5": "5min",
        "15": "15min",
        "30": "30min",
        "60": "1h",
        "D": "1day",
        "W": "1week",
        "M": "1month"
    }
    
    timeInterval = interval_map.get(interval, "15min")
    cache_key = f"{symbol}_{timeInterval}_{count}"
    now = int(time.time())
    cache_entry = CANDLE_CACHE.get(cache_key)
    
    if cache_entry and now - cache_entry[0] < CANDLE_TTL:
        print(f"[Backend] Using cached candles for {symbol}")
        return cache_entry[1]
        
    print(f"[Backend] Fetching fresh candles for {symbol}")
    try:
        data = await fetch_candles(symbol, timeInterval, count)
        CANDLE_CACHE[cache_key] = (now, data)
        print(f"[Backend] Successfully cached candles for {symbol}")
        return data
    except Exception as e:
        print(f"[Backend] Error for candles {symbol}: {str(e)}")
        return {"error": str(e)}

 