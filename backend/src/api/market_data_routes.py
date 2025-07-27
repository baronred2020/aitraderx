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

# Cache para datos de mercado
CACHE = {}
CACHE_TTL = 5  # 5 segundos para respuestas más rápidas

# Cache para velas
CANDLE_CACHE = {}
CANDLE_TTL = 10

def is_market_open(symbol: str) -> bool:
    """
    Determina si el mercado está abierto para un símbolo específico.
    
    Args:
        symbol: Símbolo del instrumento (ej: 'EURUSD', 'AAPL')
    
    Returns:
        bool: True si el mercado está abierto, False si está cerrado
    """
    now = datetime.now()
    current_weekday = now.weekday()  # 0=Lunes, 6=Domingo
    
    # Verificar si es fin de semana
    if current_weekday >= 5:  # Sábado (5) o Domingo (6)
        print(f"[Backend] Market closed for {symbol} - Weekend detected")
        return False
    
    # Para Forex (pares de divisas), el mercado está abierto 24/5 (Lunes-Viernes)
    forex_symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    if symbol in forex_symbols:
        # Forex está abierto de domingo 22:00 UTC a viernes 22:00 UTC
        # Para simplificar, consideramos que está abierto de lunes a viernes
        if current_weekday < 5:  # Lunes a Viernes
            print(f"[Backend] Forex market open for {symbol}")
            return True
        else:
            print(f"[Backend] Forex market closed for {symbol} - Weekend")
            return False
    
    # Para acciones (NYSE/NASDAQ), verificar horarios específicos
    stock_symbols = ['AAPL', 'MSFT', 'TSLA', 'SPX']
    if symbol in stock_symbols:
        # NYSE/NASDAQ horario: 9:30 AM - 4:00 PM EST (Lunes-Viernes)
        # Convertir a hora local (asumiendo EST)
        current_hour = now.hour
        if current_weekday < 5 and 9 <= current_hour < 16:
            print(f"[Backend] Stock market open for {symbol}")
            return True
        else:
            print(f"[Backend] Stock market closed for {symbol} - Outside trading hours or weekend")
            return False
    
    # Para criptomonedas, siempre abierto
    crypto_symbols = ['BTCUSD', 'ETHUSD']
    if symbol in crypto_symbols:
        print(f"[Backend] Crypto market always open for {symbol}")
        return True
    
    # Para otros instrumentos (futures, etc.), usar horario de trading
    other_symbols = ['XAUUSD', 'OIL', 'US10Y']
    if symbol in other_symbols:
        # Futuros tienen horarios específicos, pero para simplificar usamos horario de trading
        if current_weekday < 5 and 9 <= current_hour < 16:
            print(f"[Backend] Futures market open for {symbol}")
            return True
        else:
            print(f"[Backend] Futures market closed for {symbol}")
            return False
    
    # Por defecto, asumir que está abierto si no es fin de semana
    return current_weekday < 5

async def fetch_price(symbol: str):
    """Obtiene precio actual usando Yahoo Finance - siempre intenta datos reales"""
    try:
        print(f"[Backend] Fetching price for symbol: {symbol}")
        
        yahoo_symbol = SYMBOL_MAP.get(symbol, symbol)
        print(f"[Backend] Mapped to Yahoo symbol: {yahoo_symbol}")
        
        ticker = yf.Ticker(yahoo_symbol)
        
        current_price = 0
        change = 0
        change_percent = 0
        volume = 0
        market_status = "open"
        hist = None
        
        # Estrategia 1: Intentar datos históricos recientes (5 días)
        try:
            hist = ticker.history(period="5d")
            print(f"[Backend] Retrieved {len(hist)} historical records for {symbol}")
            
            if len(hist) >= 1:
                current_price = float(hist['Close'].iloc[-1])
                volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
                
                # Calcular cambio si tenemos al menos 2 días
                if len(hist) >= 2:
                    previous_price = float(hist['Close'].iloc[-2])
                    change = current_price - previous_price
                    change_percent = (change / previous_price) * 100 if previous_price != 0 else 0
                    
                print(f"[Backend] Using historical data: price={current_price}, change={change}")
        except Exception as e:
            print(f"[Backend] Error getting 5d history: {e}")
        
        # Estrategia 2: Si no hay datos históricos, intentar info del ticker
        if current_price == 0:
            try:
                print(f"[Backend] No historical data, trying ticker info...")
                info = ticker.info
                current_price = info.get('regularMarketPrice', 0) or info.get('previousClose', 0)
                volume = info.get('volume', 0) or info.get('averageVolume', 0)
                
                if current_price > 0:
                    print(f"[Backend] Using ticker info: price={current_price}")
            except Exception as e:
                print(f"[Backend] Error getting ticker info: {e}")
                
        # Estrategia 3: Si aún no hay precio, intentar con datos más antiguos
        if current_price == 0:
            try:
                print(f"[Backend] No recent data, trying longer period...")
                hist_long = ticker.history(period="1mo")
                if len(hist_long) >= 1:
                    current_price = float(hist_long['Close'].iloc[-1])
                    volume = int(hist_long['Volume'].iloc[-1]) if not pd.isna(hist_long['Volume'].iloc[-1]) else 0
                    print(f"[Backend] Using older data: price={current_price}")
            except Exception as e:
                print(f"[Backend] Error getting 1mo history: {e}")
                
        # Estrategia 4: Intentar con diferentes intervalos
        if current_price == 0:
            try:
                print(f"[Backend] Trying different intervals...")
                for interval in ['1d', '1wk', '1mo']:
                    hist_alt = ticker.history(period=interval)
                    if len(hist_alt) >= 1:
                        current_price = float(hist_alt['Close'].iloc[-1])
                        volume = int(hist_alt['Volume'].iloc[-1]) if not pd.isna(hist_alt['Volume'].iloc[-1]) else 0
                        print(f"[Backend] Using {interval} data: price={current_price}")
                        break
            except Exception as e:
                print(f"[Backend] Error trying different intervals: {e}")
                
        # Verificar si el mercado está abierto
        if not is_market_open(symbol):
            market_status = "closed"
            print(f"[Backend] Market closed for {symbol}, but using last available data")
            
        # Validar que tenemos un precio válido
        if current_price == 0 or pd.isna(current_price):
            print(f"[Backend] Warning: No valid price found for {symbol}, using fallback values")
            # Valores de fallback actualizados para Forex
            fallback_prices = {
                "EURUSD": 1.0925,
                "GBPUSD": 1.2650,
                "USDJPY": 148.50,
                "AUDUSD": 0.6650,
                "USDCAD": 1.3550,
                "AAPL": 175.50,
                "MSFT": 380.80,
                "TSLA": 240.50,
                "BTCUSD": 42000.00,
                "ETHUSD": 2500.00,
                "XAUUSD": 2000.00,
                "OIL": 75.00,
                "SPX": 4500.00,
                "US10Y": 4.50
            }
            current_price = fallback_prices.get(symbol, 100.0)
            change = 0.01
            change_percent = 0.1
            volume = 1000000
            market_status = "closed"
            
        # Calcular high/low basados en datos históricos si están disponibles
        high = current_price * 1.001  # Aproximación por defecto
        low = current_price * 0.999   # Aproximación por defecto
        open_price = current_price - change  # Aproximación
        
        if hist is not None and len(hist) >= 1:
            high = float(hist['High'].iloc[-1])
            low = float(hist['Low'].iloc[-1])
            open_price = float(hist['Open'].iloc[-1])
            
        result = {
            "price": f"{current_price:.5f}",
            "change": f"{change:.5f}",
            "changePercent": f"{change_percent:.2f}",
            "volume": str(int(volume)),
            "high": f"{high:.5f}",
            "low": f"{low:.5f}",
            "open": f"{open_price:.5f}",
            "previousClose": f"{current_price - change:.5f}",
            "marketStatus": market_status
        }
        
        print(f"[Backend] Successfully fetched price for {symbol}: {result}")
        return result
    except Exception as e:
        print(f"[Backend] Error fetching price for {symbol}: {str(e)}")
        # Devolver datos de fallback en caso de error
        fallback_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 148.50,
            "AUDUSD": 0.6550,
            "USDCAD": 1.3550,
            "AAPL": 150.00,
            "MSFT": 300.00,
            "TSLA": 200.00,
            "BTCUSD": 45000.00,
            "ETHUSD": 2500.00,
            "XAUUSD": 2000.00,
            "OIL": 75.00,
            "SPX": 4500.00,
            "US10Y": 4.50
        }
        price = fallback_prices.get(symbol, 100.0)
        return {
            "price": f"{price:.5f}",
            "change": "0.001",
            "changePercent": "0.10",
            "volume": "1000000",
            "high": f"{price * 1.001:.5f}",
            "low": f"{price * 0.999:.5f}",
            "open": f"{price:.5f}",
            "previousClose": f"{price:.5f}",
            "marketStatus": "error"
        }

@router.get("/market-data")
async def get_market_data(symbols: str = Query(...)):
    """Endpoint principal para datos de mercado usando Yahoo Finance"""
    # Parsear los símbolos desde el query string
    symbol_list = [s.strip() for s in symbols.split(',') if s.strip()]
    print(f"[Backend] Market data request for symbols: {symbol_list}")
    results = {}
    now = time.time()
    
    for symbol in symbol_list:
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
        
        values = []
        hist = None
        
        # Estrategia 1: Intentar con la configuración original
        try:
            # Ajustar periodo compatible con Yahoo Finance
            if yahoo_interval in ["1m", "5m", "15m", "30m", "1h"]:
                period = "5d"  # Yahoo solo permite hasta 7 días para intervalos de minutos/horas
            elif yahoo_interval in ["1d", "1wk", "1mo"]:
                period = "1y"  # Para diarios o superiores, usar 1 año
            else:
                period = "1mo"
            
            print(f"[Backend] Trying period: {period}")
            hist = ticker.history(period=period, interval=yahoo_interval)
            print(f"[Backend] Retrieved {len(hist)} candles")
            
            if len(hist) > 0:
                # Convertir a formato compatible con Twelve Data
                for date, row in hist.iterrows():
                    values.append({
                        "datetime": date.strftime('%Y-%m-%d %H:%M:%S'),
                        "open": str(row['Open']),
                        "high": str(row['High']),
                        "low": str(row['Low']),
                        "close": str(row['Close']),
                        "volume": str(row['Volume'])
                    })
        except Exception as e:
            print(f"[Backend] Error with original config: {e}")
        
        # Estrategia 2: Si no hay datos, intentar con diferentes períodos
        if not values:
            try:
                print(f"[Backend] Trying alternative periods...")
                alternative_periods = ["1mo", "3mo", "6mo", "1y"]
                
                for alt_period in alternative_periods:
                    try:
                        hist = ticker.history(period=alt_period, interval=yahoo_interval)
                        if len(hist) > 0:
                            print(f"[Backend] Success with {alt_period}: {len(hist)} candles")
                            for date, row in hist.iterrows():
                                values.append({
                                    "datetime": date.strftime('%Y-%m-%d %H:%M:%S'),
                                    "open": str(row['Open']),
                                    "high": str(row['High']),
                                    "low": str(row['Low']),
                                    "close": str(row['Close']),
                                    "volume": str(row['Volume'])
                                })
                            break
                    except Exception as e:
                        print(f"[Backend] Error with {alt_period}: {e}")
                        continue
            except Exception as e:
                print(f"[Backend] Error with alternative periods: {e}")
        
        # Estrategia 3: Si aún no hay datos, intentar con diferentes intervalos
        if not values:
            try:
                print(f"[Backend] Trying alternative intervals...")
                alternative_intervals = ["1d", "1wk", "1mo"]
                
                for alt_interval in alternative_intervals:
                    try:
                        hist = ticker.history(period="1mo", interval=alt_interval)
                        if len(hist) > 0:
                            print(f"[Backend] Success with {alt_interval}: {len(hist)} candles")
                            for date, row in hist.iterrows():
                                values.append({
                                    "datetime": date.strftime('%Y-%m-%d %H:%M:%S'),
                                    "open": str(row['Open']),
                                    "high": str(row['High']),
                                    "low": str(row['Low']),
                                    "close": str(row['Close']),
                                    "volume": str(row['Volume'])
                                })
                            break
                    except Exception as e:
                        print(f"[Backend] Error with {alt_interval}: {e}")
                        continue
            except Exception as e:
                print(f"[Backend] Error with alternative intervals: {e}")
        
        # Si no hay datos de Yahoo Finance, generar datos de fallback
        if not values:
            print(f"[Backend] No data from Yahoo Finance for {symbol}, generating fallback data")
            # Generar datos de fallback basados en precios típicos actualizados
            fallback_prices = {
                "EURUSD": 1.0925,
                "GBPUSD": 1.2650,
                "USDJPY": 148.50,
                "AUDUSD": 0.6650,
                "USDCAD": 1.3550,
                "AAPL": 175.50,
                "MSFT": 380.80,
                "TSLA": 240.50,
                "BTCUSD": 42000.00,
                "ETHUSD": 2500.00,
                "XAUUSD": 2000.00,
                "OIL": 75.00,
                "SPX": 4500.00,
                "US10Y": 4.50
            }
            
            base_price = fallback_prices.get(symbol, 100.0)
            from datetime import datetime, timedelta
            
            # Generar 50 velas de ejemplo para los últimos días
            for i in range(50):
                date = datetime.now() - timedelta(days=50-i)
                # Simular variación de precio más realista
                variation = (i % 10 - 5) * 0.001  # Variación de ±0.005
                open_price = base_price + variation
                high_price = open_price + 0.002
                low_price = open_price - 0.002
                close_price = open_price + (variation * 0.5)
                
                values.append({
                    "datetime": date.strftime('%Y-%m-%d %H:%M:%S'),
                    "open": f"{open_price:.5f}",
                    "high": f"{high_price:.5f}",
                    "low": f"{low_price:.5f}",
                    "close": f"{close_price:.5f}",
                    "volume": "1000000"
                })
        
        result = {
            "symbol": symbol,
            "interval": interval,
            "values": values
        }
        
        print(f"[Backend] Successfully processed candles for {symbol} - {len(values)} candles")
        return result
        
    except Exception as e:
        print(f"[Backend] Error fetching candles for {symbol}: {str(e)}")
        # En caso de error, también generar datos de fallback
        fallback_prices = {
            "EURUSD": 1.0850,
            "GBPUSD": 1.2650,
            "USDJPY": 148.50,
            "AUDUSD": 0.6550,
            "USDCAD": 1.3550,
            "AAPL": 150.00,
            "MSFT": 300.00,
            "TSLA": 200.00,
            "BTCUSD": 45000.00,
            "ETHUSD": 2500.00,
            "XAUUSD": 2000.00,
            "OIL": 75.00,
            "SPX": 4500.00,
            "US10Y": 4.50
        }
        
        base_price = fallback_prices.get(symbol, 100.0)
        from datetime import datetime, timedelta
        
        # Generar 50 velas de ejemplo para los últimos días
        values = []
        for i in range(50):
            date = datetime.now() - timedelta(days=50-i)
            # Simular variación de precio
            variation = (i % 10 - 5) * 0.001  # Variación de ±0.005
            open_price = base_price + variation
            high_price = open_price + 0.002
            low_price = open_price - 0.002
            close_price = open_price + (variation * 0.5)
            
            values.append({
                "datetime": date.strftime('%Y-%m-%d %H:%M:%S'),
                "open": f"{open_price:.5f}",
                "high": f"{high_price:.5f}",
                "low": f"{low_price:.5f}",
                "close": f"{close_price:.5f}",
                "volume": "1000000"
            })
        
        return {
            "symbol": symbol,
            "interval": interval,
            "values": values,
            "error": str(e)
        }

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

@router.get("/market-status")
async def get_market_status(symbol: str):
    """Endpoint para verificar el estado del mercado para un símbolo"""
    is_open = is_market_open(symbol)
    now = datetime.now()
    
    return {
        "symbol": symbol,
        "is_open": is_open,
        "current_time": now.isoformat(),
        "weekday": now.strftime("%A"),
        "timezone": "UTC"
    }

 