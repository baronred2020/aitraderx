#!/usr/bin/env python3
"""
Script de prueba simple para verificar que el servidor funcione
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Test Server")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Test server is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "Server is running"}

@app.get("/test-price")
async def test_price():
    """Prueba simple para obtener precio de EURUSD"""
    try:
        import yfinance as yf
        ticker = yf.Ticker("EURUSD=X")
        price = ticker.info.get('regularMarketPrice')
        if price:
            return {"pair": "EURUSD", "price": price, "status": "success"}
        else:
            # Intentar obtener precio histórico
            hist = ticker.history(period="1d")
            if not hist.empty:
                price = hist['Close'].iloc[-1]
                return {"pair": "EURUSD", "price": price, "status": "historical"}
            else:
                return {"pair": "EURUSD", "price": 1.0925, "status": "fallback"}
    except Exception as e:
        return {"pair": "EURUSD", "price": 1.0925, "status": "error", "error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002) 