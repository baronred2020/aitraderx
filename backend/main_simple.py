from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from datetime import datetime

app = FastAPI(
    title="AI Trader X - Brain Trader API",
    description="API para el sistema de trading con IA Brain Trader",
    version="1.0.0"
)

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
    return {
        "message": "AI Trader X - Brain Trader API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now()
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Trader X API",
        "version": "1.0.0",
        "timestamp": datetime.now()
    }

@app.get("/test")
async def test_endpoint():
    return {
        "message": "Test endpoint working!",
        "timestamp": datetime.now()
    }

if __name__ == "__main__":
    print("🚀 Iniciando servidor simple en puerto 8080...")
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False) 