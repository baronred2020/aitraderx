from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import logging

# Importar rutas
from api.brain_trader_routes import router as brain_trader_router
from api.mega_mind_routes import router as mega_mind_router

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="AI Trader X - Brain Trader API",
    description="API para el sistema de trading con IA Brain Trader",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar dominios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(brain_trader_router, prefix="/api/v1")
app.include_router(mega_mind_router, prefix="/api/v1")

@app.get("/")
async def root():
    """
    Endpoint raíz de la API
    """
    return {
        "message": "AI Trader X - Brain Trader API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now(),
        "services": {
            "brain_trader": "active",
            "mega_mind": "active"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check general de la API
    """
    return {
        "status": "healthy",
        "service": "AI Trader X API",
        "version": "1.0.0",
        "timestamp": datetime.now(),
        "components": {
            "brain_trader_api": "active",
            "mega_mind_api": "active",
            "database": "active",
            "models": "active"
        }
    }

@app.get("/api/v1/status")
async def api_status():
    """
    Estado detallado de la API
    """
    return {
        "api_version": "v1",
        "status": "active",
        "endpoints": {
            "brain_trader": {
                "base_url": "/api/v1/brain-trader",
                "endpoints": [
                    "/predictions/{brain_type}",
                    "/signals/{brain_type}",
                    "/trends/{brain_type}",
                    "/model-info/{brain_type}",
                    "/available-brains",
                    "/health"
                ]
            },
            "mega_mind": {
                "base_url": "/api/v1/mega-mind",
                "endpoints": [
                    "/predictions",
                    "/collaboration",
                    "/arena",
                    "/evolution",
                    "/orchestration",
                    "/performance",
                    "/health",
                    "/config"
                ]
            }
        },
        "available_brain_types": [
            "brain_max",
            "brain_ultra", 
            "brain_predictor",
            "mega_mind"
        ],
        "available_pairs": [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "EURGBP", "GBPJPY", "EURJPY"
        ],
        "available_styles": [
            "scalping", "day_trading", "swing_trading", "position_trading"
        ],
        "timestamp": datetime.now()
    }

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """
    Manejador para rutas no encontradas
    """
    return JSONResponse(
        status_code=404,
        content={
            "error": "Endpoint not found",
            "message": "The requested endpoint does not exist",
            "available_endpoints": {
                "brain_trader": "/api/v1/brain-trader",
                "mega_mind": "/api/v1/mega-mind",
                "docs": "/docs",
                "health": "/health"
            },
            "timestamp": datetime.now()
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """
    Manejador para errores internos
    """
    logger.error(f"Internal server error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now()
        }
    )

@app.on_event("startup")
async def startup_event():
    """
    Evento de inicio de la aplicación
    """
    logger.info("Starting AI Trader X - Brain Trader API")
    logger.info("Loading models and initializing services...")
    
    # Aquí se pueden inicializar servicios, cargar modelos, etc.
    # Por ejemplo:
    # - Cargar modelos de IA
    # - Inicializar conexiones a base de datos
    # - Configurar servicios de trading
    
    logger.info("API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """
    Evento de cierre de la aplicación
    """
    logger.info("Shutting down AI Trader X - Brain Trader API")
    
    # Aquí se pueden realizar tareas de limpieza
    # Por ejemplo:
    # - Guardar estado de modelos
    # - Cerrar conexiones a base de datos
    # - Limpiar recursos
    
    logger.info("API shutdown completed")

if __name__ == "__main__":
    # Configuración del servidor
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "reload": True,  # Auto-reload en desarrollo
        "log_level": "info"
    }
    
    logger.info(f"Starting server on {config['host']}:{config['port']}")
    uvicorn.run("main:app", **config) 