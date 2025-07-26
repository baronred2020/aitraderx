from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import random
from typing import List, Dict, Any
from pydantic import BaseModel

# Modelos Pydantic para las respuestas
class PredictionResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    target_price: float
    timeframe: str
    reasoning: str
    brain_type: str
    timestamp: str

class SignalResponse(BaseModel):
    pair: str
    type: str
    strength: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    brain_type: str
    timestamp: str

class TrendResponse(BaseModel):
    pair: str
    direction: str
    strength: float
    timeframe: str
    support: float
    resistance: float
    description: str
    brain_type: str
    timestamp: str

class MegaMindPredictionResponse(BaseModel):
    pair: str
    direction: str
    confidence: float
    target_price: float
    timeframe: str
    reasoning: str
    brain_type: str
    fusion_method: str
    collaboration_score: float
    fusion_details: dict
    timestamp: str

# Crear aplicación FastAPI
app = FastAPI(
    title="AI Trader X - Brain Trader API",
    description="API completa para el sistema de trading con IA Brain Trader",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== ENDPOINTS BÁSICOS =====
@app.get("/")
async def root():
    return {
        "message": "AI Trader X - Brain Trader API",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "brain_trader": "active",
            "mega_mind": "active"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Trader X API",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

# ===== BRAIN TRADER ENDPOINTS =====
@app.get("/api/v1/brain-trader/available-brains")
async def get_available_brains():
    return {
        "available_brains": [
            "brain_max",
            "brain_ultra", 
            "brain_predictor",
            "mega_mind"
        ],
        "default_brain": "brain_max"
    }

@app.get("/api/v1/brain-trader/predictions/{brain_type}")
async def get_predictions(brain_type: str, pair: str = "EURUSD", style: str = "day_trading", limit: int = 5):
    valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
    if brain_type not in valid_brain_types:
        raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
    
    predictions = []
    base_price = 1.0925 if pair == 'EURUSD' else 1.2500
    
    for i in range(min(limit, 5)):
        direction = random.choice(['up', 'down', 'sideways'])
        confidence = random.uniform(70, 95)
        if brain_type == 'mega_mind':
            confidence = random.uniform(90, 98)
        
        target_price = base_price + (random.uniform(-0.01, 0.01))
        
        prediction = PredictionResponse(
            pair=pair,
            direction=direction,
            confidence=confidence,
            target_price=target_price,
            timeframe='1H',
            reasoning=f'Análisis técnico basado en {brain_type} - {direction.upper()}',
            brain_type=brain_type,
            timestamp=datetime.now().isoformat()
        )
        predictions.append(prediction)
    
    return predictions

@app.get("/api/v1/brain-trader/signals/{brain_type}")
async def get_signals(brain_type: str, pair: str = "EURUSD", limit: int = 5):
    valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
    if brain_type not in valid_brain_types:
        raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
    
    signals = []
    base_price = 1.0925 if pair == 'EURUSD' else 1.2500
    
    for i in range(min(limit, 5)):
        signal_type = random.choice(['buy', 'sell', 'hold'])
        strength = random.choice(['strong', 'medium', 'weak'])
        confidence = random.uniform(60, 90)
        entry_price = base_price + (random.uniform(-0.005, 0.005))
        
        signal = SignalResponse(
            pair=pair,
            type=signal_type,
            strength=strength,
            confidence=confidence,
            entry_price=entry_price,
            stop_loss=entry_price - 0.005,
            take_profit=entry_price + 0.015,
            brain_type=brain_type,
            timestamp=datetime.now().isoformat()
        )
        signals.append(signal)
    
    return signals

@app.get("/api/v1/brain-trader/trends/{brain_type}")
async def get_trends(brain_type: str, pair: str = "EURUSD", limit: int = 3):
    valid_brain_types = ['brain_max', 'brain_ultra', 'brain_predictor', 'mega_mind']
    if brain_type not in valid_brain_types:
        raise HTTPException(status_code=400, detail=f"Brain type must be one of: {valid_brain_types}")
    
    trends = []
    base_price = 1.0925 if pair == 'EURUSD' else 1.2500
    
    for i in range(min(limit, 3)):
        direction = random.choice(['bullish', 'bearish', 'neutral'])
        strength = random.uniform(50, 100)
        support = base_price - 0.01
        resistance = base_price + 0.01
        
        trend = TrendResponse(
            pair=pair,
            direction=direction,
            strength=strength,
            timeframe='4H',
            support=support,
            resistance=resistance,
            description=f'Tendencia {direction} con soporte en {support:.4f}',
            brain_type=brain_type,
            timestamp=datetime.now().isoformat()
        )
        trends.append(trend)
    
    return trends

# ===== MEGA MIND ENDPOINTS =====
@app.get("/api/v1/mega-mind/predictions")
async def get_mega_mind_predictions(pair: str = "EURUSD", style: str = "day_trading", limit: int = 5):
    predictions = []
    base_price = 1.0925 if pair == 'EURUSD' else 1.2500
    
    for i in range(min(limit, 5)):
        direction = random.choice(['up', 'down', 'sideways'])
        confidence = random.uniform(90, 98)  # MEGA MIND tiene mayor precisión
        target_price = base_price + (random.uniform(-0.01, 0.01))
        
        # Simular detalles de fusión
        fusion_details = {
            'brain_max_confidence': random.uniform(75, 88),
            'brain_ultra_confidence': random.uniform(80, 92),
            'brain_predictor_confidence': random.uniform(85, 94),
            'consensus_level': random.uniform(0.6, 1.0),
            'collaboration_boost': 1.2
        }
        
        prediction = MegaMindPredictionResponse(
            pair=pair,
            direction=direction,
            confidence=confidence,
            target_price=target_price,
            timeframe='Multi-TF',
            reasoning=f'MEGA MIND fusion: {direction.upper()} consensus',
            brain_type='mega_mind',
            fusion_method='weighted_consensus',
            collaboration_score=random.uniform(0.85, 0.98),
            fusion_details=fusion_details,
            timestamp=datetime.now().isoformat()
        )
        predictions.append(prediction)
    
    return predictions

@app.get("/api/v1/mega-mind/collaboration")
async def get_brain_collaboration(pair: str = "EURUSD"):
    return {
        "pair": pair,
        "collaboration_score": random.uniform(0.85, 0.98),
        "consensus_level": random.uniform(0.75, 0.95),
        "brain_synergy": {
            "brain_max_contribution": random.uniform(0.20, 0.30),
            "brain_ultra_contribution": random.uniform(0.30, 0.40),
            "brain_predictor_contribution": random.uniform(0.35, 0.45)
        },
        "conflict_resolution": {
            "resolved_conflicts": random.randint(5, 15),
            "consensus_achieved": random.uniform(0.80, 0.95),
            "decision_confidence": random.uniform(0.90, 0.98)
        },
        "performance_metrics": {
            "accuracy_improvement": random.uniform(0.05, 0.15),
            "risk_reduction": random.uniform(0.10, 0.20),
            "prediction_stability": random.uniform(0.85, 0.95)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/mega-mind/arena")
async def get_brain_arena_results(pair: str = "EURUSD"):
    return {
        "pair": pair,
        "competition_round": random.randint(1, 10),
        "arena_results": {
            "brain_max": {
                "wins": random.randint(15, 25),
                "losses": random.randint(5, 15),
                "win_rate": random.uniform(0.65, 0.85),
                "performance_score": random.uniform(0.75, 0.88)
            },
            "brain_ultra": {
                "wins": random.randint(20, 30),
                "losses": random.randint(5, 15),
                "win_rate": random.uniform(0.75, 0.90),
                "performance_score": random.uniform(0.80, 0.92)
            },
            "brain_predictor": {
                "wins": random.randint(25, 35),
                "losses": random.randint(3, 12),
                "win_rate": random.uniform(0.80, 0.94),
                "performance_score": random.uniform(0.85, 0.94)
            }
        },
        "champion": "brain_predictor",
        "overall_performance": random.uniform(0.85, 0.95),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/mega-mind/performance")
async def get_mega_mind_performance():
    return {
        "overall_accuracy": random.uniform(92, 98),
        "prediction_success_rate": random.uniform(0.85, 0.95),
        "risk_adjusted_returns": random.uniform(0.12, 0.25),
        "sharpe_ratio": random.uniform(1.5, 2.5),
        "max_drawdown": random.uniform(0.05, 0.15),
        "win_rate": random.uniform(0.75, 0.90),
        "profit_factor": random.uniform(1.8, 3.2),
        "average_trade_duration": random.uniform(2, 8),
        "consecutive_wins": random.randint(5, 15),
        "consecutive_losses": random.randint(1, 3),
        "volatility": random.uniform(0.08, 0.18),
        "calmar_ratio": random.uniform(2.0, 4.0),
        "sortino_ratio": random.uniform(2.5, 4.5),
        "information_ratio": random.uniform(1.8, 3.0),
        "timestamp": datetime.now().isoformat()
    }

# ===== EXCEPTION HANDLERS =====
@app.exception_handler(404)
async def not_found_handler(request, exc):
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
            "timestamp": datetime.now().isoformat()
        }
    )

if __name__ == "__main__":
    print("🚀 Iniciando servidor completo de Brain Trader API...")
    print("📊 APIs disponibles:")
    print("   - Brain Trader: /api/v1/brain-trader/*")
    print("   - Mega Mind: /api/v1/mega-mind/*")
    print("   - Documentación: /docs")
    print("   - Health Check: /health")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, reload=False) 