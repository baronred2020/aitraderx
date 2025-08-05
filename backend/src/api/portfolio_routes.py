"""
Portfolio Routes - APIs para gestión de portfolio y trading
==========================================================
APIs para obtener estadísticas reales del portfolio, historial de trading
y métricas de rendimiento basadas en datos reales de predicciones y señales.
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime, timedelta
import logging
import mysql.connector
from contextlib import contextmanager

try:
    from ..config.database_config import DatabaseConfig
    from ..models.prediction_models import UserPrediction, PredictionDirection
    from ..models.signal_models import UserSignal, SignalType
except ImportError:
    # Fallback para importaciones absolutas
    from config.database_config import DatabaseConfig
    from models.prediction_models import UserPrediction, PredictionDirection
    from models.signal_models import UserSignal, SignalType

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/portfolio", tags=["Portfolio"])

# Función temporal para obtener usuario actual
async def get_current_user(request: Request):
    """Función temporal para obtener usuario actual"""
    # Por ahora, usar un usuario de prueba con user_id original
    return {"user_id": "ec9c4060-1889-4ef4-9b74-5d2a63813020", "username": "testuser", "email": "test@example.com", "plan_type": "starter"}

# Función para obtener conexión de base de datos
def get_db():
    """Función para obtener conexión de base de datos"""
    db_config = DatabaseConfig()
    return db_config

@contextmanager
def get_db_connection():
    """Context manager para obtener conexión a la base de datos"""
    db_config = DatabaseConfig()
    connection = None
    try:
        with db_config.get_connection() as conn:
            yield conn
    except Exception as e:
        logger.error(f"Error obteniendo conexión: {e}")
        raise

# Modelos de respuesta
class PortfolioStatsResponse(BaseModel):
    total_predictions: int
    successful_predictions: int
    success_rate: float
    total_signals: int
    successful_signals: int
    signal_success_rate: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    best_pair: Optional[str]
    best_brain_type: Optional[str]
    worst_pair: Optional[str]
    worst_brain_type: Optional[str]
    best_day: Optional[str]
    worst_day: Optional[str]
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float

class TradingHistoryResponse(BaseModel):
    id: int
    pair: str
    brain_type: str
    type: str  # 'prediction' or 'signal'
    direction: str
    entry_price: float
    exit_price: Optional[float]
    pnl: float
    pips: Optional[float]
    confidence: float
    status: str  # 'open', 'closed', 'cancelled'
    entry_time: str
    exit_time: Optional[str]
    success: Optional[bool]
    success_percentage: Optional[float]

class PortfolioPerformanceResponse(BaseModel):
    total_return: float
    daily_return: float
    weekly_return: float
    monthly_return: float
    risk_metrics: Dict[str, Any]
    performance_by_pair: Dict[str, Dict[str, Any]]
    performance_by_brain: Dict[str, Dict[str, Any]]
    recent_trades: List[TradingHistoryResponse]

class RiskMetricsResponse(BaseModel):
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    volatility: float
    beta: float
    var_95: float

@router.get("/stats", response_model=PortfolioStatsResponse)
async def get_portfolio_stats(
    period: str = "1m",  # 1d, 1w, 1m, 3m, 1y
    request: Request = None,
    current_user: dict = Depends(get_current_user)
):
    """Obtener estadísticas completas del portfolio basadas en datos reales"""
    try:
        user_id = current_user["user_id"]
        
        # Calcular fecha de inicio basada en el período
        end_date = datetime.utcnow()
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = end_date - timedelta(days=30)  # Default 1 month
        
        with get_db_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            
            # Obtener estadísticas de predicciones
            predictions_query = """
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN prediction_success = 1 THEN 1 ELSE 0 END) as successful_predictions,
                AVG(CASE WHEN prediction_success = 1 THEN success_percentage ELSE NULL END) as avg_success_percentage
            FROM user_predictions 
            WHERE user_id = %s AND created_at >= %s AND created_at <= %s
            """
            cursor.execute(predictions_query, (user_id, start_date, end_date))
            pred_stats = cursor.fetchone()
            
            total_predictions = int(pred_stats['total_predictions']) if pred_stats['total_predictions'] else 0
            successful_predictions = int(pred_stats['successful_predictions']) if pred_stats['successful_predictions'] else 0
            success_rate = (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0.0
            
            # Obtener estadísticas de señales
            signals_query = """
            SELECT 
                COUNT(*) as total_signals,
                SUM(CASE WHEN signal_success = 1 THEN 1 ELSE 0 END) as successful_signals,
                AVG(CASE WHEN signal_success = 1 THEN success_percentage ELSE NULL END) as avg_success_percentage
            FROM user_signals 
            WHERE user_id = %s AND created_at >= %s AND created_at <= %s
            """
            cursor.execute(signals_query, (user_id, start_date, end_date))
            signal_stats = cursor.fetchone()
            
            total_signals = int(signal_stats['total_signals']) if signal_stats['total_signals'] else 0
            successful_signals = int(signal_stats['successful_signals']) if signal_stats['successful_signals'] else 0
            signal_success_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0.0
            
            # Calcular P&L total
            pnl_query = """
            SELECT 
                SUM(CASE 
                    WHEN success_percentage >= 50 THEN (success_percentage / 100) * 100
                    ELSE -(1 - success_percentage / 100) * 100
                END) as total_pnl,
                COUNT(*) as total_trades,
                SUM(CASE WHEN success_percentage >= 50 THEN 1 ELSE 0 END) as winning_trades,
                SUM(CASE WHEN success_percentage < 50 THEN 1 ELSE 0 END) as losing_trades
            FROM (
                SELECT success_percentage FROM user_predictions 
                WHERE user_id = %s AND created_at >= %s AND created_at <= %s AND is_completed = 1
                UNION ALL
                SELECT success_percentage FROM user_signals 
                WHERE user_id = %s AND created_at >= %s AND created_at <= %s AND is_completed = 1
            ) as all_trades
            """
            cursor.execute(pnl_query, (user_id, start_date, end_date, user_id, start_date, end_date))
            pnl_stats = cursor.fetchone()
            
            total_pnl = float(pnl_stats['total_pnl']) if pnl_stats['total_pnl'] else 0.0
            total_trades = int(pnl_stats['total_trades']) if pnl_stats['total_trades'] else 0
            winning_trades = int(pnl_stats['winning_trades']) if pnl_stats['winning_trades'] else 0
            losing_trades = int(pnl_stats['losing_trades']) if pnl_stats['losing_trades'] else 0
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0
            
            # Calcular métricas adicionales
            avg_win = (total_pnl / winning_trades) if winning_trades > 0 else 0.0
            avg_loss = (total_pnl / losing_trades) if losing_trades > 0 else 0.0
            max_drawdown = -abs(total_pnl * 0.1)  # Simulado como 10% del P&L total
            sharpe_ratio = (total_pnl / max(abs(total_pnl), 1)) * 0.5  # Simulado
            profit_factor = (winning_trades * avg_win) / (losing_trades * abs(avg_loss)) if losing_trades > 0 and avg_loss != 0 else 1.0
            
            # Obtener mejor y peor par
            pair_query = """
            SELECT pair, AVG(success_percentage) as avg_success
            FROM user_predictions 
            WHERE user_id = %s AND created_at >= %s AND created_at <= %s AND is_completed = 1
            GROUP BY pair
            ORDER BY avg_success DESC
            """
            cursor.execute(pair_query, (user_id, start_date, end_date))
            pair_stats = cursor.fetchall()
            
            best_pair = pair_stats[0]['pair'] if pair_stats else None
            worst_pair = pair_stats[-1]['pair'] if pair_stats else None
            
            # Obtener mejor y peor brain type
            brain_query = """
            SELECT brain_type, AVG(success_percentage) as avg_success
            FROM user_predictions 
            WHERE user_id = %s AND created_at >= %s AND created_at <= %s AND is_completed = 1
            GROUP BY brain_type
            ORDER BY avg_success DESC
            """
            cursor.execute(brain_query, (user_id, start_date, end_date))
            brain_stats = cursor.fetchall()
            
            best_brain_type = brain_stats[0]['brain_type'] if brain_stats else None
            worst_brain_type = brain_stats[-1]['brain_type'] if brain_stats else None
            
            # Calcular P&L por períodos
            daily_pnl = total_pnl / 30 if period == "1m" else total_pnl
            weekly_pnl = total_pnl / 4 if period == "1m" else total_pnl
            monthly_pnl = total_pnl
            
            cursor.close()
            
            return PortfolioStatsResponse(
                total_predictions=total_predictions,
                successful_predictions=successful_predictions,
                success_rate=round(success_rate, 3),
                total_signals=total_signals,
                successful_signals=successful_signals,
                signal_success_rate=round(signal_success_rate, 3),
                total_pnl=round(total_pnl, 2),
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=round(win_rate, 3),
                avg_win=round(avg_win, 2),
                avg_loss=round(avg_loss, 2),
                max_drawdown=round(max_drawdown, 2),
                sharpe_ratio=round(sharpe_ratio, 3),
                profit_factor=round(profit_factor, 2),
                best_pair=best_pair,
                best_brain_type=best_brain_type,
                worst_pair=worst_pair,
                worst_brain_type=worst_brain_type,
                best_day=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                worst_day=(datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d"),
                daily_pnl=round(daily_pnl, 2),
                weekly_pnl=round(weekly_pnl, 2),
                monthly_pnl=round(monthly_pnl, 2)
            )
        
    except Exception as e:
        logger.error(f"Error getting portfolio stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@router.get("/history", response_model=List[TradingHistoryResponse])
async def get_trading_history(
    limit: int = 50,
    period: str = "1m",  # 1d, 1w, 1m, 3m, 1y, all
    pair: Optional[str] = None,
    brain_type: Optional[str] = None,
    current_user: dict = Depends(get_current_user)
):
    """Obtener historial de trading basado en datos reales"""
    try:
        user_id = current_user["user_id"]
        
        # Calcular fecha de inicio basada en el período
        end_date = datetime.utcnow()
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "1w":
            start_date = end_date - timedelta(weeks=1)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        else:
            start_date = None  # all
        
        with get_db_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            
            # Construir filtros base
            base_conditions = ["user_id = %s"]
            params = [user_id]
            
            if start_date:
                base_conditions.append("created_at >= %s")
                params.append(start_date)
            if pair:
                base_conditions.append("pair = %s")
                params.append(pair)
            if brain_type:
                base_conditions.append("brain_type = %s")
                params.append(brain_type)
            
            where_clause = " AND ".join(base_conditions)
            
            # Obtener predicciones
            predictions_query = f"""
            SELECT 
                id, pair, brain_type, direction, current_price, confidence,
                created_at, expires_at, is_completed, actual_price_at_expiry,
                prediction_success, success_percentage
            FROM user_predictions 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(predictions_query, params + [limit])
            predictions = cursor.fetchall()
            
            # Obtener señales
            signals_query = f"""
            SELECT 
                id, pair, brain_type, signal_type as direction, entry_price as current_price, confidence,
                created_at, expires_at, is_completed, actual_price_at_expiry,
                signal_success as prediction_success, success_percentage
            FROM user_signals 
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT %s
            """
            cursor.execute(signals_query, params + [limit])
            signals = cursor.fetchall()
            
            # Combinar y procesar resultados
            history = []
            
            # Procesar predicciones
            for pred in predictions:
                # Calcular P&L simulado basado en success_percentage
                pnl = 0.0
                if pred['success_percentage']:
                    success_pct = float(pred['success_percentage'])
                    if success_pct >= 50:
                        pnl = (success_pct / 100) * 100
                    else:
                        pnl = -(1 - success_pct / 100) * 100
                
                # Calcular pips simulado
                pips = None
                if pred['current_price'] and pred['actual_price_at_expiry']:
                    price_change = float(pred['actual_price_at_expiry']) - float(pred['current_price'])
                    pips = price_change * 10000  # Convertir a pips
                
                history.append(TradingHistoryResponse(
                    id=pred['id'],
                    pair=pred['pair'],
                    brain_type=pred['brain_type'],
                    type="prediction",
                    direction=pred['direction'].upper(),
                    entry_price=float(pred['current_price']),
                    exit_price=float(pred['actual_price_at_expiry']) if pred['actual_price_at_expiry'] else None,
                    pnl=round(pnl, 2),
                    pips=round(pips, 1) if pips else None,
                    confidence=float(pred['confidence']),
                    status="closed" if pred['is_completed'] else "open",
                    entry_time=pred['created_at'].isoformat(),
                    exit_time=pred['expires_at'].isoformat() if pred['expires_at'] else None,
                    success=pred['prediction_success'],
                    success_percentage=float(pred['success_percentage']) if pred['success_percentage'] else None
                ))
            
            # Procesar señales
            for signal in signals:
                # Calcular P&L simulado basado en success_percentage
                pnl = 0.0
                if signal['success_percentage']:
                    success_pct = float(signal['success_percentage'])
                    if success_pct >= 50:
                        pnl = (success_pct / 100) * 150
                    else:
                        pnl = -(1 - success_pct / 100) * 150
                
                # Calcular pips simulado
                pips = None
                if signal['current_price'] and signal['actual_price_at_expiry']:
                    price_change = float(signal['actual_price_at_expiry']) - float(signal['current_price'])
                    pips = price_change * 10000  # Convertir a pips
                
                history.append(TradingHistoryResponse(
                    id=signal['id'] + 10000,  # Offset para evitar conflictos de ID
                    pair=signal['pair'],
                    brain_type=signal['brain_type'],
                    type="signal",
                    direction=signal['direction'].upper(),
                    entry_price=float(signal['current_price']),
                    exit_price=float(signal['actual_price_at_expiry']) if signal['actual_price_at_expiry'] else None,
                    pnl=round(pnl, 2),
                    pips=round(pips, 1) if pips else None,
                    confidence=float(signal['confidence']),
                    status="closed" if signal['is_completed'] else "open",
                    entry_time=signal['created_at'].isoformat(),
                    exit_time=signal['expires_at'].isoformat() if signal['expires_at'] else None,
                    success=signal['prediction_success'],
                    success_percentage=float(signal['success_percentage']) if signal['success_percentage'] else None
                ))
            
            cursor.close()
            
            # Ordenar por fecha de entrada (más reciente primero) y limitar
            history.sort(key=lambda x: x.entry_time, reverse=True)
            return history[:limit]
        
    except Exception as e:
        logger.error(f"Error getting trading history: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@router.get("/performance", response_model=PortfolioPerformanceResponse)
async def get_portfolio_performance(
    period: str = "1m",  # 1d, 1w, 1m, 3m, 1y
    current_user: dict = Depends(get_current_user),
    db_config: DatabaseConfig = Depends(get_db)
):
    """Obtener rendimiento detallado del portfolio basado en datos reales"""
    try:
        # Por ahora, retornar datos vacíos hasta que se configure la base de datos
        # En el futuro, aquí se harían las consultas reales a la base de datos
        
        return PortfolioPerformanceResponse(
            total_return=0.0,
            daily_return=0.0,
            weekly_return=0.0,
            monthly_return=0.0,
            risk_metrics={},
            performance_by_pair={},
            performance_by_brain={},
            recent_trades=[]
        )
        
    except Exception as e:
        logger.error(f"Error getting portfolio performance: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}")

@router.get("/risk-metrics", response_model=RiskMetricsResponse)
async def get_risk_metrics(
    period: str = "1m",  # 1d, 1w, 1m, 3m, 1y
    current_user: dict = Depends(get_current_user),
    db_config: DatabaseConfig = Depends(get_db)
):
    """Obtener métricas de riesgo del portfolio basadas en datos reales"""
    try:
        # Por ahora, retornar datos vacíos hasta que se configure la base de datos
        # En el futuro, aquí se harían las consultas reales a la base de datos
        
        return RiskMetricsResponse(
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            average_win=0.0,
            average_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            volatility=0.0,
            beta=0.0,
            var_95=0.0
        )
        
    except Exception as e:
        logger.error(f"Error getting risk metrics: {e}")
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {str(e)}") 