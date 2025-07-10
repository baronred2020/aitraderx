"""
Script de Inicialización de Base de Datos
========================================
Crea las tablas y datos iniciales en trading_db
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database_config import create_tables, engine, SessionLocal
from models.database_models import SubscriptionPlan
from datetime import datetime
import uuid

def create_initial_plans():
    """Crea los planes de suscripción iniciales"""
    db = SessionLocal()
    try:
        # Verificar si ya existen planes
        existing_plans = db.query(SubscriptionPlan).count()
        if existing_plans > 0:
            print("Los planes ya existen en la base de datos")
            return
        
        # Plan Freemium
        freemium_plan = SubscriptionPlan(
            plan_id=str(uuid.uuid4()),
            name="Freemium",
            plan_type="freemium",
            price=0.0,
            currency="USD",
            billing_cycle="monthly",
            traditional_ai=True,
            reinforcement_learning=False,
            ensemble_ai=False,
            lstm_predictions=False,
            custom_models=False,
            auto_training=False,
            daily_requests=50,
            prediction_days=1,
            backtest_days=7,
            trading_pairs=1,
            alerts_limit=1,
            portfolio_size=1,
            advanced_charts=False,
            multiple_timeframes=False,
            rl_dashboard=False,
            ai_monitor=False,
            mt4_integration=False,
            api_access=False,
            custom_reports=False,
            priority_support=False,
            max_indicators=3,
            max_predictions_per_day=5,
            max_backtests_per_month=2,
            max_portfolios=1,
            support_level="email",
            response_time_hours=72,
            description="Plan gratuito con funcionalidades básicas de IA",
            benefits=["Acceso a IA tradicional", "5 predicciones diarias", "1 par de trading", "Soporte por email"],
            created_at=datetime.utcnow()
        )
        
        # Plan Basic
        basic_plan = SubscriptionPlan(
            plan_id=str(uuid.uuid4()),
            name="Basic",
            plan_type="basic",
            price=29.99,
            currency="USD",
            billing_cycle="monthly",
            traditional_ai=True,
            reinforcement_learning=True,
            ensemble_ai=False,
            lstm_predictions=False,
            custom_models=False,
            auto_training=False,
            daily_requests=200,
            prediction_days=3,
            backtest_days=30,
            trading_pairs=3,
            alerts_limit=5,
            portfolio_size=2,
            advanced_charts=True,
            multiple_timeframes=True,
            rl_dashboard=True,
            ai_monitor=False,
            mt4_integration=False,
            api_access=False,
            custom_reports=False,
            priority_support=False,
            max_indicators=5,
            max_predictions_per_day=20,
            max_backtests_per_month=10,
            max_portfolios=2,
            support_level="email",
            response_time_hours=48,
            description="Plan básico con IA tradicional y RL",
            benefits=["IA tradicional + RL", "20 predicciones diarias", "3 pares de trading", "Gráficos avanzados", "Dashboard RL"],
            created_at=datetime.utcnow()
        )
        
        # Plan Pro
        pro_plan = SubscriptionPlan(
            plan_id=str(uuid.uuid4()),
            name="Pro",
            plan_type="pro",
            price=79.99,
            currency="USD",
            billing_cycle="monthly",
            traditional_ai=True,
            reinforcement_learning=True,
            ensemble_ai=True,
            lstm_predictions=True,
            custom_models=False,
            auto_training=False,
            daily_requests=500,
            prediction_days=7,
            backtest_days=90,
            trading_pairs=10,
            alerts_limit=15,
            portfolio_size=5,
            advanced_charts=True,
            multiple_timeframes=True,
            rl_dashboard=True,
            ai_monitor=True,
            mt4_integration=True,
            api_access=True,
            custom_reports=False,
            priority_support=False,
            max_indicators=10,
            max_predictions_per_day=50,
            max_backtests_per_month=25,
            max_portfolios=5,
            support_level="chat",
            response_time_hours=24,
            description="Plan profesional con todas las IAs",
            benefits=["Todas las IAs", "50 predicciones diarias", "10 pares de trading", "Monitor IA", "Integración MT4", "API access"],
            created_at=datetime.utcnow()
        )
        
        # Plan Elite
        elite_plan = SubscriptionPlan(
            plan_id=str(uuid.uuid4()),
            name="Elite",
            plan_type="elite",
            price=199.99,
            currency="USD",
            billing_cycle="monthly",
            traditional_ai=True,
            reinforcement_learning=True,
            ensemble_ai=True,
            lstm_predictions=True,
            custom_models=True,
            auto_training=True,
            daily_requests=1000,
            prediction_days=14,
            backtest_days=365,
            trading_pairs=50,
            alerts_limit=50,
            portfolio_size=20,
            advanced_charts=True,
            multiple_timeframes=True,
            rl_dashboard=True,
            ai_monitor=True,
            mt4_integration=True,
            api_access=True,
            custom_reports=True,
            priority_support=True,
            max_indicators=20,
            max_predictions_per_day=100,
            max_backtests_per_month=100,
            max_portfolios=20,
            support_level="phone",
            response_time_hours=4,
            description="Plan elite con funcionalidades completas",
            benefits=["Todas las funcionalidades", "100 predicciones diarias", "50 pares de trading", "Modelos personalizados", "Auto-entrenamiento", "Soporte prioritario", "Reportes personalizados"],
            created_at=datetime.utcnow()
        )
        
        # Agregar planes a la base de datos
        db.add(freemium_plan)
        db.add(basic_plan)
        db.add(pro_plan)
        db.add(elite_plan)
        
        db.commit()
        print("Planes de suscripción creados exitosamente")
        
    except Exception as e:
        db.rollback()
        print(f"Error creando planes: {e}")
    finally:
        db.close()

def main():
    """Función principal para inicializar la base de datos"""
    try:
        print("Inicializando base de datos...")
        
        # Crear tablas
        print("Creando tablas...")
        create_tables()
        print("Tablas creadas exitosamente")
        
        # Crear planes iniciales
        print("Creando planes de suscripción...")
        create_initial_plans()
        
        print("Base de datos inicializada correctamente")
        
    except Exception as e:
        print(f"Error inicializando base de datos: {e}")

if __name__ == "__main__":
    main() 