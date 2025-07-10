"""
Modelos de Base de Datos - Sistema de Suscripciones
===================================================
Modelos SQLAlchemy para integrar con MySQL
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class User(Base):
    """Modelo para usuarios en MySQL"""
    __tablename__ = "users"
    
    # Identificación
    user_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(50), nullable=False, unique=True, index=True)
    email = Column(String(254), nullable=False, unique=True, index=True)
    
    # Información personal
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    phone = Column(String(20), nullable=True)
    
    # Autenticación
    password_hash = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Roles y permisos
    role = Column(String(20), default="user")  # user, admin, moderator
    
    # Configuración
    timezone = Column(String(50), default="UTC")
    language = Column(String(10), default="es")
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relaciones
    subscriptions = relationship("UserSubscription", back_populates="user")

class SubscriptionPlan(Base):
    """Modelo para planes de suscripción en MySQL"""
    __tablename__ = "subscription_plans"
    
    # Identificación
    plan_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    plan_type = Column(String(20), nullable=False, unique=True)  # freemium, basic, pro, elite
    price = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    billing_cycle = Column(String(20), default="monthly")
    
    # Capacidades de IA
    traditional_ai = Column(Boolean, default=False)
    reinforcement_learning = Column(Boolean, default=False)
    ensemble_ai = Column(Boolean, default=False)
    lstm_predictions = Column(Boolean, default=False)
    custom_models = Column(Boolean, default=False)
    auto_training = Column(Boolean, default=False)
    
    # Límites de API
    daily_requests = Column(Integer, default=100)
    prediction_days = Column(Integer, default=3)
    backtest_days = Column(Integer, default=30)
    trading_pairs = Column(Integer, default=1)
    alerts_limit = Column(Integer, default=3)
    portfolio_size = Column(Integer, default=1)
    
    # Características de UI
    advanced_charts = Column(Boolean, default=False)
    multiple_timeframes = Column(Boolean, default=False)
    rl_dashboard = Column(Boolean, default=False)
    ai_monitor = Column(Boolean, default=False)
    mt4_integration = Column(Boolean, default=False)
    api_access = Column(Boolean, default=False)
    custom_reports = Column(Boolean, default=False)
    priority_support = Column(Boolean, default=False)
    
    # Configuración de límites
    max_indicators = Column(Integer, default=1)
    max_predictions_per_day = Column(Integer, default=10)
    max_backtests_per_month = Column(Integer, default=5)
    max_portfolios = Column(Integer, default=1)
    
    # Configuración de soporte
    support_level = Column(String(20), default="email")
    response_time_hours = Column(Integer, default=48)
    
    # Descripción y beneficios
    description = Column(Text)
    benefits = Column(JSON)  # Lista de beneficios como JSON
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    subscriptions = relationship("UserSubscription", back_populates="plan")

class UserSubscription(Base):
    """Modelo para suscripciones de usuarios en MySQL"""
    __tablename__ = "user_subscriptions"
    
    # Identificación
    subscription_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    plan_id = Column(String(36), ForeignKey("subscription_plans.plan_id"), nullable=False, index=True)
    plan_type = Column(String(20), nullable=False)  # freemium, basic, pro, elite
    
    # Fechas
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    trial_end_date = Column(DateTime, nullable=True)
    
    # Estado
    status = Column(String(20), nullable=False, default="active")  # active, expired, cancelled, pending, trial
    is_trial = Column(Boolean, default=False)
    
    # Métricas de uso (JSON para flexibilidad)
    usage_metrics = Column(JSON, default=dict)
    
    # Configuración de pagos
    payment_method = Column(String(50), nullable=True)
    auto_renew = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    plan = relationship("SubscriptionPlan", back_populates="subscriptions")
    user = relationship("User", back_populates="subscriptions")
    usage_records = relationship("UsageMetrics", back_populates="subscription")

class UsageMetrics(Base):
    """Modelo para métricas de uso en MySQL"""
    __tablename__ = "usage_metrics"
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    subscription_id = Column(String(36), ForeignKey("user_subscriptions.subscription_id"), nullable=False, index=True)
    date = Column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Métricas de API
    api_requests_today = Column(Integer, default=0)
    predictions_made_today = Column(Integer, default=0)
    backtests_run_today = Column(Integer, default=0)
    alerts_created = Column(Integer, default=0)
    
    # Métricas de IA
    ai_models_used = Column(JSON, default=list)  # Lista de modelos usados
    rl_episodes_trained = Column(Integer, default=0)
    custom_models_created = Column(Integer, default=0)
    
    # Métricas de trading
    trades_executed = Column(Integer, default=0)
    portfolio_value = Column(Float, default=0.0)
    profit_loss = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relaciones
    subscription = relationship("UserSubscription", back_populates="usage_records")

class SubscriptionUpgrade(Base):
    """Modelo para solicitudes de upgrade en MySQL"""
    __tablename__ = "subscription_upgrades"
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    current_plan = Column(String(20), nullable=False)
    target_plan = Column(String(20), nullable=False)
    reason = Column(Text, nullable=True)
    status = Column(String(20), default="pending")  # pending, approved, rejected, completed
    
    # Timestamps
    requested_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SubscriptionPayment(Base):
    """Modelo para pagos de suscripciones en MySQL"""
    __tablename__ = "subscription_payments"
    
    # Identificación
    payment_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    subscription_id = Column(String(36), ForeignKey("user_subscriptions.subscription_id"), nullable=False, index=True)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    
    # Información del pago
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    payment_method = Column(String(50), nullable=False)
    payment_provider = Column(String(50), nullable=False)  # stripe, paypal, etc.
    provider_payment_id = Column(String(100), nullable=True)
    
    # Estado del pago
    status = Column(String(20), default="pending")  # pending, completed, failed, refunded
    billing_cycle = Column(String(20), default="monthly")
    
    # Timestamps
    payment_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class SubscriptionAudit(Base):
    """Modelo para auditoría de suscripciones en MySQL"""
    __tablename__ = "subscription_audit"
    
    # Identificación
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(36), ForeignKey("users.user_id"), nullable=False, index=True)
    subscription_id = Column(String(36), ForeignKey("user_subscriptions.subscription_id"), nullable=False, index=True)
    
    # Información de la acción
    action = Column(String(50), nullable=False)  # created, upgraded, cancelled, renewed, etc.
    old_plan = Column(String(20), nullable=True)
    new_plan = Column(String(20), nullable=True)
    reason = Column(Text, nullable=True)
    
    # Metadatos
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    # Timestamps
    action_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow) 