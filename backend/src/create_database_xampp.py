"""
Script de Creación de Base de Datos - XAMPP
===========================================
Script completo para crear la base de datos MySQL en XAMPP con todas las tablas
necesarias para el sistema de trading con suscripciones.

INSTRUCCIONES:
1. Asegúrate de que XAMPP esté corriendo (Apache y MySQL)
2. Ejecuta este script: python create_database_xampp.py
3. El script creará la base de datos y todas las tablas
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
import logging
from datetime import datetime
import uuid

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseCreator:
    """Clase para crear la base de datos completa en XAMPP"""
    
    def __init__(self):
        # Configuración de XAMPP MySQL
        self.host = "localhost"
        self.port = 3306
        self.user = "root"  # Usuario por defecto de XAMPP
        self.password = ""   # Contraseña vacía por defecto en XAMPP
        self.database_name = "trading_db"
        
        # Conexión inicial (sin especificar base de datos)
        self.connection = None
        self.cursor = None
    
    def connect_to_mysql(self):
        """Conectar a MySQL sin especificar base de datos"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            self.cursor = self.connection.cursor()
            logger.info("✅ Conexión a MySQL establecida")
            return True
        except Error as e:
            logger.error(f"❌ Error conectando a MySQL: {e}")
            return False
    
    def create_database(self):
        """Crear la base de datos si no existe"""
        try:
            # Crear base de datos
            create_db_query = f"CREATE DATABASE IF NOT EXISTS {self.database_name} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            self.cursor.execute(create_db_query)
            self.connection.commit()
            logger.info(f"✅ Base de datos '{self.database_name}' creada/verificada")
            
            # Usar la base de datos
            self.cursor.execute(f"USE {self.database_name}")
            logger.info(f"✅ Usando base de datos '{self.database_name}'")
            
            return True
        except Error as e:
            logger.error(f"❌ Error creando base de datos: {e}")
            return False
    
    def create_users_table(self):
        """Crear tabla de usuarios"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS users (
                user_id VARCHAR(100) PRIMARY KEY,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                role ENUM('admin', 'user') DEFAULT 'user',
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT FALSE,
                email_verified_at DATETIME,
                last_login DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_email (email),
                INDEX idx_username (username),
                INDEX idx_role (role)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'users' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla users: {e}")
            return False
    
    def create_subscription_plans_table(self):
        """Crear tabla de planes de suscripción"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS subscription_plans (
                plan_id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                plan_type VARCHAR(20) NOT NULL UNIQUE,
                price FLOAT NOT NULL,
                currency VARCHAR(3) DEFAULT 'USD',
                billing_cycle VARCHAR(20) DEFAULT 'monthly',
                
                -- Capacidades de IA
                traditional_ai BOOLEAN DEFAULT FALSE,
                reinforcement_learning BOOLEAN DEFAULT FALSE,
                ensemble_ai BOOLEAN DEFAULT FALSE,
                lstm_predictions BOOLEAN DEFAULT FALSE,
                custom_models BOOLEAN DEFAULT FALSE,
                auto_training BOOLEAN DEFAULT FALSE,
                
                -- Límites de API
                daily_requests INT DEFAULT 100,
                prediction_days INT DEFAULT 3,
                backtest_days INT DEFAULT 30,
                trading_pairs INT DEFAULT 1,
                alerts_limit INT DEFAULT 3,
                portfolio_size INT DEFAULT 1,
                
                -- Características de UI
                advanced_charts BOOLEAN DEFAULT FALSE,
                multiple_timeframes BOOLEAN DEFAULT FALSE,
                rl_dashboard BOOLEAN DEFAULT FALSE,
                ai_monitor BOOLEAN DEFAULT FALSE,
                mt4_integration BOOLEAN DEFAULT FALSE,
                api_access BOOLEAN DEFAULT FALSE,
                custom_reports BOOLEAN DEFAULT FALSE,
                priority_support BOOLEAN DEFAULT FALSE,
                
                -- Configuración de límites
                max_indicators INT DEFAULT 1,
                max_predictions_per_day INT DEFAULT 10,
                max_backtests_per_month INT DEFAULT 5,
                max_portfolios INT DEFAULT 1,
                
                -- Configuración de soporte
                support_level VARCHAR(20) DEFAULT 'email',
                response_time_hours INT DEFAULT 48,
                
                -- Descripción y beneficios
                description TEXT,
                benefits JSON,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                INDEX idx_plan_type (plan_type),
                INDEX idx_price (price)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'subscription_plans' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla subscription_plans: {e}")
            return False
    
    def create_user_subscriptions_table(self):
        """Crear tabla de suscripciones de usuarios"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS user_subscriptions (
                subscription_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                plan_id VARCHAR(36) NOT NULL,
                plan_type VARCHAR(20) NOT NULL,
                
                -- Fechas
                start_date DATETIME NOT NULL,
                end_date DATETIME NOT NULL,
                trial_end_date DATETIME,
                
                -- Estado
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                is_trial BOOLEAN DEFAULT FALSE,
                
                -- Métricas de uso
                usage_metrics JSON DEFAULT '{}',
                
                -- Configuración de pagos
                payment_method VARCHAR(50),
                auto_renew BOOLEAN DEFAULT TRUE,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (plan_id) REFERENCES subscription_plans(plan_id) ON DELETE CASCADE,
                
                INDEX idx_user_id (user_id),
                INDEX idx_plan_id (plan_id),
                INDEX idx_status (status),
                INDEX idx_end_date (end_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'user_subscriptions' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla user_subscriptions: {e}")
            return False
    
    def create_usage_metrics_table(self):
        """Crear tabla de métricas de uso"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS usage_metrics (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                subscription_id VARCHAR(36) NOT NULL,
                date DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                
                -- Métricas de API
                api_requests_today INT DEFAULT 0,
                predictions_made_today INT DEFAULT 0,
                backtests_run_today INT DEFAULT 0,
                alerts_created INT DEFAULT 0,
                
                -- Métricas de IA
                ai_models_used JSON DEFAULT '[]',
                rl_episodes_trained INT DEFAULT 0,
                custom_models_created INT DEFAULT 0,
                
                -- Métricas de trading
                trades_executed INT DEFAULT 0,
                portfolio_value FLOAT DEFAULT 0.0,
                profit_loss FLOAT DEFAULT 0.0,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (subscription_id) REFERENCES user_subscriptions(subscription_id) ON DELETE CASCADE,
                
                INDEX idx_user_id (user_id),
                INDEX idx_subscription_id (subscription_id),
                INDEX idx_date (date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'usage_metrics' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla usage_metrics: {e}")
            return False
    
    def create_portfolios_table(self):
        """Crear tabla de portafolios"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS portfolios (
                portfolio_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                name VARCHAR(100) NOT NULL,
                description TEXT,
                initial_balance FLOAT NOT NULL DEFAULT 10000.0,
                current_balance FLOAT NOT NULL DEFAULT 10000.0,
                total_value FLOAT NOT NULL DEFAULT 10000.0,
                total_pnl FLOAT DEFAULT 0.0,
                total_pnl_percent FLOAT DEFAULT 0.0,
                is_active BOOLEAN DEFAULT TRUE,
                is_default BOOLEAN DEFAULT FALSE,
                
                -- Configuración de riesgo
                max_position_size FLOAT DEFAULT 0.05,
                max_daily_loss FLOAT DEFAULT 0.05,
                risk_free_rate FLOAT DEFAULT 0.02,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                
                INDEX idx_user_id (user_id),
                INDEX idx_is_active (is_active)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'portfolios' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla portfolios: {e}")
            return False
    
    def create_positions_table(self):
        """Crear tabla de posiciones"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS positions (
                position_id VARCHAR(36) PRIMARY KEY,
                portfolio_id VARCHAR(36) NOT NULL,
                user_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                position_type ENUM('LONG', 'SHORT') NOT NULL,
                quantity FLOAT NOT NULL,
                entry_price FLOAT NOT NULL,
                current_price FLOAT NOT NULL,
                avg_price FLOAT NOT NULL,
                value FLOAT NOT NULL,
                pnl FLOAT DEFAULT 0.0,
                pnl_percent FLOAT DEFAULT 0.0,
                allocation_percent FLOAT DEFAULT 0.0,
                is_open BOOLEAN DEFAULT TRUE,
                
                -- Metadatos
                entry_date DATETIME NOT NULL,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                
                INDEX idx_portfolio_id (portfolio_id),
                INDEX idx_user_id (user_id),
                INDEX idx_symbol (symbol),
                INDEX idx_is_open (is_open)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'positions' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla positions: {e}")
            return False
    
    def create_trades_table(self):
        """Crear tabla de operaciones"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS trades (
                trade_id VARCHAR(36) PRIMARY KEY,
                portfolio_id VARCHAR(36) NOT NULL,
                user_id VARCHAR(100) NOT NULL,
                position_id VARCHAR(36),
                symbol VARCHAR(20) NOT NULL,
                trade_type ENUM('BUY', 'SELL') NOT NULL,
                quantity FLOAT NOT NULL,
                price FLOAT NOT NULL,
                value FLOAT NOT NULL,
                commission FLOAT DEFAULT 0.0,
                pnl FLOAT DEFAULT 0.0,
                pnl_percent FLOAT DEFAULT 0.0,
                
                -- Estado de la orden
                order_status ENUM('pending', 'filled', 'cancelled', 'rejected') DEFAULT 'pending',
                execution_time DATETIME,
                
                -- Metadatos
                trade_date DATETIME NOT NULL,
                notes TEXT,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (portfolio_id) REFERENCES portfolios(portfolio_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (position_id) REFERENCES positions(position_id) ON DELETE SET NULL,
                
                INDEX idx_portfolio_id (portfolio_id),
                INDEX idx_user_id (user_id),
                INDEX idx_symbol (symbol),
                INDEX idx_trade_date (trade_date),
                INDEX idx_order_status (order_status)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'trades' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla trades: {e}")
            return False
    
    def create_ai_models_table(self):
        """Crear tabla de modelos de IA"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS ai_models (
                model_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                model_name VARCHAR(100) NOT NULL,
                model_type ENUM('traditional', 'lstm', 'rl_dqn', 'rl_ppo', 'ensemble', 'custom') NOT NULL,
                model_version VARCHAR(20) NOT NULL,
                file_path VARCHAR(255) NOT NULL,
                
                -- Métricas del modelo
                accuracy FLOAT DEFAULT 0.0,
                precision_score FLOAT DEFAULT 0.0,
                recall FLOAT DEFAULT 0.0,
                f1_score FLOAT DEFAULT 0.0,
                mse FLOAT DEFAULT 0.0,
                mae FLOAT DEFAULT 0.0,
                
                -- Configuración
                parameters JSON,
                features_used JSON,
                training_data_size INT DEFAULT 0,
                training_duration_seconds INT DEFAULT 0,
                
                -- Estado
                is_active BOOLEAN DEFAULT TRUE,
                is_production BOOLEAN DEFAULT FALSE,
                last_used DATETIME,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                
                INDEX idx_user_id (user_id),
                INDEX idx_model_type (model_type),
                INDEX idx_is_active (is_active),
                INDEX idx_is_production (is_production)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'ai_models' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla ai_models: {e}")
            return False
    
    def create_predictions_table(self):
        """Crear tabla de predicciones"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                model_id VARCHAR(36),
                symbol VARCHAR(20) NOT NULL,
                prediction_type ENUM('price', 'signal', 'trend') NOT NULL,
                
                -- Datos de entrada
                input_data JSON NOT NULL,
                technical_indicators JSON,
                
                -- Resultados
                predicted_value FLOAT,
                predicted_signal ENUM('BUY', 'SELL', 'HOLD'),
                confidence FLOAT DEFAULT 0.0,
                target_price FLOAT,
                timeframe VARCHAR(20),
                
                -- Metadatos
                prediction_date DATETIME NOT NULL,
                actual_value FLOAT,
                actual_signal VARCHAR(10),
                accuracy FLOAT,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (model_id) REFERENCES ai_models(model_id) ON DELETE SET NULL,
                
                INDEX idx_user_id (user_id),
                INDEX idx_symbol (symbol),
                INDEX idx_prediction_date (prediction_date),
                INDEX idx_model_id (model_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'predictions' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla predictions: {e}")
            return False
    
    def create_alerts_table(self):
        """Crear tabla de alertas"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id VARCHAR(36) PRIMARY KEY,
                user_id VARCHAR(100) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                alert_type ENUM('price', 'signal', 'technical', 'custom') NOT NULL,
                condition_type ENUM('above', 'below', 'equals', 'crosses') NOT NULL,
                condition_value FLOAT NOT NULL,
                message TEXT,
                
                -- Estado
                is_active BOOLEAN DEFAULT TRUE,
                is_triggered BOOLEAN DEFAULT FALSE,
                triggered_at DATETIME,
                triggered_value FLOAT,
                
                -- Configuración
                notification_email BOOLEAN DEFAULT FALSE,
                notification_sms BOOLEAN DEFAULT FALSE,
                notification_push BOOLEAN DEFAULT FALSE,
                
                -- Timestamps
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                -- Foreign Keys
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                
                INDEX idx_user_id (user_id),
                INDEX idx_symbol (symbol),
                INDEX idx_is_active (is_active),
                INDEX idx_is_triggered (is_triggered)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'alerts' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla alerts: {e}")
            return False
    
    def create_audit_logs_table(self):
        """Crear tabla de logs de auditoría"""
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS audit_logs (
                log_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(100),
                action VARCHAR(100) NOT NULL,
                table_name VARCHAR(50),
                record_id VARCHAR(36),
                
                -- Datos de la acción
                old_values JSON,
                new_values JSON,
                action_details TEXT,
                
                -- Metadatos
                ip_address VARCHAR(45),
                user_agent TEXT,
                session_id VARCHAR(100),
                
                -- Timestamps
                action_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                
                INDEX idx_user_id (user_id),
                INDEX idx_action (action),
                INDEX idx_table_name (table_name),
                INDEX idx_action_date (action_date)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.cursor.execute(create_table_query)
            self.connection.commit()
            logger.info("✅ Tabla 'audit_logs' creada")
            return True
        except Error as e:
            logger.error(f"❌ Error creando tabla audit_logs: {e}")
            return False
    
    def insert_default_plans(self):
        """Insertar planes por defecto"""
        import json
        try:
            # Definir todos los campos posibles
            all_fields = [
                'plan_id', 'name', 'plan_type', 'price', 'currency', 'billing_cycle',
                'description', 'traditional_ai', 'reinforcement_learning', 'ensemble_ai', 'lstm_predictions',
                'custom_models', 'auto_training', 'daily_requests', 'prediction_days', 'backtest_days',
                'trading_pairs', 'alerts_limit', 'portfolio_size', 'advanced_charts', 'multiple_timeframes',
                'rl_dashboard', 'ai_monitor', 'mt4_integration', 'api_access', 'custom_reports',
                'priority_support', 'max_indicators', 'max_predictions_per_day', 'max_backtests_per_month',
                'max_portfolios', 'support_level', 'response_time_hours', 'benefits'
            ]

            # Helper para convertir bool a int y rellenar campos faltantes
            def normalize_plan(plan):
                normalized = {}
                for field in all_fields:
                    value = plan.get(field, None)
                    if isinstance(value, bool):
                        normalized[field] = int(value)
                    elif value is None:
                        # Defaults
                        if field in [
                            'traditional_ai', 'reinforcement_learning', 'ensemble_ai', 'lstm_predictions',
                            'custom_models', 'auto_training', 'advanced_charts', 'multiple_timeframes',
                            'rl_dashboard', 'ai_monitor', 'mt4_integration', 'api_access', 'custom_reports',
                            'priority_support'
                        ]:
                            normalized[field] = 0
                        elif field in [
                            'daily_requests', 'prediction_days', 'backtest_days', 'trading_pairs', 'alerts_limit',
                            'portfolio_size', 'max_indicators', 'max_predictions_per_day', 'max_backtests_per_month',
                            'max_portfolios', 'response_time_hours'
                        ]:
                            normalized[field] = 0
                        elif field == 'benefits':
                            normalized[field] = json.dumps([])
                        elif field == 'currency':
                            normalized[field] = 'USD'
                        elif field == 'billing_cycle':
                            normalized[field] = 'monthly'
                        else:
                            normalized[field] = None
                    elif field == 'benefits':
                        # Asegurar que benefits sea JSON
                        if isinstance(value, str):
                            try:
                                normalized[field] = json.dumps(json.loads(value))
                            except Exception:
                                normalized[field] = json.dumps([value])
                        else:
                            normalized[field] = json.dumps(value)
                    else:
                        normalized[field] = value
                return normalized

            # Planes
            freemium_plan = {
                'plan_id': str(uuid.uuid4()),
                'name': 'Freemium',
                'plan_type': 'freemium',
                'price': 0.0,
                'description': 'Plan gratuito para empezar con trading básico',
                'traditional_ai': True,
                'daily_requests': 100,
                'prediction_days': 3,
                'backtest_days': 30,
                'trading_pairs': 1,
                'alerts_limit': 3,
                'portfolio_size': 1,
                'max_indicators': 1,
                'max_predictions_per_day': 10,
                'max_backtests_per_month': 5,
                'max_portfolios': 1,
                'support_level': 'community',
                'response_time_hours': 72,
                'benefits': ["AI Tradicional básica", "1 indicador técnico", "Predicciones limitadas", "Backtesting básico", "1 par de trading", "3 alertas básicas"]
            }
            basic_plan = {
                'plan_id': str(uuid.uuid4()),
                'name': 'Básico',
                'plan_type': 'basic',
                'price': 29.0,
                'description': 'Plan para usuarios serios con capacidades avanzadas',
                'traditional_ai': True,
                'daily_requests': 500,
                'prediction_days': 7,
                'backtest_days': 90,
                'trading_pairs': 5,
                'alerts_limit': 10,
                'portfolio_size': 3,
                'advanced_charts': True,
                'max_indicators': 3,
                'max_predictions_per_day': 50,
                'max_backtests_per_month': 20,
                'max_portfolios': 3,
                'support_level': 'email',
                'response_time_hours': 48,
                'benefits': ["AI Tradicional completa", "3 indicadores técnicos", "Predicciones mejoradas", "Backtesting avanzado", "5 pares de trading", "10 alertas avanzadas", "Gráficos avanzados"]
            }
            pro_plan = {
                'plan_id': str(uuid.uuid4()),
                'name': 'Pro',
                'plan_type': 'pro',
                'price': 99.0,
                'description': 'Plan para traders profesionales con IA avanzada',
                'traditional_ai': True,
                'reinforcement_learning': True,
                'ensemble_ai': True,
                'lstm_predictions': True,
                'auto_training': True,
                'daily_requests': 2000,
                'prediction_days': 14,
                'backtest_days': 365,
                'trading_pairs': 50,
                'alerts_limit': 50,
                'portfolio_size': 10,
                'advanced_charts': True,
                'multiple_timeframes': True,
                'rl_dashboard': True,
                'ai_monitor': True,
                'mt4_integration': True,
                'custom_reports': True,
                'max_indicators': 10,
                'max_predictions_per_day': 200,
                'max_backtests_per_month': 100,
                'max_portfolios': 10,
                'support_level': 'email',
                'response_time_hours': 24,
                'benefits': ["AI Tradicional Premium + LSTM", "Reinforcement Learning (DQN)", "Ensemble AI", "Todos los indicadores técnicos", "Predicciones avanzadas", "Backtesting profesional", "Todos los pares de trading", "Risk Management básico", "Portfolio Optimization básico", "Integración MT4 básica", "RL Dashboard", "AI Monitor"]
            }
            elite_plan = {
                'plan_id': str(uuid.uuid4()),
                'name': 'Elite',
                'plan_type': 'elite',
                'price': 299.0,
                'description': 'Plan para traders institucionales con capacidades completas',
                'traditional_ai': True,
                'reinforcement_learning': True,
                'ensemble_ai': True,
                'lstm_predictions': True,
                'custom_models': True,
                'auto_training': True,
                'daily_requests': 10000,
                'prediction_days': 30,
                'backtest_days': 1825,
                'trading_pairs': 1000,
                'alerts_limit': 1000,
                'portfolio_size': 100,
                'advanced_charts': True,
                'multiple_timeframes': True,
                'rl_dashboard': True,
                'ai_monitor': True,
                'mt4_integration': True,
                'api_access': True,
                'custom_reports': True,
                'priority_support': True,
                'max_indicators': 100,
                'max_predictions_per_day': 1000,
                'max_backtests_per_month': 500,
                'max_portfolios': 100,
                'support_level': 'phone',
                'response_time_hours': 4,
                'benefits': ["AI Tradicional Elite + máxima precisión", "Reinforcement Learning completo (DQN + PPO)", "Ensemble AI avanzado optimizado", "Predicciones elite (30 días)", "Backtesting institucional", "Todos los instrumentos (Forex, Stocks, Crypto)", "Risk Management avanzado", "Portfolio Optimization avanzado", "Auto-Trading con AI", "Custom Models personalizados", "Integración MT4 completa", "API personalizada", "Soporte prioritario 24/7"]
            }
            plans = [freemium_plan, basic_plan, pro_plan, elite_plan]
            for plan in plans:
                normalized = normalize_plan(plan)
                insert_query = """
                INSERT INTO subscription_plans (
                    plan_id, name, plan_type, price, currency, billing_cycle, description, traditional_ai, 
                    reinforcement_learning, ensemble_ai, lstm_predictions, custom_models, auto_training, 
                    daily_requests, prediction_days, backtest_days, trading_pairs, alerts_limit, portfolio_size, 
                    advanced_charts, multiple_timeframes, rl_dashboard, ai_monitor, mt4_integration, api_access, 
                    custom_reports, priority_support, max_indicators, max_predictions_per_day, max_backtests_per_month, 
                    max_portfolios, support_level, response_time_hours, benefits
                ) VALUES (
                    %(plan_id)s, %(name)s, %(plan_type)s, %(price)s, %(currency)s, %(billing_cycle)s, %(description)s, %(traditional_ai)s, 
                    %(reinforcement_learning)s, %(ensemble_ai)s, %(lstm_predictions)s, %(custom_models)s, %(auto_training)s, 
                    %(daily_requests)s, %(prediction_days)s, %(backtest_days)s, %(trading_pairs)s, %(alerts_limit)s, %(portfolio_size)s, 
                    %(advanced_charts)s, %(multiple_timeframes)s, %(rl_dashboard)s, %(ai_monitor)s, %(mt4_integration)s, %(api_access)s, 
                    %(custom_reports)s, %(priority_support)s, %(max_indicators)s, %(max_predictions_per_day)s, %(max_backtests_per_month)s, 
                    %(max_portfolios)s, %(support_level)s, %(response_time_hours)s, %(benefits)s
                )
                """
                self.cursor.execute(insert_query, normalized)
            self.connection.commit()
            logger.info("✅ Planes por defecto insertados")
            return True
        except Error as e:
            logger.error(f"❌ Error insertando planes: {e}")
            return False
    
    def create_admin_user(self):
        """Crear usuario administrador por defecto"""
        try:
            admin_user = {
                'user_id': str(uuid.uuid4()),
                'username': 'admin',
                'email': 'admin@aitraderx.com',
                'password_hash': '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8KqKqKq',  # password: admin123
                'first_name': 'Admin',
                'last_name': 'System',
                'role': 'admin',
                'is_active': True,
                'is_verified': True
            }
            
            insert_query = """
            INSERT INTO users (
                user_id, username, email, password_hash, first_name, last_name, 
                role, is_active, is_verified
            ) VALUES (
                %(user_id)s, %(username)s, %(email)s, %(password_hash)s, 
                %(first_name)s, %(last_name)s, %(role)s, %(is_active)s, %(is_verified)s
            )
            """
            self.cursor.execute(insert_query, admin_user)
            self.connection.commit()
            logger.info("✅ Usuario administrador creado")
            return True
        except Error as e:
            logger.error(f"❌ Error creando usuario admin: {e}")
            return False
    
    def verify_tables(self):
        """Verificar que todas las tablas se crearon correctamente"""
        try:
            expected_tables = [
                'users',
                'subscription_plans',
                'user_subscriptions',
                'usage_metrics',
                'portfolios',
                'positions',
                'trades',
                'ai_models',
                'predictions',
                'alerts',
                'audit_logs'
            ]
            
            self.cursor.execute("SHOW TABLES")
            existing_tables = [row[0] for row in self.cursor.fetchall()]
            
            logger.info("\n📊 VERIFICACIÓN DE TABLAS:")
            logger.info("-" * 40)
            
            all_tables_exist = True
            for table in expected_tables:
                if table in existing_tables:
                    logger.info(f"✅ Tabla '{table}' creada")
                else:
                    logger.info(f"❌ Tabla '{table}' NO encontrada")
                    all_tables_exist = False
            
            # Verificar datos en subscription_plans
            self.cursor.execute("SELECT COUNT(*) FROM subscription_plans")
            plan_count = self.cursor.fetchone()[0]
            logger.info(f"📈 Planes en base de datos: {plan_count}")
            
            if plan_count == 4:
                logger.info("✅ Todos los planes por defecto insertados")
            else:
                logger.info(f"⚠️  Solo {plan_count}/4 planes encontrados")
            
            # Verificar usuario admin
            self.cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
            admin_count = self.cursor.fetchone()[0]
            logger.info(f"👤 Usuarios admin: {admin_count}")
            
            return all_tables_exist
        except Error as e:
            logger.error(f"❌ Error verificando tablas: {e}")
            return False
    
    def close_connection(self):
        """Cerrar conexión a la base de datos"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        logger.info("🔌 Conexión cerrada")
    
    def create_database_complete(self):
        """Crear la base de datos completa"""
        try:
            logger.info("🚀 INICIANDO CREACIÓN DE BASE DE DATOS")
            logger.info("=" * 50)
            
            # 1. Conectar a MySQL
            if not self.connect_to_mysql():
                return False
            
            # 2. Crear base de datos
            if not self.create_database():
                return False
            
            # 3. Crear todas las tablas
            tables_creation = [
                self.create_users_table,
                self.create_subscription_plans_table,
                self.create_user_subscriptions_table,
                self.create_usage_metrics_table,
                self.create_portfolios_table,
                self.create_positions_table,
                self.create_trades_table,
                self.create_ai_models_table,
                self.create_predictions_table,
                self.create_alerts_table,
                self.create_audit_logs_table
            ]
            
            for create_table_func in tables_creation:
                if not create_table_func():
                    return False
            
            # 4. Insertar datos por defecto
            if not self.insert_default_plans():
                return False
            
            if not self.create_admin_user():
                return False
            
            # 5. Verificar todo
            if not self.verify_tables():
                return False
            
            logger.info("\n🎉 ¡BASE DE DATOS CREADA EXITOSAMENTE!")
            logger.info("=" * 50)
            logger.info("📋 RESUMEN:")
            logger.info("• 11 tablas creadas")
            logger.info("• 4 planes de suscripción insertados")
            logger.info("• 1 usuario administrador creado")
            logger.info("• Sistema listo para usar")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error general: {e}")
            return False
        finally:
            self.close_connection()

def main():
    """Función principal"""
    print("🔧 CONFIGURADOR DE BASE DE DATOS - XAMPP")
    print("=" * 50)
    print("Este script creará la base de datos completa para el sistema de trading")
    print("Asegúrate de que XAMPP esté corriendo (Apache y MySQL)")
    print()
    
    # Verificar que MySQL esté disponible
    try:
        import mysql.connector
    except ImportError:
        print("❌ Error: mysql-connector-python no está instalado")
        print("Instala con: pip install mysql-connector-python")
        return
    
    # Crear base de datos
    creator = DatabaseCreator()
    success = creator.create_database_complete()
    
    if success:
        print("\n✅ ¡Configuración completada exitosamente!")
        print("\n📊 INFORMACIÓN DE CONEXIÓN:")
        print(f"• Host: {creator.host}")
        print(f"• Puerto: {creator.port}")
        print(f"• Usuario: {creator.user}")
        print(f"• Base de datos: {creator.database_name}")
        print(f"• URL de conexión: mysql://{creator.user}@{creator.host}:{creator.port}/{creator.database_name}")
        
        print("\n🔑 CREDENCIALES POR DEFECTO:")
        print("• Usuario admin: admin")
        print("• Contraseña admin: admin123")
        print("• Email admin: admin@aitraderx.com")
        
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Configura las variables de entorno en .env")
        print("2. Ejecuta las migraciones de Alembic")
        print("3. Inicia el servidor backend")
        print("4. Accede al frontend")
        
    else:
        print("\n❌ Error en la configuración")
        print("Verifica que XAMPP esté corriendo y MySQL esté disponible")

if __name__ == "__main__":
    main() 