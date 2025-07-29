-- Script para crear las tablas de predicciones manualmente
-- Ejecutar este script en MySQL para crear las tablas necesarias

USE trading_db;

-- Crear tabla de predicciones de usuarios
CREATE TABLE IF NOT EXISTS user_predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    pair VARCHAR(10) NOT NULL,
    direction ENUM('up', 'down', 'sideways') NOT NULL,
    current_price DECIMAL(10, 5) NOT NULL,
    target_price DECIMAL(10, 5) NOT NULL,
    confidence DECIMAL(5, 2) NOT NULL,
    timeframe VARCHAR(10) DEFAULT '15M',
    reasoning TEXT,
    brain_type VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    is_completed BOOLEAN DEFAULT FALSE,
    actual_price_at_expiry DECIMAL(10, 5) NULL,
    prediction_success BOOLEAN NULL,
    success_percentage DECIMAL(5, 2) NULL
);

-- Crear tabla de límites de predicciones por usuario
CREATE TABLE IF NOT EXISTS user_prediction_limits (
    user_id INT PRIMARY KEY,
    plan_type VARCHAR(20) NOT NULL,
    max_predictions_per_day INT NOT NULL,
    predictions_used_today INT DEFAULT 0,
    last_reset_date DATE DEFAULT CURRENT_DATE
);

-- Crear índices para mejor rendimiento
CREATE INDEX idx_user_predictions_user_id ON user_predictions(user_id);
CREATE INDEX idx_user_predictions_created_at ON user_predictions(created_at);
CREATE INDEX idx_user_predictions_is_completed ON user_predictions(is_completed);

-- Insertar límites por defecto para usuarios existentes (plan starter)
INSERT INTO user_prediction_limits (user_id, plan_type, max_predictions_per_day)
SELECT id, 'starter', 5 FROM users WHERE id NOT IN (SELECT user_id FROM user_prediction_limits); 