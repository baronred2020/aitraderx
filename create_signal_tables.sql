-- Script para crear tablas de señales de trading
-- Ejecutar en la base de datos MySQL

-- Tabla para almacenar las señales de los usuarios
CREATE TABLE IF NOT EXISTS user_signals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    pair VARCHAR(10) NOT NULL,
    signal_type ENUM('buy', 'sell', 'hold') NOT NULL,
    strength ENUM('weak', 'medium', 'strong') NOT NULL,
    confidence DECIMAL(5,2) NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss DECIMAL(10,5) NULL,
    take_profit DECIMAL(10,5) NULL,
    brain_type VARCHAR(20) NOT NULL,
    style VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL,
    is_completed BOOLEAN DEFAULT FALSE,
    actual_price_at_expiry DECIMAL(10,5) NULL,
    signal_success BOOLEAN NULL,
    success_percentage DECIMAL(5,2) NULL,
    signal_quality DECIMAL(5,2) NULL,
    
    INDEX idx_user_id (user_id),
    INDEX idx_created_at (created_at),
    INDEX idx_pair (pair),
    INDEX idx_brain_type (brain_type),
    INDEX idx_style (style),
    INDEX idx_is_completed (is_completed)
);

-- Tabla para almacenar los límites de señales por usuario
CREATE TABLE IF NOT EXISTS user_signal_limits (
    user_id INT PRIMARY KEY,
    plan_type VARCHAR(20) NOT NULL,
    max_signals_per_day INT NOT NULL,
    signals_used_today INT DEFAULT 0,
    last_reset_date DATE DEFAULT CURRENT_DATE,
    
    INDEX idx_plan_type (plan_type),
    INDEX idx_last_reset_date (last_reset_date)
);

-- Insertar límites por defecto para los planes existentes
INSERT INTO user_signal_limits (user_id, plan_type, max_signals_per_day) VALUES 
(1, 'starter', 5),
(2, 'trader', 20),
(3, 'expert', 50),
(4, 'premium', 100),
(5, 'institutional', -1) -- Ilimitado
ON DUPLICATE KEY UPDATE 
    plan_type = VALUES(plan_type),
    max_signals_per_day = VALUES(max_signals_per_day);

-- Crear índices adicionales para optimizar consultas
CREATE INDEX idx_user_signals_user_created ON user_signals(user_id, created_at);
CREATE INDEX idx_user_signals_completed ON user_signals(is_completed, created_at);
CREATE INDEX idx_user_signal_limits_reset ON user_signal_limits(last_reset_date, user_id);

-- Comentarios sobre la estructura
-- user_signals: Almacena todas las señales generadas por los usuarios
-- user_signal_limits: Controla los límites diarios de señales por plan de suscripción
-- Los límites se resetean automáticamente cada día
-- Planes con límite -1 tienen señales ilimitadas 