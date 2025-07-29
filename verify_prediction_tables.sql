-- Script para verificar las tablas de predicciones
-- Ejecutar este script en phpMyAdmin

USE trading_db;

-- Verificar si las tablas existen
SELECT 
    TABLE_NAME,
    TABLE_ROWS,
    CREATE_TIME,
    UPDATE_TIME
FROM information_schema.TABLES 
WHERE TABLE_SCHEMA = 'trading_db' 
AND TABLE_NAME IN ('user_predictions', 'user_prediction_limits');

-- Mostrar estructura de user_predictions
DESCRIBE user_predictions;

-- Mostrar estructura de user_prediction_limits
DESCRIBE user_prediction_limits;

-- Verificar índices de user_predictions
SHOW INDEX FROM user_predictions;

-- Verificar foreign keys
SELECT 
    CONSTRAINT_NAME,
    COLUMN_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM information_schema.KEY_COLUMN_USAGE 
WHERE TABLE_SCHEMA = 'trading_db' 
AND TABLE_NAME = 'user_predictions'
AND REFERENCED_TABLE_NAME IS NOT NULL;

-- Insertar datos de prueba para usuarios existentes
INSERT INTO user_prediction_limits (user_id, plan_type, max_predictions_per_day) 
SELECT user_id, 'starter', 5 FROM users 
WHERE user_id NOT IN (SELECT user_id FROM user_prediction_limits);

-- Verificar datos insertados
SELECT 
    COUNT(*) as total_limits,
    plan_type,
    max_predictions_per_day
FROM user_prediction_limits 
GROUP BY plan_type, max_predictions_per_day;

-- Mostrar algunos registros de ejemplo
SELECT * FROM user_prediction_limits LIMIT 5;