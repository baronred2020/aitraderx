-- Script para verificar que las tablas de señales se crearon correctamente
-- Ejecutar después de aplicar las migraciones

USE trading_db;

-- Verificar que las tablas existen
SHOW TABLES LIKE 'user_signals';
SHOW TABLES LIKE 'user_signal_limits';

-- Verificar estructura de user_signals
DESCRIBE user_signals;

-- Verificar estructura de user_signal_limits
DESCRIBE user_signal_limits;

-- Verificar índices de user_signals
SHOW INDEX FROM user_signals;

-- Verificar índices de user_signal_limits
SHOW INDEX FROM user_signal_limits;

-- Verificar que hay límites insertados
SELECT 
    usl.user_id,
    u.email,
    usl.plan_type,
    usl.max_signals_per_day,
    usl.signals_used_today,
    usl.last_reset_date
FROM user_signal_limits usl
LEFT JOIN users u ON usl.user_id = u.id
ORDER BY usl.user_id;

-- Contar registros en cada tabla
SELECT 'user_signals' as tabla, COUNT(*) as total FROM user_signals
UNION ALL
SELECT 'user_signal_limits' as tabla, COUNT(*) as total FROM user_signal_limits;

-- Verificar foreign keys
SELECT 
    TABLE_NAME,
    COLUMN_NAME,
    CONSTRAINT_NAME,
    REFERENCED_TABLE_NAME,
    REFERENCED_COLUMN_NAME
FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
WHERE TABLE_SCHEMA = 'trading_db' 
AND TABLE_NAME IN ('user_signals', 'user_signal_limits')
AND REFERENCED_TABLE_NAME IS NOT NULL; 