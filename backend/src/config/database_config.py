"""
Database Configuration
"""
import os
import mysql.connector
from mysql.connector import pooling
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Configuración de la base de datos"""
    
    def __init__(self):
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = int(os.getenv('DB_PORT', 3306))
        self.database = os.getenv('DB_NAME', 'trading_db')
        self.user = os.getenv('DB_USER', 'root')
        self.password = os.getenv('DB_PASSWORD', 'root')
        self.pool_size = int(os.getenv('DATABASE_POOL_SIZE', 5))
        self.max_overflow = int(os.getenv('DATABASE_MAX_OVERFLOW', 10))
        
        # Configuración del pool de conexiones
        self.pool_config = {
            'host': self.host,
            'port': self.port,
            'database': self.database,
            'user': self.user,
            'password': self.password,
            'pool_name': 'trading_pool',
            'pool_size': self.pool_size,
            'pool_reset_session': True,
            'autocommit': True,
            'charset': 'utf8mb4',
            'collation': 'utf8mb4_unicode_ci'
        }
        
        self.connection_pool = None
    
    def create_connection_pool(self):
        """Crear el pool de conexiones"""
        try:
            self.connection_pool = pooling.MySQLConnectionPool(**self.pool_config)
            logger.info("Pool de conexiones creado exitosamente")
            return True
        except Exception as e:
            logger.error(f"Error creando pool de conexiones: {e}")
            return False
    
    @contextmanager
    def get_connection(self):
        """Obtener una conexión del pool"""
        connection = None
        try:
            if self.connection_pool:
                connection = self.connection_pool.get_connection()
                yield connection
            else:
                # Fallback: conexión directa
                connection = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    autocommit=True,
                    charset='utf8mb4',
                    collation='utf8mb4_unicode_ci'
                )
                yield connection
        except Exception as e:
            logger.error(f"Error obteniendo conexión: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def test_connection(self):
        """Probar la conexión a la base de datos"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                cursor.close()
                logger.info("Conexión a la base de datos exitosa")
                return True
        except Exception as e:
            logger.error(f"Error probando conexión: {e}")
            return False
    
    def create_tables_if_not_exist(self):
        """Crear las tablas necesarias si no existen"""
        try:
            with self.get_connection() as connection:
                cursor = connection.cursor()
                
                # Crear tabla predictions si no existe
                create_predictions_table = """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    pair VARCHAR(10) NOT NULL,
                    direction ENUM('up', 'down', 'sideways') NOT NULL,
                    current_price DECIMAL(10,5) NOT NULL,
                    target_price DECIMAL(10,5) NOT NULL,
                    confidence DECIMAL(5,2) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    reasoning TEXT,
                    brain_type VARCHAR(50) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    is_completed BOOLEAN DEFAULT FALSE,
                    actual_price_at_expiry DECIMAL(10,5),
                    prediction_success BOOLEAN,
                    success_percentage DECIMAL(5,2),
                    INDEX idx_user_id (user_id),
                    INDEX idx_created_at (created_at),
                    INDEX idx_pair (pair)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                """
                cursor.execute(create_predictions_table)
                
                # Verificar si la tabla users existe
                cursor.execute("SHOW TABLES LIKE 'users'")
                if cursor.fetchone():
                    # La tabla users existe, verificar si tiene plan_type
                    cursor.execute("SHOW COLUMNS FROM users LIKE 'plan_type'")
                    if not cursor.fetchone():
                        # Agregar columna plan_type si no existe
                        cursor.execute("ALTER TABLE users ADD COLUMN plan_type VARCHAR(20) DEFAULT 'starter'")
                        logger.info("Columna plan_type agregada a la tabla users")
                else:
                    # Crear tabla users si no existe
                    create_users_table = """
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        user_id VARCHAR(36) UNIQUE NOT NULL,
                        username VARCHAR(50) NOT NULL,
                        email VARCHAR(100) NOT NULL,
                        plan_type VARCHAR(20) DEFAULT 'starter',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        INDEX idx_user_id (user_id)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
                    """
                    cursor.execute(create_users_table)
                
                # Insertar usuario de prueba si no existe
                insert_test_user = """
                INSERT IGNORE INTO users (user_id, username, email, plan_type) 
                VALUES ('test-user-001', 'test_user', 'test@example.com', 'starter')
                """
                cursor.execute(insert_test_user)
                
                connection.commit()
                cursor.close()
                logger.info("Tablas creadas/verificadas exitosamente")
                return True
                
        except Exception as e:
            logger.error(f"Error creando tablas: {e}")
            return False

    def initialize_db(self):
        """Inicializar la base de datos - crear pool de conexiones y verificar conexión"""
        try:
            # Crear pool de conexiones
            if not self.create_connection_pool():
                logger.error("No se pudo crear el pool de conexiones")
                return False
            
            # Probar conexión
            if not self.test_connection():
                logger.error("No se pudo conectar a la base de datos")
                return False
            
            # Crear tablas si no existen
            if not self.create_tables_if_not_exist():
                logger.error("No se pudieron crear/verificar las tablas")
                return False
            
            logger.info("Base de datos inicializada exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando base de datos: {e}")
            return False

# Instancia global de la configuración
db_config = DatabaseConfig() 