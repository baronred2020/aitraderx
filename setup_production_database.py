#!/usr/bin/env python3
"""
Script para configurar la base de datos de producción con datos reales
"""
import mysql.connector
import sys
import os
from datetime import datetime, timedelta
import random

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def setup_production_database():
    """Configurar la base de datos de producción con datos reales"""
    
    print("🔧 Configurando base de datos de producción...")
    
    try:
        # Configuración de la base de datos
        config = {
            'host': 'localhost',
            'user': 'root',
            'password': '',  # Ajustar según tu configuración
            'database': 'trading_db',
            'port': 3306
        }
        
        # Intentar conectar
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            print("✅ Conexión exitosa a la base de datos")
            
            cursor = connection.cursor()
            
            # 1. Verificar que la tabla predictions existe
            cursor.execute("SHOW TABLES LIKE 'predictions'")
            if not cursor.fetchone():
                print("❌ Tabla predictions NO existe. Creando...")
                
                # Crear tabla predictions si no existe
                create_table_query = """
                CREATE TABLE predictions (
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
                )
                """
                cursor.execute(create_table_query)
                print("✅ Tabla predictions creada")
            else:
                print("✅ Tabla predictions existe")
            
            # 2. Verificar que la tabla users existe
            cursor.execute("SHOW TABLES LIKE 'users'")
            if not cursor.fetchone():
                print("❌ Tabla users NO existe. Creando...")
                
                # Crear tabla users si no existe
                create_users_query = """
                CREATE TABLE users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id VARCHAR(36) UNIQUE NOT NULL,
                    username VARCHAR(50) NOT NULL,
                    email VARCHAR(100) NOT NULL,
                    plan_type VARCHAR(20) DEFAULT 'starter',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_user_id (user_id)
                )
                """
                cursor.execute(create_users_query)
                print("✅ Tabla users creada")
                
                # Insertar usuario de prueba
                insert_user_query = """
                INSERT INTO users (user_id, username, email, plan_type) 
                VALUES ('test-user-001', 'test_user', 'test@example.com', 'starter')
                """
                cursor.execute(insert_user_query)
                print("✅ Usuario de prueba creado")
            else:
                print("✅ Tabla users existe")
            
            # 3. Insertar datos de ejemplo reales en predictions
            print("📊 Insertando datos de ejemplo reales...")
            
            # Limpiar datos existentes
            cursor.execute("DELETE FROM predictions WHERE user_id = 1")
            
            # Insertar predicciones de ejemplo reales
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
            directions = ['up', 'down']
            brain_types = ['brain_max', 'brain_ultra', 'brain_predictor']
            
            for i in range(10):  # 10 predicciones de ejemplo
                pair = random.choice(pairs)
                direction = random.choice(directions)
                brain_type = random.choice(brain_types)
                
                # Precios realistas
                if pair == 'EURUSD':
                    current_price = 1.0850 + random.uniform(-0.01, 0.01)
                elif pair == 'GBPUSD':
                    current_price = 1.2650 + random.uniform(-0.01, 0.01)
                elif pair == 'USDJPY':
                    current_price = 148.50 + random.uniform(-0.5, 0.5)
                elif pair == 'AUDUSD':
                    current_price = 0.6650 + random.uniform(-0.005, 0.005)
                else:  # USDCAD
                    current_price = 1.3550 + random.uniform(-0.005, 0.005)
                
                # Calcular target price
                if direction == 'up':
                    target_price = current_price * (1 + random.uniform(0.001, 0.003))
                else:
                    target_price = current_price * (1 - random.uniform(0.001, 0.003))
                
                confidence = random.uniform(65, 95)
                created_at = datetime.now() - timedelta(hours=i+1)
                expires_at = created_at + timedelta(minutes=15)
                
                # Determinar si la predicción fue exitosa
                is_completed = random.choice([True, False])
                prediction_success = None
                success_percentage = None
                actual_price_at_expiry = None
                
                if is_completed:
                    # Simular resultado real
                    if random.random() > 0.3:  # 70% éxito
                        prediction_success = True
                        success_percentage = random.uniform(80, 100)
                        # Precio real cercano al target
                        if direction == 'up':
                            actual_price_at_expiry = target_price * random.uniform(0.998, 1.002)
                        else:
                            actual_price_at_expiry = target_price * random.uniform(0.998, 1.002)
                    else:
                        prediction_success = False
                        success_percentage = random.uniform(20, 60)
                        # Precio real lejos del target
                        if direction == 'up':
                            actual_price_at_expiry = current_price * random.uniform(0.995, 0.999)
                        else:
                            actual_price_at_expiry = current_price * random.uniform(1.001, 1.005)
                
                insert_query = """
                INSERT INTO predictions 
                (user_id, pair, direction, current_price, target_price, confidence,
                 timeframe, reasoning, brain_type, created_at, expires_at,
                 is_completed, actual_price_at_expiry, prediction_success, success_percentage)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                reasoning = f"Análisis técnico para {pair} usando {brain_type} - {direction.upper()}"
                
                cursor.execute(insert_query, (
                    1,  # user_id
                    pair,
                    direction,
                    current_price,
                    target_price,
                    confidence,
                    "15M",
                    reasoning,
                    brain_type,
                    created_at,
                    expires_at,
                    is_completed,
                    actual_price_at_expiry,
                    prediction_success,
                    success_percentage
                ))
            
            connection.commit()
            print("✅ 10 predicciones de ejemplo insertadas")
            
            # 4. Verificar datos
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = 1")
            count = cursor.fetchone()[0]
            print(f"📊 Total de predicciones en BD: {count}")
            
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE user_id = 1 AND DATE(created_at) = CURDATE()")
            today_count = cursor.fetchone()[0]
            print(f"📊 Predicciones de hoy: {today_count}")
            
            cursor.close()
            connection.close()
            
            print("\n✅ Base de datos de producción configurada correctamente")
            
        else:
            print("❌ No se pudo conectar a la base de datos")
            
    except mysql.connector.Error as e:
        print(f"❌ Error de MySQL: {e}")
    except Exception as e:
        print(f"❌ Error general: {e}")

if __name__ == "__main__":
    setup_production_database()