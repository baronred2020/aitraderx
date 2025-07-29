#!/usr/bin/env python3
"""
Script para verificar que las tablas de predicciones se crearon correctamente
"""
import mysql.connector
from mysql.connector import Error

try:
    # Conectar a la base de datos
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='trading_db'
    )
    
    if connection.is_connected():
        cursor = connection.cursor()
        
        print("🔍 Verificando tablas de predicciones...")
        
        # Verificar tabla user_predictions
        cursor.execute("SHOW TABLES LIKE 'user_predictions'")
        user_predictions_table = cursor.fetchall()
        
        if user_predictions_table:
            print("✅ Tabla 'user_predictions' existe")
            cursor.execute("DESCRIBE user_predictions")
            columns = cursor.fetchall()
            print("📋 Estructura de user_predictions:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
        else:
            print("❌ Tabla 'user_predictions' NO existe")
        
        # Verificar tabla user_prediction_limits
        cursor.execute("SHOW TABLES LIKE 'user_prediction_limits'")
        user_prediction_limits_table = cursor.fetchall()
        
        if user_prediction_limits_table:
            print("\n✅ Tabla 'user_prediction_limits' existe")
            cursor.execute("DESCRIBE user_prediction_limits")
            columns = cursor.fetchall()
            print("📋 Estructura de user_prediction_limits:")
            for col in columns:
                print(f"  - {col[0]}: {col[1]} {col[2]} {col[3]} {col[4]} {col[5]}")
        else:
            print("\n❌ Tabla 'user_prediction_limits' NO existe")
        
        # Verificar índices
        print("\n🔍 Verificando índices...")
        cursor.execute("SHOW INDEX FROM user_predictions")
        indexes = cursor.fetchall()
        print("📋 Índices de user_predictions:")
        for idx in indexes:
            print(f"  - {idx[2]}: {idx[4]}")
        
        # Verificar foreign keys
        print("\n🔍 Verificando foreign keys...")
        cursor.execute("""
            SELECT 
                CONSTRAINT_NAME,
                COLUMN_NAME,
                REFERENCED_TABLE_NAME,
                REFERENCED_COLUMN_NAME
            FROM information_schema.KEY_COLUMN_USAGE 
            WHERE TABLE_SCHEMA = 'trading_db' 
            AND TABLE_NAME = 'user_predictions'
            AND REFERENCED_TABLE_NAME IS NOT NULL
        """)
        foreign_keys = cursor.fetchall()
        print("📋 Foreign keys de user_predictions:")
        for fk in foreign_keys:
            print(f"  - {fk[0]}: {fk[1]} -> {fk[2]}.{fk[3]}")
        
        # Insertar algunos datos de prueba
        print("\n🧪 Insertando datos de prueba...")
        try:
            # Insertar límites de prueba para usuarios existentes
            cursor.execute("""
                INSERT INTO user_prediction_limits (user_id, plan_type, max_predictions_per_day) 
                SELECT user_id, 'starter', 5 FROM users 
                WHERE user_id NOT IN (SELECT user_id FROM user_prediction_limits)
            """)
            connection.commit()
            print("✅ Datos de prueba insertados correctamente")
            
            # Verificar datos insertados
            cursor.execute("SELECT COUNT(*) FROM user_prediction_limits")
            count = cursor.fetchone()[0]
            print(f"📊 Total de registros en user_prediction_limits: {count}")
            
        except Error as e:
            print(f"⚠️ Error insertando datos de prueba: {e}")
            
        cursor.close()
        connection.close()
        print("\n✅ Verificación completada")

except Error as e:
    print(f"❌ Error conectando a MySQL: {e}")
except Exception as e:
    print(f"❌ Error: {e}")