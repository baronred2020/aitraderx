#!/usr/bin/env python3
"""
Script para probar la conexión a la base de datos y verificar tabla predictions
"""
import mysql.connector
import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def test_database_connection():
    """Probar la conexión a la base de datos y verificar tabla predictions"""
    
    print("🔍 Probando conexión a la base de datos...")
    
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
            
            # Verificar que las tablas existen
            cursor = connection.cursor()
            
            # Verificar tabla predictions
            cursor.execute("SHOW TABLES LIKE 'predictions'")
            if cursor.fetchone():
                print("✅ Tabla predictions existe")
                
                # Mostrar estructura de la tabla
                cursor.execute("DESCRIBE predictions")
                columns = cursor.fetchall()
                print("📋 Estructura de la tabla predictions:")
                for column in columns:
                    print(f"   - {column[0]}: {column[1]}")
                
                # Contar registros
                cursor.execute("SELECT COUNT(*) FROM predictions")
                count = cursor.fetchone()[0]
                print(f"📊 Registros en predictions: {count}")
                
                # Mostrar algunos registros de ejemplo
                if count > 0:
                    cursor.execute("SELECT * FROM predictions LIMIT 3")
                    sample_records = cursor.fetchall()
                    print("📄 Registros de ejemplo:")
                    for i, record in enumerate(sample_records, 1):
                        print(f"   {i}. {record}")
                
            else:
                print("❌ Tabla predictions NO existe")
            
            # Verificar tabla users
            cursor.execute("SHOW TABLES LIKE 'users'")
            if cursor.fetchone():
                print("✅ Tabla users existe")
                
                # Contar registros
                cursor.execute("SELECT COUNT(*) FROM users")
                count = cursor.fetchone()[0]
                print(f"📊 Registros en users: {count}")
            else:
                print("❌ Tabla users NO existe")
            
            cursor.close()
            connection.close()
            
        else:
            print("❌ No se pudo conectar a la base de datos")
            
    except mysql.connector.Error as e:
        print(f"❌ Error de MySQL: {e}")
    except Exception as e:
        print(f"❌ Error general: {e}")

if __name__ == "__main__":
    test_database_connection()