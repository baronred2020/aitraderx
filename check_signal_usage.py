#!/usr/bin/env python3
"""
Script para verificar el uso actual de señales en la base de datos
"""

import mysql.connector

def check_signal_usage():
    """Verificar el uso actual de señales"""
    
    # Configuración de la base de datos (XAMPP local)
    config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'root',
        'database': 'trading_db',
        'charset': 'utf8mb4'
    }
    
    try:
        # Conectar a la base de datos
        connection = mysql.connector.connect(**config)
        cursor = connection.cursor()
        
        print("✅ Conectado a la base de datos")
        
        # Verificar uso de señales para el usuario de prueba
        test_user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
        
        query = """
        SELECT user_id, plan_type, max_signals_per_day, signals_used_today, last_reset_date
        FROM user_signal_limits 
        WHERE user_id = %s
        """
        
        cursor.execute(query, (test_user_id,))
        result = cursor.fetchone()
        
        if result:
            user_id, plan_type, max_signals, used_today, last_reset = result
            remaining = max_signals - used_today
            
            print(f"\n📊 Estado actual del usuario {user_id}:")
            print(f"   📋 Plan: {plan_type}")
            print(f"   📋 Máximo por día: {max_signals}")
            print(f"   📋 Usadas hoy: {used_today}")
            print(f"   📋 Restantes: {remaining}")
            print(f"   📋 Último reset: {last_reset}")
            print(f"   📋 Puede generar: {'Sí' if remaining > 0 else 'No'}")
        else:
            print(f"❌ No se encontraron datos para el usuario {test_user_id}")
        
        # Mostrar todos los usuarios
        print(f"\n📊 Todos los usuarios:")
        cursor.execute("""
        SELECT user_id, plan_type, max_signals_per_day, signals_used_today, last_reset_date
        FROM user_signal_limits 
        ORDER BY signals_used_today DESC
        """)
        
        all_users = cursor.fetchall()
        for user in all_users:
            user_id, plan_type, max_signals, used_today, last_reset = user
            remaining = max_signals - used_today if max_signals != -1 else -1
            
            print(f"   👤 {user_id[:8]}... | Plan: {plan_type} | Usadas: {used_today}/{max_signals} | Restantes: {remaining}")
        
    except mysql.connector.Error as err:
        print(f"❌ Error de MySQL: {err}")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("✅ Conexión cerrada")

if __name__ == "__main__":
    print("🔍 Verificando uso de señales...")
    check_signal_usage()
    print("🎉 Verificación completada") 