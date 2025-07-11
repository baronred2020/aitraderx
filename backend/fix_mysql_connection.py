#!/usr/bin/env python3
"""
Script para diagnosticar y arreglar problemas de conexión MySQL
"""

import mysql.connector
import time
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

def test_mysql_connection():
    """Probar la conexión MySQL y arreglar problemas comunes"""
    print("🔍 Diagnosticando conexión MySQL...")
    
    # Configuración de conexión
    config = {
        'host': 'localhost',
        'port': 3306,
        'user': 'root',
        'password': '',  # XAMPP por defecto no tiene password
        'database': 'aitraderx_db',
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_unicode_ci',
        'autocommit': True,
        'raise_on_warnings': True
    }
    
    try:
        print("📡 Intentando conectar a MySQL...")
        connection = mysql.connector.connect(**config)
        
        if connection.is_connected():
            print("✅ Conexión MySQL exitosa!")
            
            # Verificar la base de datos
            cursor = connection.cursor()
            
            # Rollback cualquier transacción pendiente
            print("🔄 Limpiando transacciones pendientes...")
            connection.rollback()
            
            # Verificar tablas
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"📊 Tablas encontradas: {len(tables)}")
            
            # Verificar tabla users
            cursor.execute("SELECT COUNT(*) FROM users")
            user_count = cursor.fetchone()[0]
            print(f"👥 Usuarios en la base de datos: {user_count}")
            
            # Cerrar cursor y conexión
            cursor.close()
            connection.close()
            print("✅ Conexión cerrada correctamente")
            return True
            
    except mysql.connector.Error as e:
        print(f"❌ Error de MySQL: {e}")
        
        # Intentar arreglos comunes
        if "Can't connect" in str(e):
            print("🔧 Verificar que XAMPP MySQL esté ejecutándose...")
            print("   - Abrir XAMPP Control Panel")
            print("   - Iniciar Apache y MySQL")
            
        elif "Access denied" in str(e):
            print("🔧 Problema de autenticación:")
            print("   - Verificar usuario y contraseña")
            print("   - En XAMPP, el usuario por defecto es 'root' sin contraseña")
            
        elif "Unknown database" in str(e):
            print("🔧 Base de datos no encontrada:")
            print("   - Ejecutar script de creación de base de datos")
            
        return False
        
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def restart_database_connection():
    """Reiniciar las conexiones de la base de datos"""
    print("🔄 Reiniciando conexiones de la base de datos...")
    
    try:
        # Importar y reinicializar la configuración de la base de datos
        from config.database_config import DatabaseConfig
        
        # Limpiar el pool de conexiones si existe
        DatabaseConfig.cleanup_connections()
        print("✅ Pool de conexiones limpiado")
        
        # Probar nueva conexión
        db = DatabaseConfig()
        engine = db.get_engine()
        
        with engine.connect() as conn:
            result = conn.execute("SELECT 1 as test")
            print("✅ Nueva conexión establecida correctamente")
            
        return True
        
    except Exception as e:
        print(f"❌ Error al reiniciar conexiones: {e}")
        return False

def main():
    """Función principal"""
    print("🚀 Iniciando diagnóstico y reparación de MySQL...")
    print("=" * 50)
    
    # Paso 1: Probar conexión básica
    if test_mysql_connection():
        print("\n✅ Conexión MySQL funcionando correctamente")
        
        # Paso 2: Reiniciar conexiones de la aplicación
        if restart_database_connection():
            print("\n✅ Todas las conexiones han sido reparadas")
        else:
            print("\n⚠️  Conexión básica OK, pero hay problemas con la aplicación")
    else:
        print("\n❌ Hay problemas graves con MySQL")
        print("\n🔧 Soluciones sugeridas:")
        print("   1. Verificar que XAMPP esté ejecutándose")
        print("   2. Reiniciar XAMPP")
        print("   3. Verificar puerto 3306 libre")
        print("   4. Ejecutar script de creación de base de datos")
    
    print("\n" + "=" * 50)
    print("🏁 Diagnóstico completado")

if __name__ == "__main__":
    main() 