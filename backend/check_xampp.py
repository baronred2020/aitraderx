"""
Verificador de Estado de XAMPP
==============================
Script para verificar si XAMPP está corriendo y proporcionar instrucciones
"""

import socket
import subprocess
import os
import sys
from pathlib import Path

def check_port(host, port):
    """Verificar si un puerto está abierto"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except:
        return False

def check_xampp_status():
    """Verificar el estado de XAMPP"""
    print("🔍 VERIFICANDO ESTADO DE XAMPP")
    print("=" * 40)
    
    # Verificar puertos comunes de XAMPP
    ports_to_check = {
        3306: "MySQL",
        80: "Apache",
        443: "Apache (HTTPS)",
        8080: "Apache (Alternativo)"
    }
    
    all_services_running = True
    
    for port, service in ports_to_check.items():
        if check_port("localhost", port):
            print(f"✅ {service} está corriendo en puerto {port}")
        else:
            print(f"❌ {service} NO está corriendo en puerto {port}")
            all_services_running = False
    
    print()
    
    if all_services_running:
        print("🎉 ¡XAMPP está funcionando correctamente!")
        return True
    else:
        print("⚠️  XAMPP no está corriendo completamente")
        print("\n📋 INSTRUCCIONES PARA INICIAR XAMPP:")
        print("1. Abre XAMPP Control Panel")
        print("2. Haz clic en 'Start' para Apache")
        print("3. Haz clic en 'Start' para MySQL")
        print("4. Verifica que ambos servicios estén en verde")
        print("5. Ejecuta este script nuevamente")
        
        # Intentar encontrar XAMPP
        xampp_paths = [
            "C:\\xampp\\xampp-control.exe",
            "C:\\xampp\\xampp_start.exe",
            "C:\\Program Files\\xampp\\xampp-control.exe",
            "C:\\Program Files (x86)\\xampp\\xampp-control.exe"
        ]
        
        xampp_found = False
        for path in xampp_paths:
            if os.path.exists(path):
                print(f"\n📍 XAMPP encontrado en: {path}")
                xampp_found = True
                break
        
        if not xampp_found:
            print("\n❌ XAMPP no encontrado en las ubicaciones comunes")
            print("Asegúrate de que XAMPP esté instalado")
        
        return False

def test_mysql_connection():
    """Probar conexión a MySQL"""
    print("\n🔌 PROBANDO CONEXIÓN A MYSQL")
    print("-" * 30)
    
    try:
        import mysql.connector
        from mysql.connector import Error
        
        connection = mysql.connector.connect(
            host="localhost",
            port=3306,
            user="root",
            password=""
        )
        
        if connection.is_connected():
            print("✅ Conexión a MySQL exitosa")
            cursor = connection.cursor()
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()
            print(f"📊 Versión de MySQL: {version[0]}")
            connection.close()
            return True
        else:
            print("❌ No se pudo conectar a MySQL")
            return False
            
    except ImportError:
        print("❌ mysql-connector-python no está instalado")
        print("Instala con: pip install mysql-connector-python")
        return False
    except Error as e:
        print(f"❌ Error conectando a MySQL: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 VERIFICADOR DE XAMPP")
    print("=" * 50)
    print("Este script verifica si XAMPP está corriendo correctamente")
    print()
    
    # Verificar estado de XAMPP
    xampp_ok = check_xampp_status()
    
    if xampp_ok:
        # Probar conexión a MySQL
        mysql_ok = test_mysql_connection()
        
        if mysql_ok:
            print("\n🎉 ¡Todo está listo!")
            print("Puedes ejecutar: python src/create_database_xampp.py")
        else:
            print("\n❌ MySQL no está disponible")
            print("Verifica que MySQL esté iniciado en XAMPP")
    else:
        print("\n❌ XAMPP no está corriendo")
        print("Inicia XAMPP y ejecuta este script nuevamente")

if __name__ == "__main__":
    main() 