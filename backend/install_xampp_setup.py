"""
Script de Instalación para XAMPP
================================
Script para instalar dependencias necesarias para trabajar con XAMPP local
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Instalar un paquete usando pip"""
    try:
        print(f"📦 Instalando {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ {package} instalado correctamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando {package}: {e}")
        return False

def main():
    """Función principal de instalación"""
    print("🔧 INSTALADOR DE DEPENDENCIAS - XAMPP")
    print("=" * 50)
    print("Este script instalará las dependencias necesarias para trabajar con XAMPP")
    print()
    
    # Lista de paquetes necesarios para XAMPP
    packages = [
        "mysql-connector-python",
        "sqlalchemy",
        "alembic",
        "fastapi",
        "uvicorn[standard]",
        "pandas",
        "numpy",
        "scikit-learn",
        "yfinance",
        "ta",
        "python-dotenv",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "python-multipart",
        "pydantic",
        "websockets",
        "requests",
        "aiohttp",
        "redis",
        "celery",
        "gymnasium",
        "stable-baselines3",
        "tensorflow",
        "torch",
        "torchvision",
        "prometheus-client",
        "structlog"
    ]
    
    print("📋 Paquetes a instalar:")
    for package in packages:
        print(f"  • {package}")
    print()
    
    # Confirmar instalación
    response = input("¿Deseas continuar con la instalación? (y/n): ")
    if response.lower() != 'y':
        print("❌ Instalación cancelada")
        return
    
    print("\n🚀 Iniciando instalación...")
    print("-" * 30)
    
    success_count = 0
    total_count = len(packages)
    
    for package in packages:
        if install_package(package):
            success_count += 1
        print()
    
    print("📊 RESUMEN DE INSTALACIÓN")
    print("=" * 30)
    print(f"✅ Paquetes instalados: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 ¡Todas las dependencias instaladas correctamente!")
        print("\n📋 PRÓXIMOS PASOS:")
        print("1. Asegúrate de que XAMPP esté corriendo (Apache y MySQL)")
        print("2. Ejecuta: python src/create_database_xampp.py")
        print("3. Copia env_xampp.txt a .env")
        print("4. Inicia el servidor: python src/main.py")
    else:
        print("⚠️  Algunos paquetes no se pudieron instalar")
        print("Revisa los errores e intenta instalar manualmente los paquetes faltantes")

if __name__ == "__main__":
    main() 