"""
Script para probar conexión MySQL
================================
Prueba diferentes configuraciones de MySQL para encontrar la correcta
"""

import os
import sys
import mysql.connector
from mysql.connector import Error
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_mysql_connection():
    """Prueba diferentes configuraciones de MySQL"""
    
    # Configuraciones a probar
    configs_to_try = [
        {"user": "root", "password": ""},
        {"user": "root", "password": "root"},
        {"user": "root", "password": "password"},
        {"user": "root", "password": "admin"},
        {"user": "root", "password": "123456"},
        {"user": "root", "password": "mysql"},
    ]
    
    for i, config in enumerate(configs_to_try, 1):
        try:
            logger.info(f"Probando configuración {i}: usuario={config['user']}, contraseña={'(vacía)' if config['password'] == '' else config['password']}")
            
            connection = mysql.connector.connect(
                host="localhost",
                port=3306,
                user=config["user"],
                password=config["password"]
            )
            
            if connection.is_connected():
                logger.info(f"✅ CONEXIÓN EXITOSA con configuración {i}")
                
                # Verificar si existe la base de datos
                cursor = connection.cursor()
                cursor.execute("SHOW DATABASES LIKE 'trading_db'")
                result = cursor.fetchone()
                
                if result:
                    logger.info("✅ Base de datos 'trading_db' existe")
                else:
                    logger.info("⚠️ Base de datos 'trading_db' no existe, creando...")
                    cursor.execute("CREATE DATABASE trading_db")
                    logger.info("✅ Base de datos 'trading_db' creada")
                
                cursor.close()
                connection.close()
                
                # Actualizar el archivo .env con la configuración correcta
                update_env_file(config["password"])
                return True
                
        except Error as e:
            logger.info(f"❌ Error con configuración {i}: {e}")
            continue
    
    logger.error("❌ No se pudo conectar con ninguna configuración")
    return False

def update_env_file(password):
    """Actualiza el archivo .env con la contraseña correcta"""
    try:
        # Leer el archivo env_fixed.txt
        with open('../env_fixed.txt', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Actualizar la contraseña
        content = content.replace('DB_PASSWORD=', f'DB_PASSWORD={password}')
        content = content.replace('DATABASE_URL=mysql://root@localhost:3306/trading_db', 
                               f'DATABASE_URL=mysql://root:{password}@localhost:3306/trading_db')
        
        # Escribir el archivo .env
        with open('../.env', 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✅ Archivo .env actualizado con contraseña: {password}")
        
    except Exception as e:
        logger.error(f"❌ Error actualizando archivo .env: {e}")

def test_sqlalchemy_connection():
    """Prueba la conexión usando SQLAlchemy"""
    try:
        import env_config
        from config.database_config import engine
        
        # Probar la conexión
        with engine.connect() as connection:
            from sqlalchemy import text
            result = connection.execute(text("SELECT 1"))
            logger.info("✅ Conexión SQLAlchemy exitosa")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error con SQLAlchemy: {e}")
        return False

if __name__ == "__main__":
    logger.info("🔍 Probando conexiones MySQL...")
    
    if test_mysql_connection():
        logger.info("📊 Probando conexión SQLAlchemy...")
        if test_sqlalchemy_connection():
            logger.info("✅ Configuración completada exitosamente")
        else:
            logger.error("❌ Error con SQLAlchemy")
    else:
        logger.error("❌ No se pudo configurar la base de datos")
        sys.exit(1) 