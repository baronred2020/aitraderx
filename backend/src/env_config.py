"""
Configuración de Variables de Entorno
====================================
Carga las variables desde el archivo .env del directorio padre
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables desde el archivo .env del directorio padre (backend/)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configuración de Base de Datos MySQL
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_NAME", "trading_db")
os.environ.setdefault("DB_USER", "root")
os.environ.setdefault("DB_PASSWORD", "")

# Configuración de JWT
os.environ.setdefault("SECRET_KEY", "dev-secret-key-change-in-production")
os.environ.setdefault("ALGORITHM", "HS256")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "60")

# Configuración del servidor
os.environ.setdefault("HOST", "0.0.0.0")
os.environ.setdefault("PORT", "8000")
os.environ.setdefault("DEBUG", "true")

# Configuración de logging
os.environ.setdefault("LOG_LEVEL", "info")
os.environ.setdefault("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s") 