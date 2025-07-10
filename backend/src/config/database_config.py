"""
Configuración de Base de Datos MySQL
===================================
Configuración para conectar con trading_db
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de la base de datos MySQL
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "trading_db")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# URL de conexión MySQL
DATABASE_URL = f"mysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Configuración del engine SQLAlchemy
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Cambiar a True para ver las consultas SQL
)

# Configuración de la sesión
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Obtiene una sesión de base de datos"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Crea todas las tablas en la base de datos"""
    from models.database_models import Base
    Base.metadata.create_all(bind=engine)

def drop_tables():
    """Elimina todas las tablas de la base de datos"""
    from models.database_models import Base
    Base.metadata.drop_all(bind=engine) 