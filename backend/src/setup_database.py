"""
Script de Configuración de Base de Datos
========================================
Script para configurar MySQL y ejecutar migraciones del sistema de suscripciones
"""

import os
import sys
from pathlib import Path
import logging
from sqlalchemy import create_engine, text
from alembic import command
from alembic.config import Config
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Configura la base de datos MySQL"""
    
    print("🚀 Configurando Base de Datos MySQL")
    print("=" * 50)
    
    # Obtener configuración de base de datos
    database_url = os.getenv(
        'DATABASE_URL', 
        'mysql://trader:password123@localhost:3306/trading_db'
    )
    
    print(f"📊 URL de Base de Datos: {database_url}")
    
    try:
        # Crear engine para verificar conexión
        engine = create_engine(database_url, pool_pre_ping=True)
        
        # Verificar conexión
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Conexión a MySQL establecida")
        
        # Crear base de datos si no existe
        create_database_if_not_exists(database_url)
        
        # Ejecutar migraciones
        run_migrations()
        
        # Verificar tablas creadas
        verify_tables(engine)
        
        print("✅ Base de datos configurada exitosamente")
        
    except Exception as e:
        print(f"❌ Error configurando base de datos: {e}")
        sys.exit(1)

def create_database_if_not_exists(database_url: str):
    """Crea la base de datos si no existe"""
    try:
        # Extraer información de la URL
        if database_url.startswith('mysql://'):
            # mysql://user:pass@host:port/db
            parts = database_url.replace('mysql://', '').split('/')
            if len(parts) == 2:
                connection_info = parts[0]
                database_name = parts[1]
                
                # Crear URL sin nombre de base de datos
                base_url = f"mysql://{connection_info}"
                
                # Conectar sin especificar base de datos
                engine = create_engine(base_url)
                
                with engine.connect() as conn:
                    # Crear base de datos si no existe
                    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {database_name}"))
                    conn.execute(text(f"USE {database_name}"))
                    print(f"✅ Base de datos '{database_name}' creada/verificada")
        
    except Exception as e:
        print(f"⚠️  No se pudo crear la base de datos: {e}")
        print("   Asegúrate de que la base de datos existe manualmente")

def run_migrations():
    """Ejecuta las migraciones de Alembic"""
    try:
        print("\n📋 Ejecutando migraciones...")
        
        # Configurar Alembic
        alembic_cfg = Config()
        alembic_cfg.set_main_option("script_location", "database/migrations")
        alembic_cfg.set_main_option("sqlalchemy.url", os.getenv('DATABASE_URL', 'mysql://trader:password123@localhost:3306/trading_db'))
        
        # Ejecutar migraciones
        command.upgrade(alembic_cfg, "head")
        print("✅ Migraciones ejecutadas exitosamente")
        
    except Exception as e:
        print(f"❌ Error ejecutando migraciones: {e}")
        print("   Asegúrate de que Alembic esté configurado correctamente")

def verify_tables(engine):
    """Verifica que las tablas se crearon correctamente"""
    try:
        print("\n🔍 Verificando tablas creadas...")
        
        expected_tables = [
            'subscription_plans',
            'user_subscriptions', 
            'usage_metrics',
            'subscription_upgrades',
            'subscription_payments',
            'subscription_audit'
        ]
        
        with engine.connect() as conn:
            # Obtener tablas existentes
            result = conn.execute(text("SHOW TABLES"))
            existing_tables = [row[0] for row in result.fetchall()]
            
            # Verificar tablas esperadas
            for table in expected_tables:
                if table in existing_tables:
                    print(f"  ✅ Tabla '{table}' creada")
                else:
                    print(f"  ❌ Tabla '{table}' NO encontrada")
            
            # Verificar datos en subscription_plans
            result = conn.execute(text("SELECT COUNT(*) FROM subscription_plans"))
            plan_count = result.fetchone()[0]
            print(f"  📊 Planes en base de datos: {plan_count}")
            
            if plan_count == 4:
                print("  ✅ Todos los planes por defecto insertados")
            else:
                print(f"  ⚠️  Solo {plan_count}/4 planes encontrados")
        
    except Exception as e:
        print(f"❌ Error verificando tablas: {e}")

def test_database_connection():
    """Prueba la conexión a la base de datos"""
    try:
        print("\n🧪 Probando conexión a base de datos...")
        
        database_url = os.getenv(
            'DATABASE_URL', 
            'mysql://trader:password123@localhost:3306/trading_db'
        )
        
        engine = create_engine(database_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # Probar consulta simple
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row[0] == 1:
                print("✅ Conexión a base de datos exitosa")
                
                # Probar consulta a tablas de suscripciones
                try:
                    result = conn.execute(text("SELECT COUNT(*) FROM subscription_plans"))
                    count = result.fetchone()[0]
                    print(f"✅ Tabla subscription_plans accesible ({count} registros)")
                except Exception as e:
                    print(f"❌ Error accediendo a subscription_plans: {e}")
            else:
                print("❌ Error en conexión a base de datos")
                
    except Exception as e:
        print(f"❌ Error probando conexión: {e}")

def show_database_info():
    """Muestra información de la base de datos"""
    try:
        print("\n📊 Información de Base de Datos")
        print("-" * 30)
        
        database_url = os.getenv(
            'DATABASE_URL', 
            'mysql://trader:password123@localhost:3306/trading_db'
        )
        
        engine = create_engine(database_url, pool_pre_ping=True)
        
        with engine.connect() as conn:
            # Información de tablas
            result = conn.execute(text("SHOW TABLES"))
            tables = [row[0] for row in result.fetchall()]
            
            print(f"📋 Tablas encontradas: {len(tables)}")
            for table in tables:
                print(f"  - {table}")
            
            # Conteo de registros
            if 'subscription_plans' in tables:
                result = conn.execute(text("SELECT plan_type, COUNT(*) FROM subscription_plans GROUP BY plan_type"))
                plans = result.fetchall()
                print(f"\n📈 Planes disponibles:")
                for plan_type, count in plans:
                    print(f"  - {plan_type}: {count}")
            
            if 'user_subscriptions' in tables:
                result = conn.execute(text("SELECT status, COUNT(*) FROM user_subscriptions GROUP BY status"))
                subscriptions = result.fetchall()
                print(f"\n👥 Suscripciones:")
                for status, count in subscriptions:
                    print(f"  - {status}: {count}")
            
            if 'usage_metrics' in tables:
                result = conn.execute(text("SELECT COUNT(*) FROM usage_metrics"))
                usage_count = result.fetchone()[0]
                print(f"\n📊 Métricas de uso: {usage_count} registros")
        
    except Exception as e:
        print(f"❌ Error obteniendo información: {e}")

def main():
    """Función principal"""
    print("🔧 Configuración de Base de Datos - Sistema de Suscripciones")
    print("=" * 60)
    
    # Verificar variables de entorno
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("⚠️  DATABASE_URL no configurada, usando valor por defecto")
        print("   Configura DATABASE_URL en tu archivo .env")
    
    # Configurar base de datos
    setup_database()
    
    # Probar conexión
    test_database_connection()
    
    # Mostrar información
    show_database_info()
    
    print("\n🎉 Configuración completada!")
    print("\n📝 Próximos pasos:")
    print("1. Verifica que MySQL esté ejecutándose")
    print("2. Configura las variables de entorno en .env")
    print("3. Ejecuta las migraciones: alembic upgrade head")
    print("4. Inicia el servidor: python main.py")

if __name__ == "__main__":
    main() 