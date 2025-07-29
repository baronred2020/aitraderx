#!/usr/bin/env python3
"""
Script para probar el sistema de reinicio diario de predicciones
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "backend" / "src"))

from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_daily_reset():
    """Probar el sistema de reinicio diario"""
    print("🧪 Probando sistema de reinicio diario...")
    
    try:
        # 1. Verificar estado actual
        print("\n📊 1. Estado actual de predicciones...")
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Contar predicciones de hoy
            today = datetime.now().date()
            cursor.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE DATE(prediction_date) = %s
            """, (today,))
            today_predictions = cursor.fetchone()[0]
            print(f"   Predicciones de hoy: {today_predictions}")
            
            # Contar predicciones de ayer
            yesterday = today - timedelta(days=1)
            cursor.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE DATE(prediction_date) = %s
            """, (yesterday,))
            yesterday_predictions = cursor.fetchone()[0]
            print(f"   Predicciones de ayer: {yesterday_predictions}")
            
            cursor.close()
        
        # 2. Simular reinicio manual
        print("\n🔄 2. Simulando reinicio manual...")
        import requests
        
        # Nota: Esto requiere que el backend esté ejecutándose
        try:
            response = requests.post("http://localhost:8000/api/v1/predictions/reset-daily")
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ Reinicio manual exitoso: {result}")
            else:
                print(f"   ❌ Error en reinicio manual: {response.status_code}")
        except Exception as e:
            print(f"   ⚠️ No se pudo probar reinicio manual (backend no ejecutándose): {e}")
        
        # 3. Verificar logs del sistema
        print("\n📋 3. Verificando logs del sistema...")
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Verificar si existe la tabla system_logs
            cursor.execute("SHOW TABLES LIKE 'system_logs'")
            if cursor.fetchone():
                cursor.execute("""
                    SELECT action, details, created_at 
                    FROM system_logs 
                    WHERE action = 'daily_predictions_reset'
                    ORDER BY created_at DESC 
                    LIMIT 5
                """)
                logs = cursor.fetchall()
                print(f"   Logs de reinicio encontrados: {len(logs)}")
                for log in logs:
                    print(f"   - {log[0]}: {log[1]} ({log[2]})")
            else:
                print("   ⚠️ Tabla system_logs no existe")
            
            cursor.close()
        
        # 4. Probar el script directamente
        print("\n🔧 4. Probando script de reinicio directamente...")
        try:
            sys.path.append(str(Path(__file__).parent / "backend"))
            from reset_daily_predictions import reset_daily_predictions
            
            success = reset_daily_predictions()
            if success:
                print("   ✅ Script de reinicio ejecutado exitosamente")
            else:
                print("   ❌ Error ejecutando script de reinicio")
        except Exception as e:
            print(f"   ❌ Error importando script: {e}")
        
        print("\n🎉 Pruebas completadas")
        return True
        
    except Exception as e:
        print(f"❌ Error en pruebas: {e}")
        return False

if __name__ == "__main__":
    success = test_daily_reset()
    sys.exit(0 if success else 1)