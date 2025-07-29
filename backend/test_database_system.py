#!/usr/bin/env python3
"""
Script de prueba para el sistema completo con base de datos real
"""
import requests
import json
import time
import sys
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "src"))

from config.database_config import db_config

def test_database_system():
    """Probar el sistema completo con base de datos real"""
    
    base_url = "http://localhost:8000"
    
    print("🧪 Iniciando pruebas del sistema con base de datos real...\n")
    
    # 1. Probar conexión a la base de datos
    print("🔍 1. Probando conexión a la base de datos...")
    if not db_config.test_connection():
        print("❌ Error: No se puede conectar a la base de datos")
        return False
    print("✅ Conexión a la base de datos exitosa")
    
    # 2. Probar endpoint de límites
    print("\n📊 2. Probando endpoint de límites...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        if response.status_code == 200:
            limits = response.json()
            print(f"✅ Límites obtenidos: {limits}")
        else:
            print(f"❌ Error obteniendo límites: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en endpoint de límites: {e}")
        return False
    
    # 3. Probar generación de predicción
    print("\n🎯 3. Probando generación de predicción...")
    try:
        prediction_data = {
            "pair": "EURUSD",
            "brain_type": "brain_max",
            "style": "day_trading"
        }
        response = requests.post(
            f"{base_url}/api/v1/predictions/generate",
            json=prediction_data
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Predicción generada: {result.get('success', False)}")
            if result.get('prediction'):
                print(f"   - Par: {result['prediction'].get('pair')}")
                print(f"   - Dirección: {result['prediction'].get('direction')}")
                print(f"   - Confianza: {result['prediction'].get('confidence')}%")
        else:
            print(f"❌ Error generando predicción: {response.status_code}")
            print(f"   Respuesta: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error en generación de predicción: {e}")
        return False
    
    # 4. Probar historial de predicciones
    print("\n📜 4. Probando historial de predicciones...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/history?limit=5")
        if response.status_code == 200:
            history = response.json()
            print(f"✅ Historial obtenido: {len(history)} predicciones")
            for i, pred in enumerate(history[:3], 1):
                print(f"   {i}. {pred.get('pair')} - {pred.get('direction')} - {pred.get('confidence')}%")
        else:
            print(f"❌ Error obteniendo historial: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en historial: {e}")
        return False
    
    # 5. Verificar límites después de generar predicción
    print("\n📊 5. Verificando límites después de generar predicción...")
    try:
        response = requests.get(f"{base_url}/api/v1/predictions/limits?style=day_trading")
        if response.status_code == 200:
            updated_limits = response.json()
            print(f"✅ Límites actualizados: {updated_limits}")
        else:
            print(f"❌ Error obteniendo límites actualizados: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error en límites actualizados: {e}")
        return False
    
    print("\n🎉 ¡Todas las pruebas pasaron exitosamente!")
    print("✅ El sistema está funcionando correctamente con base de datos real")
    return True

if __name__ == "__main__":
    success = test_database_system()
    sys.exit(0 if success else 1)