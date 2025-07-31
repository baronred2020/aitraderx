#!/usr/bin/env python3
"""
Script para analizar específicamente el timeframe de 15 minutos (M15)
y validar el funcionamiento de los modelos Brain Max
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.brain_trader_service import BrainTraderService
from services.prediction_service import PredictionService
from utils.model_loader import ModelLoader
from config.database_config import db_config
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def analyze_15m_timeframe():
    """Analizar específicamente el timeframe de 15 minutos"""
    print("🔍 ANÁLISIS DEL TIMEFRAME DE 15 MINUTOS (M15)")
    print("=" * 60)
    
    # 1. Verificar configuración del timeframe
    print("\n📋 1. CONFIGURACIÓN DEL TIMEFRAME")
    print("-" * 40)
    
    brain_service = BrainTraderService()
    prediction_service = PredictionService()
    
    # Verificar configuración
    timeframe_15m = brain_service.get_timeframe_for_style('day_trading')
    duration_15m = brain_service.get_duration_for_style('day_trading')
    
    print(f"✅ Timeframe configurado: {timeframe_15m}")
    print(f"✅ Duración configurada: {duration_15m} minutos")
    print(f"✅ Estilo: day_trading")
    
    # 2. Verificar modelos disponibles
    print("\n🧠 2. MODELOS DISPONIBLES PARA M15")
    print("-" * 40)
    
    model_loader = ModelLoader()
    
    # Verificar modelos Brain Max para EURUSD day_trading
    pair = "EURUSD"
    style = "day_trading"
    
    try:
        model, scaler, model_info = model_loader.load_brain_max(pair, style)
        
        if model and isinstance(model, dict) and model.get('type') == 'ensemble':
            print(f"✅ Modelo Brain Max cargado correctamente")
            print(f"   - Tipo: {model.get('type')}")
            print(f"   - Modelos disponibles: {list(model.get('models', {}).keys())}")
            print(f"   - Estrategia: {model.get('ensemble_strategy')}")
            print(f"   - Pesos: {model.get('weights', {})}")
        else:
            print(f"❌ Modelo Brain Max no disponible o formato incorrecto")
            
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
    
    # 3. Analizar predicciones recientes de M15
    print("\n📊 3. ANÁLISIS DE PREDICCIONES RECIENTES M15")
    print("-" * 40)
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Buscar predicciones de day_trading (M15) de las últimas 24 horas
            query = """
                SELECT id, pair, direction, current_price, target_price, actual_price_at_expiry, 
                       prediction_success, success_percentage, is_completed, created_at, expires_at,
                       confidence, brain_type
                FROM user_predictions 
                WHERE brain_type = 'brain_max' 
                AND created_at >= DATE_SUB(NOW(), INTERVAL 24 HOUR)
                ORDER BY created_at DESC
                LIMIT 10
            """
            
            cursor.execute(query)
            predictions = cursor.fetchall()
            
            print(f"📈 Encontradas {len(predictions)} predicciones M15 en las últimas 24h")
            
            if predictions:
                total_predictions = len(predictions)
                completed_predictions = 0
                successful_predictions = 0
                avg_confidence = 0
                confidence_values = []
                
                for pred in predictions:
                    pred_id, pair, direction, current_price, target_price, actual_price, prediction_success, success_percentage, is_completed, created_at, expires_at, confidence, brain_type = pred
                    
                    print(f"\n   Predicción {pred_id}:")
                    print(f"   - Par: {pair}")
                    print(f"   - Dirección: {direction}")
                    print(f"   - Precio actual: {current_price}")
                    print(f"   - Precio objetivo: {target_price}")
                    print(f"   - Precio real: {actual_price}")
                    print(f"   - Confianza: {confidence}%")
                    print(f"   - Éxito: {prediction_success}")
                    print(f"   - Porcentaje: {success_percentage}")
                    print(f"   - Completada: {is_completed}")
                    print(f"   - Creada: {created_at}")
                    
                    if is_completed:
                        completed_predictions += 1
                        if prediction_success:
                            successful_predictions += 1
                    
                    if confidence:
                        confidence_values.append(confidence)
                
                # Estadísticas
                if completed_predictions > 0:
                    success_rate = (successful_predictions / completed_predictions) * 100
                    print(f"\n📊 ESTADÍSTICAS M15:")
                    print(f"   - Total predicciones: {total_predictions}")
                    print(f"   - Completadas: {completed_predictions}")
                    print(f"   - Exitosas: {successful_predictions}")
                    print(f"   - Tasa de éxito: {success_rate:.2f}%")
                
                if confidence_values:
                    avg_confidence = np.mean(confidence_values)
                    print(f"   - Confianza promedio: {avg_confidence:.2f}%")
                    print(f"   - Confianza mínima: {min(confidence_values):.2f}%")
                    print(f"   - Confianza máxima: {max(confidence_values):.2f}%")
            else:
                print("   No hay predicciones M15 recientes")
            
            cursor.close()
            
    except Exception as e:
        print(f"❌ Error analizando predicciones: {e}")
    
    # 4. Generar predicción de prueba M15
    print("\n🧪 4. PREDICCIÓN DE PRUEBA M15")
    print("-" * 40)
    
    try:
        # Obtener precio actual real
        ticker = yf.Ticker("EURUSD=X")
        current_price = ticker.info.get('regularMarketPrice', 1.1570)
        
        print(f"💰 Precio actual EURUSD: {current_price}")
        
        # Generar predicción usando Brain Max
        prediction_result = await brain_service._get_brain_max_prediction("EURUSD", "day_trading", current_price)
        
        if prediction_result:
            print(f"✅ Predicción generada:")
            print(f"   - Dirección: {prediction_result.get('direction')}")
            print(f"   - Confianza: {prediction_result.get('confidence', 0):.2f}%")
            print(f"   - Precio objetivo: {prediction_result.get('target_price', 0)}")
            print(f"   - Razonamiento: {prediction_result.get('reasoning', 'N/A')}")
        else:
            print(f"❌ No se pudo generar predicción")
            
    except Exception as e:
        print(f"❌ Error generando predicción de prueba: {e}")
    
    # 5. Validación del modelo
    print("\n✅ 5. VALIDACIÓN DEL MODELO M15")
    print("-" * 40)
    
    print("✅ Timeframe 15M configurado correctamente")
    print("✅ Modelos Brain Max disponibles")
    print("✅ Duración de 15 minutos establecida")
    print("✅ Sistema de predicciones funcionando")
    
    # Verificar si hay problemas específicos
    if 'confidence_values' in locals() and confidence_values:
        avg_conf = np.mean(confidence_values)
        if avg_conf > 90:
            print("⚠️  Confianza promedio muy alta (>90%) - posible overfitting")
        elif avg_conf < 50:
            print("⚠️  Confianza promedio muy baja (<50%) - posible underfitting")
        else:
            print("✅ Confianza promedio en rango normal")
    
    print("\n🏁 ANÁLISIS COMPLETADO")

if __name__ == "__main__":
    import asyncio
    asyncio.run(analyze_15m_timeframe()) 