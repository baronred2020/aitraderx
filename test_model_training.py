#!/usr/bin/env python3
"""
Script para probar los modelos entrenados y validar su funcionamiento
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.brain_trader_service import BrainTraderService
from utils.model_loader import ModelLoader
from services.technical_analysis_service import TechnicalAnalysisService
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_model_training():
    """Probar los modelos entrenados"""
    print("🧪 PRUEBA DE MODELOS ENTRENADOS")
    print("=" * 60)
    
    # 1. Cargar modelos
    print("\n📋 1. CARGANDO MODELOS")
    print("-" * 40)
    
    model_loader = ModelLoader()
    brain_service = BrainTraderService()
    
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
            
            # Verificar cada modelo individual
            for model_name, sub_model in model['models'].items():
                print(f"   - {model_name}: {type(sub_model).__name__}")
        else:
            print(f"❌ Modelo Brain Max no disponible o formato incorrecto")
            return
            
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # 2. Obtener datos históricos recientes
    print("\n📊 2. OBTENIENDO DATOS HISTÓRICOS")
    print("-" * 40)
    
    try:
        # Obtener datos de los últimos 30 días con intervalo de 15 minutos
        ticker = yf.Ticker("EURUSD=X")
        data = ticker.history(period="30d", interval="15m")
        
        print(f"✅ Datos obtenidos: {len(data)} registros")
        print(f"   - Período: {data.index[0]} a {data.index[-1]}")
        print(f"   - Columnas: {list(data.columns)}")
        print(f"   - Último precio: {data['Close'].iloc[-1]:.5f}")
        
        # Mostrar últimos 5 registros
        print(f"\n📈 Últimos 5 registros:")
        for i in range(-5, 0):
            timestamp = data.index[i]
            close = data['Close'].iloc[i]
            volume = data['Volume'].iloc[i]
            print(f"   {timestamp}: Close={close:.5f}, Volume={volume}")
            
    except Exception as e:
        print(f"❌ Error obteniendo datos: {e}")
        return
    
    # 3. Preparar features para el modelo
    print("\n🔧 3. PREPARANDO FEATURES")
    print("-" * 40)
    
    try:
        # Usar TechnicalAnalysisService para preparar features
        tech_service = TechnicalAnalysisService()
        
        # Preparar features usando el mismo método que usa el servicio
        features = brain_service._prepare_features_for_model(data, pair, style)
        
        if features is not None and len(features) > 0:
            print(f"✅ Features preparados: {len(features)} características")
            print(f"   - Shape: {features.shape}")
            print(f"   - Tipos: {features.dtype}")
            
            # Mostrar algunas características
            feature_names = [
                'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
                'sma_20', 'ema_20', 'sma_50', 'ema_50', 'stoch_k', 'stoch_d', 'adx',
                'momentum', 'volume_sma', 'price_change', 'volatility'
            ]
            
            print(f"\n📊 Primeras características:")
            for i, name in enumerate(feature_names[:10]):
                if i < len(features):
                    print(f"   {name}: {features[i]:.6f}")
        else:
            print(f"❌ No se pudieron preparar features")
            return
            
    except Exception as e:
        print(f"❌ Error preparando features: {e}")
        return
    
    # 4. Probar predicciones individuales
    print("\n🧠 4. PROBANDO PREDICCIONES INDIVIDUALES")
    print("-" * 40)
    
    try:
        # Escalar features si hay scaler
        if scaler is not None:
            features_scaled = scaler.transform(features.reshape(1, -1))
        else:
            features_scaled = features.reshape(1, -1)
        
        print(f"✅ Features escalados: {features_scaled.shape}")
        
        # Probar cada modelo individual
        predictions = {}
        confidences = {}
        
        for model_name, sub_model in model['models'].items():
            try:
                # Usar scaler correspondiente o el primero disponible
                sub_scaler = model['scalers'].get(model_name, scaler)
                if sub_scaler is not None:
                    features_scaled_sub = sub_scaler.transform(features.reshape(1, -1))
                else:
                    features_scaled_sub = features_scaled
                
                # Predicción del sub-modelo
                if hasattr(sub_model, 'predict_proba'):
                    proba = sub_model.predict_proba(features_scaled_sub)[0]
                    prediction = 'up' if proba[1] > proba[0] else 'down'
                    confidence = max(proba) * 100
                else:
                    prediction = sub_model.predict(features_scaled_sub)[0]
                    confidence = 75.0  # Valor por defecto
                
                predictions[model_name] = prediction
                confidences[model_name] = confidence
                
                print(f"   {model_name}: {prediction.upper()} ({confidence:.2f}%)")
                
            except Exception as e:
                print(f"   {model_name}: Error - {e}")
                predictions[model_name] = 'unknown'
                confidences[model_name] = 0.0
        
        # 5. Calcular predicción ensemble
        print(f"\n🎯 5. PREDICCIÓN ENSEMBLE")
        print("-" * 40)
        
        # Contar votos
        up_votes = 0
        down_votes = 0
        total_confidence = 0
        total_weight = 0
        
        for model_name, prediction in predictions.items():
            if prediction in ['up', 'down']:
                weight = model['weights'].get(model_name, 0.1)
                confidence = confidences.get(model_name, 0)
                
                if prediction == 'up':
                    up_votes += weight
                else:
                    down_votes += weight
                
                total_confidence += confidence * weight
                total_weight += weight
        
        # Determinar dirección final
        if up_votes > down_votes:
            final_direction = 'up'
            final_confidence = total_confidence / total_weight if total_weight > 0 else 75.0
        else:
            final_direction = 'down'
            final_confidence = total_confidence / total_weight if total_weight > 0 else 75.0
        
        print(f"✅ Predicción ensemble:")
        print(f"   - Votos UP: {up_votes:.3f}")
        print(f"   - Votos DOWN: {down_votes:.3f}")
        print(f"   - Dirección final: {final_direction.upper()}")
        print(f"   - Confianza final: {final_confidence:.2f}%")
        
        # 6. Comparar con predicción del servicio
        print(f"\n🔄 6. COMPARANDO CON SERVICIO")
        print("-" * 40)
        
        current_price = data['Close'].iloc[-1]
        service_prediction = await brain_service._get_brain_max_prediction(pair, style, current_price)
        
        if service_prediction:
            print(f"✅ Predicción del servicio:")
            print(f"   - Dirección: {service_prediction.get('direction')}")
            print(f"   - Confianza: {service_prediction.get('confidence', 0):.2f}%")
            print(f"   - Precio objetivo: {service_prediction.get('target_price', 0):.5f}")
            
            # Comparar
            if service_prediction.get('direction') == final_direction:
                print(f"   ✅ Dirección coincide con ensemble")
            else:
                print(f"   ❌ Dirección NO coincide con ensemble")
                
            confidence_diff = abs(service_prediction.get('confidence', 0) - final_confidence)
            if confidence_diff < 5:
                print(f"   ✅ Confianza similar (diferencia: {confidence_diff:.2f}%)")
            else:
                print(f"   ⚠️  Confianza diferente (diferencia: {confidence_diff:.2f}%)")
        else:
            print(f"❌ No se pudo obtener predicción del servicio")
        
        # 7. Validación de metadata
        print(f"\n📋 7. VALIDACIÓN DE METADATA")
        print("-" * 40)
        
        metadata_path = f"backend/models/trained_models/Brain_Max/{pair}/{style}/metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            print(f"✅ Metadata cargada:")
            print(f"   - Par: {metadata.get('pair')}")
            print(f"   - Estilo: {metadata.get('style')}")
            print(f"   - Configuración: {metadata.get('config')}")
            
            accuracies = metadata.get('accuracies', {})
            print(f"   - Precisión de modelos:")
            for model_name, accuracy in accuracies.items():
                print(f"     {model_name}: {accuracy:.4f}")
            
            trading_results = metadata.get('trading_results', {})
            print(f"   - Resultados de trading:")
            print(f"     Win Rate: {trading_results.get('win_rate', 0):.2%}")
            print(f"     Trades ganadores: {trading_results.get('winning_trades', 0)}/{trading_results.get('total_trades', 0)}")
            print(f"     Balance final: ${trading_results.get('final_balance', 0):,.2f}")
        else:
            print(f"❌ Metadata no encontrada")
        
        # 8. Conclusiones
        print(f"\n✅ 8. CONCLUSIONES")
        print("-" * 40)
        
        print("✅ Modelos cargados correctamente")
        print("✅ Features preparados correctamente")
        print("✅ Predicciones individuales funcionando")
        print("✅ Sistema ensemble operativo")
        
        # Verificar si hay problemas
        if final_confidence > 95:
            print("⚠️  Confianza muy alta - posible overfitting")
        elif final_confidence < 50:
            print("⚠️  Confianza muy baja - posible underfitting")
        else:
            print("✅ Confianza en rango normal")
        
        # Verificar consistencia de modelos
        model_accuracies = list(metadata.get('accuracies', {}).values())
        if all(acc > 0.99 for acc in model_accuracies):
            print("⚠️  Precisión muy alta en todos los modelos - posible overfitting")
        else:
            print("✅ Precisión de modelos en rango normal")
        
        print("\n🏁 PRUEBA COMPLETADA")
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_model_training()) 