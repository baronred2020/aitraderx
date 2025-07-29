#!/usr/bin/env python3
"""
Script para insertar predicciones de prueba
"""
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import uuid
import json

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / "backend" / "src"))

from config.database_config import db_config
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_test_predictions():
    """Insertar predicciones de prueba"""
    print("🔧 Insertando predicciones de prueba...")
    
    try:
        with db_config.get_connection() as connection:
            cursor = connection.cursor()
            
            # Insertar 3 predicciones de prueba para el usuario demo_user
            test_predictions = [
                {
                    "prediction_id": str(uuid.uuid4()),
                    "user_id": "0bb94f45-4299-4506-b8c4-9d12d438c79c",  # demo_user
                    "model_id": None,  # NULL ya que no hay modelos
                    "symbol": "EURUSD",
                    "prediction_type": "price",
                    "input_data": json.dumps({"price": 1.1587, "volume": 1000}),
                    "technical_indicators": json.dumps({"rsi": 65, "macd": 0.002}),
                    "predicted_value": 1.1595,
                    "predicted_signal": "BUY",
                    "confidence": 85.5,
                    "target_price": 1.1595,
                    "timeframe": "15M",
                    "prediction_date": datetime.now(),
                    "actual_value": None,
                    "actual_signal": None,
                    "accuracy": None,
                    "updated_at": datetime.now()
                },
                {
                    "prediction_id": str(uuid.uuid4()),
                    "user_id": "0bb94f45-4299-4506-b8c4-9d12d438c79c",  # demo_user
                    "model_id": None,  # NULL ya que no hay modelos
                    "symbol": "GBPUSD",
                    "prediction_type": "price",
                    "input_data": json.dumps({"price": 1.2850, "volume": 1200}),
                    "technical_indicators": json.dumps({"rsi": 45, "macd": -0.001}),
                    "predicted_value": 1.2830,
                    "predicted_signal": "SELL",
                    "confidence": 78.2,
                    "target_price": 1.2830,
                    "timeframe": "15M",
                    "prediction_date": datetime.now() - timedelta(minutes=30),
                    "actual_value": 1.2835,
                    "actual_signal": "SELL",
                    "accuracy": 85.0,
                    "updated_at": datetime.now() - timedelta(minutes=15)
                },
                {
                    "prediction_id": str(uuid.uuid4()),
                    "user_id": "0bb94f45-4299-4506-b8c4-9d12d438c79c",  # demo_user
                    "model_id": None,  # NULL ya que no hay modelos
                    "symbol": "USDJPY",
                    "prediction_type": "price",
                    "input_data": json.dumps({"price": 148.50, "volume": 800}),
                    "technical_indicators": json.dumps({"rsi": 70, "macd": 0.003}),
                    "predicted_value": 148.80,
                    "predicted_signal": "BUY",
                    "confidence": 92.1,
                    "target_price": 148.80,
                    "timeframe": "15M",
                    "prediction_date": datetime.now() - timedelta(hours=2),
                    "actual_value": 148.85,
                    "actual_signal": "BUY",
                    "accuracy": 95.0,
                    "updated_at": datetime.now() - timedelta(hours=1, minutes=45)
                }
            ]
            
            for pred in test_predictions:
                insert_query = """
                    INSERT INTO predictions 
                    (prediction_id, user_id, model_id, symbol, prediction_type, input_data,
                     technical_indicators, predicted_value, predicted_signal, confidence,
                     target_price, timeframe, prediction_date, actual_value, actual_signal,
                     accuracy, updated_at) 
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    pred["prediction_id"],
                    pred["user_id"],
                    pred["model_id"],
                    pred["symbol"],
                    pred["prediction_type"],
                    pred["input_data"],
                    pred["technical_indicators"],
                    pred["predicted_value"],
                    pred["predicted_signal"],
                    pred["confidence"],
                    pred["target_price"],
                    pred["timeframe"],
                    pred["prediction_date"],
                    pred["actual_value"],
                    pred["actual_signal"],
                    pred["accuracy"],
                    pred["updated_at"]
                ))
            
            connection.commit()
            cursor.close()
            print("✅ 3 predicciones de prueba insertadas exitosamente")
            return True
            
    except Exception as e:
        print(f"❌ Error insertando predicciones de prueba: {e}")
        return False

if __name__ == "__main__":
    success = insert_test_predictions()
    sys.exit(0 if success else 1)