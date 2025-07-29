#!/usr/bin/env python3
"""
Script de prueba para el historial de predicciones mejorado
Prueba la funcionalidad de captura de precios actuales vs precios reales
"""

import asyncio
import json
from datetime import datetime, timedelta
import numpy as np

# Simular el servicio de predicciones
class MockPredictionService:
    def __init__(self):
        self.predictions = []
        self.base_price = 1.0850
    
    async def get_real_price(self, pair: str) -> float:
        """Simular obtención de precio real"""
        # Simular pequeñas variaciones en el precio
        variation = np.random.uniform(-0.001, 0.001)
        return self.base_price + variation
    
    async def generate_prediction(self, user_id: int, pair: str, brain_type: str, style: str) -> dict:
        """Generar predicción con precio real capturado"""
        # Obtener precio actual real
        current_price = await self.get_real_price(pair)
        
        # Generar predicción
        direction = "up" if np.random.random() > 0.5 else "down"
        confidence = np.random.uniform(70, 95)
        volatility = 0.001  # 0.1% base volatility
        
        if direction == "up":
            target_price = current_price * (1 + volatility)
        else:
            target_price = current_price * (1 - volatility)
        
        prediction = {
            "id": len(self.predictions) + 1,
            "pair": pair,
            "direction": direction,
            "current_price": current_price,  # Precio real capturado
            "target_price": target_price,
            "confidence": confidence,
            "timeframe": "15M",
            "reasoning": f"Análisis técnico para {pair} usando {brain_type} - {direction.upper()}",
            "brain_type": brain_type,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(minutes=15)).isoformat(),
            "is_completed": False,
            "actual_price_at_expiry": None,
            "prediction_success": None,
            "success_percentage": None
        }
        
        self.predictions.append(prediction)
        return prediction
    
    async def complete_prediction(self, prediction_id: int) -> dict:
        """Completar predicción con precio real al expirar"""
        if prediction_id > len(self.predictions):
            return {"success": False, "error": "Prediction not found"}
        
        prediction = self.predictions[prediction_id - 1]
        
        # Simular precio real al expirar
        actual_price = await self.get_real_price(prediction["pair"])
        
        # Calcular éxito de la predicción
        prediction_success = False
        if prediction["direction"] == "up":
            prediction_success = actual_price >= prediction["target_price"]
        else:
            prediction_success = actual_price <= prediction["target_price"]
        
        # Calcular porcentaje de éxito
        if prediction["direction"] == "up":
            if actual_price >= prediction["target_price"]:
                success_percentage = 100.0
            else:
                movement = (actual_price - prediction["current_price"]) / (prediction["target_price"] - prediction["current_price"])
                success_percentage = max(0, min(100, movement * 100))
        else:
            if actual_price <= prediction["target_price"]:
                success_percentage = 100.0
            else:
                movement = (prediction["current_price"] - actual_price) / (prediction["current_price"] - prediction["target_price"])
                success_percentage = max(0, min(100, movement * 100))
        
        # Actualizar predicción
        prediction.update({
            "is_completed": True,
            "actual_price_at_expiry": actual_price,
            "prediction_success": prediction_success,
            "success_percentage": success_percentage
        })
        
        return {
            "success": True,
            "prediction": prediction
        }
    
    def get_prediction_history(self) -> list:
        """Obtener historial de predicciones"""
        return self.predictions

async def test_prediction_system():
    """Probar el sistema de predicciones mejorado"""
    print("🧪 Probando Sistema de Historial de Predicciones")
    print("=" * 60)
    
    service = MockPredictionService()
    
    # Generar algunas predicciones
    print("\n📊 Generando predicciones...")
    for i in range(5):
        prediction = await service.generate_prediction(
            user_id=1,
            pair="EURUSD",
            brain_type="brain_max",
            style="day_trading"
        )
        
        print(f"✅ Predicción {i+1}:")
        print(f"   Par: {prediction['pair']}")
        print(f"   Dirección: {prediction['direction'].upper()}")
        print(f"   Precio Actual: ${prediction['current_price']:.5f}")
        print(f"   Precio Objetivo: ${prediction['target_price']:.5f}")
        print(f"   Confianza: {prediction['confidence']:.1f}%")
        print(f"   Brain: {prediction['brain_type']}")
        print()
    
    # Completar algunas predicciones
    print("⏰ Completando predicciones...")
    for i in range(3):
        result = await service.complete_prediction(i + 1)
        if result["success"]:
            pred = result["prediction"]
            print(f"✅ Predicción {i+1} completada:")
            print(f"   Precio Real: ${pred['actual_price_at_expiry']:.5f}")
            print(f"   Éxito: {'✓' if pred['prediction_success'] else '✗'}")
            print(f"   Porcentaje Éxito: {pred['success_percentage']:.1f}%")
            
            # Calcular diferencias
            price_diff = pred['actual_price_at_expiry'] - pred['target_price']
            price_diff_pct = (price_diff / pred['target_price']) * 100
            movement_real = pred['actual_price_at_expiry'] - pred['current_price']
            movement_real_pct = (movement_real / pred['current_price']) * 100
            
            print(f"   Diferencia vs Objetivo: {price_diff_pct:.3f}%")
            print(f"   Movimiento Real: {movement_real_pct:.3f}%")
            print()
    
    # Mostrar historial completo
    print("📋 Historial Completo de Predicciones:")
    print("-" * 60)
    
    history = service.get_prediction_history()
    for i, pred in enumerate(history):
        status = "✅ Completada" if pred['is_completed'] else "⏳ Pendiente"
        success_icon = "✓" if pred.get('prediction_success') else "✗" if pred.get('prediction_success') is False else "?"
        
        print(f"{i+1}. {pred['pair']} - {pred['direction'].upper()} - {status} {success_icon}")
        print(f"   Precio al Generar: ${pred['current_price']:.5f}")
        print(f"   Precio Objetivo: ${pred['target_price']:.5f}")
        
        if pred['is_completed']:
            print(f"   Precio Real: ${pred['actual_price_at_expiry']:.5f}")
            print(f"   Éxito: {pred['prediction_success']}")
            print(f"   Porcentaje: {pred['success_percentage']:.1f}%")
        else:
            print(f"   Precio Real: Pendiente")
            print(f"   Éxito: Pendiente")
            print(f"   Porcentaje: Pendiente")
        
        print(f"   Brain: {pred['brain_type']}")
        print(f"   Confianza: {pred['confidence']:.1f}%")
        print(f"   Fecha: {pred['created_at']}")
        print()

async def main():
    """Función principal"""
    try:
        await test_prediction_system()
        print("🎉 Prueba completada exitosamente!")
    except Exception as e:
        print(f"❌ Error en la prueba: {e}")

if __name__ == "__main__":
    asyncio.run(main())