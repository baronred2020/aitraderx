#!/usr/bin/env python3
"""
Script de prueba para el nuevo modelo EURUSD day_trading entrenado
"""

import asyncio
import sys
import os
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def test_new_model():
    """Probar el nuevo modelo entrenado"""
    print("🧠 PROBANDO NUEVO MODELO EURUSD DAY TRADING")
    print("=" * 50)
    
    try:
        from services.brain_trader_service import BrainTraderService
        
        # Crear instancia del servicio
        service = BrainTraderService()
        
        # Precio actual de EURUSD (aproximado)
        current_price = 1.0925
        
        print(f"📊 Precio actual EURUSD: {current_price}")
        print("🔄 Generando predicción...")
        
        # Generar predicción usando el nuevo modelo
        prediction = await service._get_brain_max_prediction('EURUSD', 'day_trading', current_price)
        
        print("\n✅ PREDICCIÓN GENERADA:")
        print(f"   Dirección: {prediction['direction']}")
        print(f"   Confianza: {prediction['confidence']:.1f}%")
        print(f"   Precio objetivo: {prediction['target_price']:.5f}")
        print(f"   Razón: {prediction['reasoning']}")
        
        if 'model_info' in prediction:
            print(f"   Modelo: {prediction['model_info']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando modelo: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_full_prediction_flow():
    """Probar el flujo completo de predicciones"""
    print("\n🔄 PROBANDO FLUJO COMPLETO DE PREDICCIONES")
    print("=" * 50)
    
    try:
        from services.brain_trader_service import BrainTraderService
        
        service = BrainTraderService()
        
        # Generar múltiples predicciones
        predictions = await service.get_predictions(
            brain_type='brain_max',
            pair='EURUSD',
            style='day_trading',
            limit=3,
            plan_type='trader'
        )
        
        print(f"✅ Generadas {len(predictions)} predicciones:")
        
        for i, pred in enumerate(predictions, 1):
            print(f"\n   Predicción {i}:")
            print(f"   - Par: {pred.pair}")
            print(f"   - Dirección: {pred.direction}")
            print(f"   - Confianza: {pred.confidence:.1f}%")
            print(f"   - Precio objetivo: {pred.target_price:.5f}")
            print(f"   - Timeframe: {pred.timeframe}")
            print(f"   - Expira: {pred.expires_at}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en flujo completo: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Función principal"""
    print(f"🚀 INICIANDO PRUEBAS - {datetime.now()}")
    print("=" * 60)
    
    # Probar modelo individual
    success1 = await test_new_model()
    
    # Probar flujo completo
    success2 = await test_full_prediction_flow()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("✅ El nuevo modelo está integrado y funcionando correctamente")
    else:
        print("⚠️ Algunas pruebas fallaron")
        print("🔧 Revisar logs para más detalles")
    
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main()) 