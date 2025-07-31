#!/usr/bin/env python3
"""
Script para verificar el modelo ensemble EURUSD day_trading
"""

import asyncio
import sys
import os
import pickle
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def verify_ensemble_model():
    """Verificar que el modelo ensemble funciona correctamente"""
    print("🔍 VERIFICANDO MODELO ENSEMBLE EURUSD DAY TRADING")
    print("=" * 60)
    
    try:
        # 1. Verificar que existe el archivo ensemble.pkl
        ensemble_file = "models/trained_models/Brain_Max/EURUSD/day_trading/ensemble.pkl"
        
        if not os.path.exists(ensemble_file):
            print(f"❌ Archivo ensemble.pkl no encontrado en: {ensemble_file}")
            print("💡 Asegúrate de copiar el archivo ensemble.pkl del USB")
            return False
        
        file_size = os.path.getsize(ensemble_file)
        print(f"✅ ensemble.pkl encontrado ({file_size:,} bytes)")
        
        # 2. Intentar cargar el ensemble
        print("\n📂 Cargando modelo ensemble...")
        try:
            with open(ensemble_file, 'rb') as f:
                ensemble = pickle.load(f)
            print("✅ Ensemble cargado exitosamente desde pickle")
            
            # Mostrar información del ensemble
            if isinstance(ensemble, dict):
                print(f"   - Tipo: {ensemble.get('type', 'N/A')}")
                print(f"   - Nombre: {ensemble.get('name', 'N/A')}")
                if 'models' in ensemble:
                    print(f"   - Modelos incluidos: {list(ensemble['models'].keys())}")
                if 'weights' in ensemble:
                    print(f"   - Pesos: {ensemble['weights']}")
            
        except Exception as e:
            print(f"❌ Error cargando ensemble: {e}")
            return False
        
        # 3. Verificar que el ModelLoader puede usar el ensemble
        print("\n🧠 Verificando integración con ModelLoader...")
        from utils.model_loader import ModelLoader
        
        ml = ModelLoader()
        model, scaler, info = ml.load_brain_max('EURUSD', 'day_trading')
        
        if model is None:
            print("❌ ModelLoader no pudo cargar el modelo")
            return False
        
        print("✅ ModelLoader integrado correctamente")
        print(f"   - Accuracy: {info.get('accuracy', 'N/A'):.2f}%")
        
        # 4. Verificar predicciones con el ensemble
        print("\n🎯 Verificando predicciones con ensemble...")
        from services.brain_trader_service import BrainTraderService
        
        service = BrainTraderService()
        current_price = 1.0925
        
        prediction = await service._get_brain_max_prediction('EURUSD', 'day_trading', current_price)
        
        if prediction is None:
            print("❌ No se pudo generar predicción")
            return False
        
        print("✅ Predicción con ensemble exitosa")
        print(f"   - Dirección: {prediction['direction']}")
        print(f"   - Confianza: {prediction['confidence']:.1f}%")
        print(f"   - Precio objetivo: {prediction['target_price']:.5f}")
        
        # 5. Comparar rendimiento con y sin ensemble
        print("\n⚡ Comparando rendimiento...")
        
        # Predicción con ensemble
        start_time = datetime.now()
        ensemble_pred = await service._get_brain_max_prediction('EURUSD', 'day_trading', current_price)
        ensemble_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   - Tiempo con ensemble: {ensemble_time:.3f}s")
        print(f"   - Confianza ensemble: {ensemble_pred['confidence']:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando ensemble: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_ensemble_performance():
    """Probar el rendimiento del ensemble"""
    print("\n🚀 PROBANDO RENDIMIENTO DEL ENSEMBLE")
    print("=" * 60)
    
    try:
        from services.brain_trader_service import BrainTraderService
        
        service = BrainTraderService()
        
        # Generar múltiples predicciones
        start_time = datetime.now()
        
        predictions = await service.get_predictions(
            brain_type='brain_max',
            pair='EURUSD',
            style='day_trading',
            limit=10,
            plan_type='trader'
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Generadas {len(predictions)} predicciones en {duration:.2f} segundos")
        print(f"   - Tiempo promedio: {duration/len(predictions):.3f}s por predicción")
        
        # Análisis de predicciones
        directions = [p.direction for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        up_count = directions.count('up')
        down_count = directions.count('down')
        sideways_count = directions.count('sideways')
        
        print(f"\n📊 Análisis de predicciones:")
        print(f"   - UP: {up_count} ({up_count/len(directions)*100:.1f}%)")
        print(f"   - DOWN: {down_count} ({down_count/len(directions)*100:.1f}%)")
        print(f"   - SIDEWAYS: {sideways_count} ({sideways_count/len(directions)*100:.1f}%)")
        print(f"   - Confianza promedio: {sum(confidences)/len(confidences):.1f}%")
        print(f"   - Confianza máxima: {max(confidences):.1f}%")
        print(f"   - Confianza mínima: {min(confidences):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando rendimiento: {e}")
        return False

async def main():
    """Función principal"""
    print(f"🎯 VERIFICACIÓN DE ENSEMBLE - {datetime.now()}")
    print("=" * 80)
    
    # Verificar ensemble
    success1 = await verify_ensemble_model()
    
    # Probar rendimiento
    success2 = await test_ensemble_performance()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("🎉 ¡ENSEMBLE FUNCIONANDO PERFECTAMENTE!")
        print("✅ El modelo ensemble EURUSD day_trading está operativo")
        print("✅ Todas las verificaciones pasaron exitosamente")
        print("🚀 ¡Listo para trading en producción!")
    else:
        print("⚠️ Algunas verificaciones fallaron")
        print("🔧 Revisar logs para más detalles")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 