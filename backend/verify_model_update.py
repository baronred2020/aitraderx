#!/usr/bin/env python3
"""
Script para verificar la actualización del modelo EURUSD day_trading
"""

import asyncio
import sys
import os
from datetime import datetime
import pickle

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

async def verify_model_update():
    """Verificar que el modelo actualizado funciona correctamente"""
    print("🔍 VERIFICANDO ACTUALIZACIÓN DEL MODELO EURUSD DAY TRADING")
    print("=" * 60)
    
    try:
        from services.brain_trader_service import BrainTraderService
        from utils.model_loader import ModelLoader
        
        # 1. Verificar que el ModelLoader puede cargar el modelo
        print("📂 Verificando carga del modelo...")
        ml = ModelLoader()
        model, scaler, info = ml.load_brain_max('EURUSD', 'day_trading')
        
        if model is None:
            print("❌ Error: No se pudo cargar el modelo")
            return False
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"   - Nombre: {info.get('name', 'N/A')}")
        print(f"   - Accuracy: {info.get('accuracy', 'N/A'):.2f}%")
        print(f"   - Último entrenamiento: {info.get('last_training', 'N/A')}")
        
        # 2. Verificar que el modelo puede hacer predicciones
        print("\n🧠 Verificando predicciones...")
        service = BrainTraderService()
        current_price = 1.0925
        
        prediction = await service._get_brain_max_prediction('EURUSD', 'day_trading', current_price)
        
        if prediction is None:
            print("❌ Error: No se pudo generar predicción")
            return False
        
        print(f"✅ Predicción generada exitosamente")
        print(f"   - Dirección: {prediction['direction']}")
        print(f"   - Confianza: {prediction['confidence']:.1f}%")
        print(f"   - Precio objetivo: {prediction['target_price']:.5f}")
        
        # 3. Verificar archivos del modelo
        print("\n📁 Verificando archivos del modelo...")
        model_dir = "models/trained_models/Brain_Max/EURUSD/day_trading"
        expected_files = [
            'et_model.pkl', 'gb_model.pkl', 'lgb_model.pkl', 
            'mlp_model.pkl', 'rf_model.pkl', 'xgb_model.pkl', 'metadata.json'
        ]
        
        missing_files = []
        for file in expected_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                print(f"   ✅ {file} ({size:,} bytes)")
            else:
                print(f"   ❌ {file} - NO ENCONTRADO")
                missing_files.append(file)
        
        if missing_files:
            print(f"\n⚠️ Archivos faltantes: {missing_files}")
        else:
            print(f"\n✅ Todos los archivos del modelo están presentes")
        
        # 4. Verificar que el modelo es diferente al anterior (si hay backup)
        print("\n🔄 Verificando cambios en el modelo...")
        backup_dir = "backup_models_20250730_175237/EURUSD_day_trading_backup"
        
        if os.path.exists(backup_dir):
            backup_metadata = os.path.join(backup_dir, 'metadata.json')
            current_metadata = os.path.join(model_dir, 'metadata.json')
            
            if os.path.exists(backup_metadata) and os.path.exists(current_metadata):
                try:
                    import json
                    with open(backup_metadata, 'r') as f:
                        backup_info = json.load(f)
                    with open(current_metadata, 'r') as f:
                        current_info = json.load(f)
                    
                    print(f"   - Backup timestamp: {backup_info.get('timestamp', 'N/A')}")
                    print(f"   - Current timestamp: {current_info.get('timestamp', 'N/A')}")
                    
                    if backup_info.get('timestamp') != current_info.get('timestamp'):
                        print("   ✅ El modelo ha sido actualizado")
                    else:
                        print("   ⚠️ El modelo parece ser el mismo")
                        
                except Exception as e:
                    print(f"   ⚠️ No se pudo comparar timestamps: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verificando actualización: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_performance():
    """Probar el rendimiento del modelo actualizado"""
    print("\n⚡ PROBANDO RENDIMIENTO DEL MODELO ACTUALIZADO")
    print("=" * 60)
    
    try:
        from services.brain_trader_service import BrainTraderService
        
        service = BrainTraderService()
        
        # Generar múltiples predicciones para probar rendimiento
        start_time = datetime.now()
        
        predictions = await service.get_predictions(
            brain_type='brain_max',
            pair='EURUSD',
            style='day_trading',
            limit=5,
            plan_type='trader'
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ Generadas {len(predictions)} predicciones en {duration:.2f} segundos")
        print(f"   - Tiempo promedio por predicción: {duration/len(predictions):.3f}s")
        
        # Mostrar resumen de predicciones
        directions = [p.direction for p in predictions]
        confidences = [p.confidence for p in predictions]
        
        print(f"   - Direcciones: {directions}")
        print(f"   - Confianza promedio: {sum(confidences)/len(confidences):.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error probando rendimiento: {e}")
        return False

async def main():
    """Función principal"""
    print(f"🚀 VERIFICACIÓN DE ACTUALIZACIÓN - {datetime.now()}")
    print("=" * 80)
    
    # Verificar actualización
    success1 = await verify_model_update()
    
    # Probar rendimiento
    success2 = await test_performance()
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("🎉 ¡ACTUALIZACIÓN EXITOSA!")
        print("✅ El modelo EURUSD day_trading ha sido actualizado correctamente")
        print("✅ Todas las verificaciones pasaron exitosamente")
    else:
        print("⚠️ Algunas verificaciones fallaron")
        print("🔧 Revisar logs para más detalles")
    
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main()) 