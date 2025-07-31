#!/usr/bin/env python3
"""
Script de diagnóstico para identificar problemas con el guardado de modelos individuales en Colab
"""

import os
import pickle
import json
from datetime import datetime

def test_colab_save_diagnostic():
    """Diagnóstico completo del problema de guardado en Colab"""
    
    print("🔍 DIAGNÓSTICO DE GUARDADO EN COLAB")
    print("=" * 60)
    
    # 1. Verificar entorno
    print("\n1️⃣ VERIFICANDO ENTORNO:")
    try:
        import google.colab
        print("✅ Detectado Google Colab")
        colab_env = True
    except ImportError:
        print("❌ No es Google Colab")
        colab_env = False
    
    # 2. Verificar directorio de salida
    print("\n2️⃣ VERIFICANDO DIRECTORIO DE SALIDA:")
    output_dir = "/content/test_models_diagnostic"
    print(f"📁 Directorio de prueba: {output_dir}")
    
    try:
        os.makedirs(output_dir, exist_ok=True)
        print("✅ Directorio creado exitosamente")
        
        # Verificar permisos
        test_file = os.path.join(output_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        print("✅ Permisos de escritura OK")
        
        # Limpiar archivo de prueba
        os.remove(test_file)
        
    except Exception as e:
        print(f"❌ Error creando directorio: {e}")
        return False
    
    # 3. Crear modelos de prueba
    print("\n3️⃣ CREANDO MODELOS DE PRUEBA:")
    try:
        from sklearn.ensemble import RandomForestClassifier
        import numpy as np
        
        # Crear datos de prueba
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        
        # Crear modelos de prueba
        test_models = {}
        
        # RandomForest
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X, y)
        test_models['rf'] = rf
        print("✅ RandomForest creado")
        
        # GradientBoosting
        from sklearn.ensemble import GradientBoostingClassifier
        gb = GradientBoostingClassifier(n_estimators=10, random_state=42)
        gb.fit(X, y)
        test_models['gb'] = gb
        print("✅ GradientBoosting creado")
        
        # ExtraTrees
        from sklearn.ensemble import ExtraTreesClassifier
        et = ExtraTreesClassifier(n_estimators=10, random_state=42)
        et.fit(X, y)
        test_models['et'] = et
        print("✅ ExtraTrees creado")
        
        print(f"✅ {len(test_models)} modelos de prueba creados")
        
    except Exception as e:
        print(f"❌ Error creando modelos: {e}")
        return False
    
    # 4. Simular la función save_model_kaggle
    print("\n4️⃣ SIMULANDO save_model_kaggle:")
    
    # Crear additional_data similar al que se pasa en el script real
    additional_data = {
        'feature_columns': ['feature1', 'feature2', 'feature3'],
        'model_performances': {'rf': 0.8, 'gb': 0.75, 'et': 0.82},
        'ensemble_accuracy': 0.79,
        'training_data_sample': None,
        'trained_models': test_models  # Los modelos de prueba
    }
    
    print(f"🔍 additional_data creado:")
    print(f"   - Keys: {list(additional_data.keys())}")
    print(f"   - trained_models disponible: {additional_data.get('trained_models') is not None}")
    print(f"   - trained_models tipo: {type(additional_data.get('trained_models'))}")
    if additional_data.get('trained_models'):
        print(f"   - trained_models keys: {list(additional_data['trained_models'].keys())}")
    
    # 5. Intentar guardar modelos individuales
    print("\n5️⃣ INTENTANDO GUARDAR MODELOS INDIVIDUALES:")
    
    try:
        if additional_data:
            print(f"🔍 additional_data = {additional_data}")
            print(f"🔍 additional_data keys = {list(additional_data.keys())}")
            
            if 'trained_models' in additional_data:
                trained_models = additional_data['trained_models']
                print(f"🔍 trained_models = {trained_models}")
                print(f"🔍 trained_models keys = {list(trained_models.keys()) if isinstance(trained_models, dict) else 'Not a dict'}")
                print("💾 Guardando modelos individuales...")
                
                if isinstance(trained_models, dict) and trained_models:
                    for model_name, model in trained_models.items():
                        try:
                            model_file = os.path.join(output_dir, f"{model_name}_model.pkl")
                            print(f"   🔍 Intentando guardar: {model_file}")
                            
                            with open(model_file, 'wb') as f:
                                pickle.dump(model, f)
                            
                            # Verificar que el archivo se creó
                            if os.path.exists(model_file):
                                file_size = os.path.getsize(model_file)
                                print(f"   ✅ {model_name}_model.pkl guardado ({file_size} bytes)")
                            else:
                                print(f"   ❌ {model_name}_model.pkl NO se creó")
                                
                        except Exception as e:
                            print(f"   ❌ Error guardando {model_name}_model.pkl: {e}")
                            import traceback
                            traceback.print_exc()
                else:
                    print(f"   ⚠️ trained_models no es un diccionario válido: {type(trained_models)}")
            else:
                print("   ⚠️ 'trained_models' no encontrado en additional_data")
        else:
            print("   ⚠️ additional_data es None o vacío")
            
    except Exception as e:
        print(f"❌ Error en el proceso de guardado: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Verificar archivos creados
    print("\n6️⃣ VERIFICANDO ARCHIVOS CREADOS:")
    try:
        files_created = os.listdir(output_dir)
        print(f"📁 Archivos en {output_dir}:")
        for file in files_created:
            file_path = os.path.join(output_dir, file)
            file_size = os.path.getsize(file_path)
            print(f"   - {file} ({file_size} bytes)")
        
        # Contar modelos individuales
        individual_models = [f for f in files_created if f.endswith('_model.pkl')]
        print(f"\n📊 Resumen:")
        print(f"   - Total archivos: {len(files_created)}")
        print(f"   - Modelos individuales: {len(individual_models)}")
        print(f"   - Modelos esperados: {len(test_models)}")
        
        if len(individual_models) == len(test_models):
            print("✅ Todos los modelos individuales se guardaron correctamente")
        else:
            print("❌ No todos los modelos individuales se guardaron")
            
    except Exception as e:
        print(f"❌ Error verificando archivos: {e}")
    
    # 7. Probar carga de modelos
    print("\n7️⃣ PROBANDO CARGA DE MODELOS:")
    try:
        for model_name in test_models.keys():
            model_file = os.path.join(output_dir, f"{model_name}_model.pkl")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    loaded_model = pickle.load(f)
                print(f"   ✅ {model_name}_model.pkl se puede cargar correctamente")
            else:
                print(f"   ❌ {model_name}_model.pkl no existe")
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
    
    print("\n" + "=" * 60)
    print("🔍 DIAGNÓSTICO COMPLETADO")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    test_colab_save_diagnostic() 