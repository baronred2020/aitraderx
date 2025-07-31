#!/usr/bin/env python3
"""
Script de diagnóstico para identificar problemas de pickle en Colab
"""

import os
import sys
import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

def is_colab_environment():
    """Detectar si estamos en Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def test_basic_pickle():
    """Probar pickle básico en Colab"""
    print("🔍 PROBANDO PICKLE BÁSICO EN COLAB")
    print("=" * 50)
    
    # Crear directorio de prueba
    test_dir = "/content/test_pickle_diagnostic"
    os.makedirs(test_dir, exist_ok=True)
    print(f"📁 Directorio de prueba: {test_dir}")
    
    # Crear modelo simple
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    model.fit(X, y)
    
    # Intentar guardar con pickle
    try:
        model_file = os.path.join(test_dir, "test_model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print("✅ Pickle básico funciona")
        
        # Verificar que se puede cargar
        with open(model_file, 'rb') as f:
            loaded_model = pickle.load(f)
        print("✅ Modelo se puede cargar correctamente")
        
        return True
    except Exception as e:
        print(f"❌ Error con pickle básico: {e}")
        return False

def test_ensemble_pickle():
    """Probar pickle de ensemble personalizado"""
    print("\n🔍 PROBANDO PICKLE DE ENSEMBLE PERSONALIZADO")
    print("=" * 50)
    
    # Crear clase ensemble simple
    class SimpleEnsemble:
        def __init__(self, models, weights):
            self.models = models
            self.weights = weights
    
    # Crear modelos
    models = {
        'rf': RandomForestClassifier(n_estimators=10, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=10, random_state=42),
        'et': ExtraTreesClassifier(n_estimators=10, random_state=42)
    }
    
    weights = {'rf': 0.8, 'gb': 0.75, 'et': 0.82}
    
    # Crear ensemble
    ensemble = SimpleEnsemble(models, weights)
    
    test_dir = "/content/test_pickle_diagnostic"
    
    # Intentar guardar ensemble
    try:
        ensemble_file = os.path.join(test_dir, "test_ensemble.pkl")
        with open(ensemble_file, 'wb') as f:
            pickle.dump(ensemble, f)
        print("✅ Ensemble se puede guardar con pickle")
        return True
    except Exception as e:
        print(f"❌ Error guardando ensemble: {e}")
        return False

def test_individual_models_save():
    """Probar guardado de modelos individuales"""
    print("\n🔍 PROBANDO GUARDADO DE MODELOS INDIVIDUALES")
    print("=" * 50)
    
    test_dir = "/content/test_pickle_diagnostic"
    
    # Crear modelos
    models = {
        'rf': RandomForestClassifier(n_estimators=10, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=10, random_state=42),
        'et': ExtraTreesClassifier(n_estimators=10, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=10, random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1),
        'mlp': MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
    }
    
    # Entrenar modelos
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    for name, model in models.items():
        try:
            model.fit(X, y)
            print(f"✅ {name} entrenado")
        except Exception as e:
            print(f"❌ Error entrenando {name}: {e}")
    
    # Guardar modelos individuales
    saved_count = 0
    for model_name, model in models.items():
        try:
            model_file = os.path.join(test_dir, f"{model_name}_model.pkl")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            print(f"✅ {model_name}_model.pkl guardado")
            saved_count += 1
        except Exception as e:
            print(f"❌ Error guardando {model_name}_model.pkl: {e}")
    
    print(f"\n📊 Resumen: {saved_count}/6 modelos guardados")
    return saved_count == 6

def test_save_model_kaggle_simulation():
    """Simular la función save_model_kaggle"""
    print("\n🔍 SIMULANDO save_model_kaggle")
    print("=" * 50)
    
    # Crear datos de prueba
    models = {
        'rf': RandomForestClassifier(n_estimators=10, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=10, random_state=42),
        'et': ExtraTreesClassifier(n_estimators=10, random_state=42),
        'xgb': xgb.XGBClassifier(n_estimators=10, random_state=42),
        'lgb': lgb.LGBMClassifier(n_estimators=10, random_state=42, verbose=-1),
        'mlp': MLPClassifier(hidden_layer_sizes=(10,), max_iter=100, random_state=42)
    }
    
    # Entrenar modelos
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    for name, model in models.items():
        model.fit(X, y)
    
    # Crear ensemble mock
    class MockEnsemble:
        def __init__(self, models, weights):
            self.models = list(models.keys())
            self.weights = weights
    
    ensemble = MockEnsemble(models, {'rf': 0.8, 'gb': 0.75, 'et': 0.82, 'xgb': 0.78, 'lgb': 0.79, 'mlp': 0.76})
    
    # Simular additional_data
    additional_data = {
        'feature_columns': ['feature1', 'feature2', 'feature3'],
        'model_performances': {'rf': 0.8, 'gb': 0.75, 'et': 0.82, 'xgb': 0.78, 'lgb': 0.79, 'mlp': 0.76},
        'ensemble_accuracy': 0.79,
        'training_data_sample': None,
        'trained_models': models
    }
    
    # Simular save_model_kaggle
    try:
        output_dir = "/content/test_save_model_kaggle"
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar ensemble metadata
        ensemble_metadata = {
            'models_list': list(ensemble.models),
            'weights': ensemble.weights,
            'ensemble_type': 'MockEnsemble',
            'created_at': datetime.now().isoformat()
        }
        
        ensemble_file = os.path.join(output_dir, "ensemble_metadata.json")
        with open(ensemble_file, 'w') as f:
            json.dump(ensemble_metadata, f, indent=2, default=str)
        
        print("✅ ensemble_metadata.json guardado")
        
        # Guardar modelos individuales
        print("💾 Guardando modelos individuales...")
        saved_count = 0
        
        if additional_data and 'trained_models' in additional_data:
            trained_models = additional_data['trained_models']
            print(f"🔍 trained_models keys: {list(trained_models.keys())}")
            
            for model_name, model in trained_models.items():
                try:
                    model_file = os.path.join(output_dir, f"{model_name}_model.pkl")
                    with open(model_file, 'wb') as f:
                        pickle.dump(model, f)
                    print(f"   ✅ {model_name}_model.pkl guardado")
                    saved_count += 1
                except Exception as e:
                    print(f"   ❌ Error guardando {model_name}_model.pkl: {e}")
        
        print(f"📊 Modelos individuales guardados: {saved_count}/6")
        
        # Listar archivos creados
        files = os.listdir(output_dir)
        print(f"📁 Archivos en {output_dir}:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            size = os.path.getsize(file_path)
            print(f"   - {file} ({size} bytes)")
        
        return saved_count == 6
        
    except Exception as e:
        print(f"❌ Error en simulación: {e}")
        return False

def main():
    """Función principal de diagnóstico"""
    print("🔍 DIAGNÓSTICO COMPLETO DE PICKLE EN COLAB")
    print("=" * 60)
    
    # Verificar entorno
    print(f"🌐 Entorno: {'Google Colab' if is_colab_environment() else 'Otro'}")
    
    # Ejecutar pruebas
    results = {}
    
    results['basic_pickle'] = test_basic_pickle()
    results['ensemble_pickle'] = test_ensemble_pickle()
    results['individual_models'] = test_individual_models_save()
    results['save_model_kaggle'] = test_save_model_kaggle_simulation()
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE DIAGNÓSTICO")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{test_name}: {status}")
    
    # Recomendaciones
    print("\n💡 RECOMENDACIONES:")
    if not results['basic_pickle']:
        print("   - Problema fundamental con pickle en Colab")
    elif not results['ensemble_pickle']:
        print("   - Problema específico con ensembles personalizados")
    elif not results['individual_models']:
        print("   - Problema con modelos específicos (XGBoost/LightGBM)")
    elif not results['save_model_kaggle']:
        print("   - Problema en la función save_model_kaggle")
    else:
        print("   - Todos los tests pasaron. El problema puede estar en el script principal")
        print("   - Verifica que tienes la versión más reciente del Modelo_Brain_Max.py")

if __name__ == "__main__":
    main() 