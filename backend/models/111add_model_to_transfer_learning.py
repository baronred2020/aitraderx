#!/usr/bin/env python3
"""
➕ AÑADIR MODELOS AL SISTEMA DE TRANSFER LEARNING
🎯 Script para añadir fácilmente nuevos modelos exitosos
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class ModelAdder:
    """Clase para añadir nuevos modelos al sistema de transfer learning"""
    
    def __init__(self, knowledge_file="models/transfer_learning/knowledge_registry.json"):
        self.knowledge_file = Path(knowledge_file)
        self.knowledge_file.parent.mkdir(parents=True, exist_ok=True)
        self.load_knowledge_registry()
    
    def load_knowledge_registry(self):
        """Carga el registro de conocimiento"""
        if self.knowledge_file.exists():
            with open(self.knowledge_file, 'r') as f:
                self.knowledge_registry = json.load(f)
        else:
            self.knowledge_registry = {}
    
    def save_knowledge_registry(self):
        """Guarda el registro de conocimiento"""
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge_registry, f, indent=2)
    
    def add_custom_model(self, model_name: str, features: List[str], strategies: Dict[str, Any], 
                        optimizations: Dict[str, Any], performance: Dict[str, float]):
        """Añade un modelo personalizado al sistema"""
        print(f"➕ Añadiendo modelo: {model_name}")
        
        model_knowledge = {
            "model_name": model_name,
            "successful_features": features,
            "successful_strategies": strategies,
            "successful_optimizations": optimizations,
            "performance_metrics": performance,
            "extraction_date": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.knowledge_registry[model_name] = model_knowledge
        self.save_knowledge_registry()
        
        print(f"✅ Modelo {model_name} añadido exitosamente")
        print(f"   • Features: {len(features)}")
        print(f"   • Estrategias: {len(strategies)}")
        print(f"   • Optimizaciones: {len(optimizations)}")
    
    def add_brain_max_v2(self):
        """Añade Brain_Max v2 con mejoras"""
        print("🧠 Añadiendo Brain_Max v2...")
        
        features = [
            'advanced_rsi', 'enhanced_macd', 'adaptive_bollinger_bands',
            'dynamic_support_resistance', 'market_regime_detection',
            'sentiment_analysis', 'correlation_analysis',
            'multi_timeframe_analysis', 'volatility_regime',
            'momentum_divergence', 'volume_profile_analysis'
        ]
        
        strategies = {
            'adaptive_scalping': {
                'target_precision': 0.72,
                'target_pips': 3,
                'stop_loss_pips': 1.5,
                'timeframe': 'adaptive',
                'indicators': ['advanced_rsi', 'enhanced_macd', 'market_regime_detection']
            },
            'smart_day_trading': {
                'target_precision': 0.68,
                'target_pips': 20,
                'stop_loss_pips': 10,
                'timeframe': '15m',
                'indicators': ['dynamic_support_resistance', 'sentiment_analysis']
            },
            'intelligent_swing': {
                'target_precision': 0.62,
                'target_pips': 150,
                'stop_loss_pips': 75,
                'timeframe': '1h',
                'indicators': ['multi_timeframe_analysis', 'correlation_analysis']
            }
        }
        
        optimizations = {
            'advanced_hyperparameter_tuning': {
                'n_estimators': 150,
                'max_depth': 15,
                'learning_rate': 0.08,
                'subsample': 0.85,
                'colsample_bytree': 0.9
            },
            'intelligent_feature_selection': {
                'method': 'genetic_algorithm',
                'n_features': 25,
                'threshold': 0.005,
                'selection_criteria': 'mutual_information'
            },
            'ensemble_optimization': {
                'voting': 'weighted_soft',
                'weights': [0.35, 0.35, 0.30],
                'base_models': ['rf', 'gbm', 'xgboost'],
                'meta_learner': 'stacking_with_cv'
            },
            'risk_management_v2': {
                'max_drawdown': 0.03,
                'profit_factor': 2.5,
                'sharpe_ratio': 2.0,
                'calmar_ratio': 3.0,
                'sortino_ratio': 2.5
            }
        }
        
        performance = {
            'accuracy': 0.78,
            'precision': 0.75,
            'recall': 0.82,
            'f1_score': 0.78,
            'sharpe_ratio': 2.2,
            'max_drawdown': 0.025,
            'profit_factor': 2.8,
            'win_rate': 0.68
        }
        
        self.add_custom_model("Brain_Max_v2", features, strategies, optimizations, performance)
    
    def add_ultra_model_v2(self):
        """Añade Ultra_Model v2 con mejoras"""
        print("🚀 Añadiendo Ultra_Model v2...")
        
        features = [
            'transformer_attention', 'lstm_sequence_analysis',
            'neural_market_regime', 'deep_sentiment_analysis',
            'attention_mechanisms_v2', 'sequence_patterns_v2',
            'neural_correlation', 'deep_volatility_analysis',
            'transformer_embeddings', 'lstm_attention_weights'
        ]
        
        strategies = {
            'neural_scalping': {
                'target_precision': 0.75,
                'target_pips': 2,
                'stop_loss_pips': 1,
                'timeframe': '1m',
                'indicators': ['transformer_attention', 'lstm_sequence_analysis']
            },
            'deep_day_trading': {
                'target_precision': 0.72,
                'target_pips': 25,
                'stop_loss_pips': 12,
                'timeframe': '5m',
                'indicators': ['neural_market_regime', 'deep_sentiment_analysis']
            }
        }
        
        optimizations = {
            'neural_architecture_v2': {
                'transformer_layers': 6,
                'attention_heads': 8,
                'lstm_layers': 3,
                'dropout': 0.15,
                'activation': 'gelu'
            },
            'deep_learning_optimization': {
                'learning_rate': 0.001,
                'batch_size': 64,
                'epochs': 100,
                'early_stopping': True,
                'patience': 10
            }
        }
        
        performance = {
            'accuracy': 0.82,
            'precision': 0.79,
            'recall': 0.85,
            'f1_score': 0.82,
            'sharpe_ratio': 2.5,
            'max_drawdown': 0.02,
            'profit_factor': 3.2
        }
        
        self.add_custom_model("Ultra_Model_v2", features, strategies, optimizations, performance)
    
    def add_hybrid_model_v2(self):
        """Añade Hybrid_Model v2 con mejoras"""
        print("🔄 Añadiendo Hybrid_Model v2...")
        
        features = [
            'ensemble_learning_v2', 'multi_model_correlation',
            'adaptive_regime_switching', 'dynamic_threshold_optimization',
            'cross_model_validation', 'ensemble_diversity_analysis',
            'meta_learning_features', 'stacking_optimization',
            'blending_techniques', 'voting_mechanisms'
        ]
        
        strategies = {
            'ensemble_adaptive': {
                'target_precision': 0.74,
                'target_pips': 12,
                'stop_loss_pips': 6,
                'timeframe': 'adaptive',
                'indicators': ['ensemble_learning_v2', 'adaptive_regime_switching']
            },
            'meta_learning_trading': {
                'target_precision': 0.70,
                'target_pips': 30,
                'stop_loss_pips': 15,
                'timeframe': 'adaptive',
                'indicators': ['meta_learning_features', 'cross_model_validation']
            }
        }
        
        optimizations = {
            'ensemble_optimization_v2': {
                'voting_method': 'weighted_adaptive',
                'base_models': ['rf', 'gbm', 'lstm', 'transformer'],
                'meta_learner': 'stacking_with_blending',
                'diversity_measure': 'correlation_based'
            },
            'adaptive_ensemble': {
                'dynamic_weighting': True,
                'regime_detection': True,
                'online_learning': True,
                'performance_tracking': True
            }
        }
        
        performance = {
            'accuracy': 0.80,
            'precision': 0.77,
            'recall': 0.83,
            'f1_score': 0.80,
            'sharpe_ratio': 2.8,
            'max_drawdown': 0.018,
            'profit_factor': 3.5
        }
        
        self.add_custom_model("Hybrid_Model_v2", features, strategies, optimizations, performance)
    
    def add_custom_model_interactive(self):
        """Añade un modelo personalizado de forma interactiva"""
        print("➕ AÑADIR MODELO PERSONALIZADO")
        print("=" * 40)
        
        # Obtener información del modelo
        model_name = input("Nombre del modelo: ").strip()
        
        print("\n📊 Features exitosas (separadas por coma):")
        features_input = input("Ej: rsi,macd,bollinger_bands,atr: ").strip()
        features = [f.strip() for f in features_input.split(',') if f.strip()]
        
        print("\n🎯 Estrategias (formato JSON):")
        print("Ejemplo: {'scalping': {'target_precision': 0.70, 'target_pips': 5}}")
        strategies_input = input("Estrategias: ").strip()
        try:
            strategies = json.loads(strategies_input)
        except:
            strategies = {}
        
        print("\n⚡ Optimizaciones (formato JSON):")
        print("Ejemplo: {'hyperparameter_tuning': {'n_estimators': 100}}")
        optimizations_input = input("Optimizaciones: ").strip()
        try:
            optimizations = json.loads(optimizations_input)
        except:
            optimizations = {}
        
        print("\n📈 Métricas de performance (formato JSON):")
        print("Ejemplo: {'accuracy': 0.75, 'precision': 0.72}")
        performance_input = input("Performance: ").strip()
        try:
            performance = json.loads(performance_input)
        except:
            performance = {}
        
        # Añadir modelo
        self.add_custom_model(model_name, features, strategies, optimizations, performance)
    
    def list_models(self):
        """Lista todos los modelos en el registro"""
        print("📋 MODELOS EN EL SISTEMA DE TRANSFER LEARNING")
        print("=" * 50)
        
        if not self.knowledge_registry:
            print("❌ No hay modelos registrados")
            return
        
        for model_name, model_data in self.knowledge_registry.items():
            status = model_data.get('status', 'unknown')
            features_count = len(model_data.get('successful_features', []))
            strategies_count = len(model_data.get('successful_strategies', {}))
            
            print(f"   • {model_name} ({status})")
            print(f"     Features: {features_count}, Estrategias: {strategies_count}")
            
            if 'performance_metrics' in model_data:
                perf = model_data['performance_metrics']
                accuracy = perf.get('accuracy', 0)
                print(f"     Accuracy: {accuracy:.3f}")
            print()
    
    def remove_model(self, model_name: str):
        """Elimina un modelo del registro"""
        if model_name in self.knowledge_registry:
            del self.knowledge_registry[model_name]
            self.save_knowledge_registry()
            print(f"✅ Modelo {model_name} eliminado")
        else:
            print(f"❌ Modelo {model_name} no encontrado")
    
    def update_model(self, model_name: str, updates: Dict[str, Any]):
        """Actualiza un modelo existente"""
        if model_name in self.knowledge_registry:
            self.knowledge_registry[model_name].update(updates)
            self.save_knowledge_registry()
            print(f"✅ Modelo {model_name} actualizado")
        else:
            print(f"❌ Modelo {model_name} no encontrado")

def main():
    """Función principal"""
    print("➕ SISTEMA DE AÑADIR MODELOS AL TRANSFER LEARNING")
    print("=" * 60)
    
    adder = ModelAdder()
    
    while True:
        print("\n🎯 ¿Qué quieres hacer?")
        print("1. 📋 Listar modelos existentes")
        print("2. 🧠 Añadir Brain_Max v2")
        print("3. 🚀 Añadir Ultra_Model v2")
        print("4. 🔄 Añadir Hybrid_Model v2")
        print("5. ➕ Añadir modelo personalizado")
        print("6. 🗑️ Eliminar modelo")
        print("7. 📊 Ver estadísticas")
        print("8. ❌ Salir")
        
        try:
            choice = input("\nSelecciona una opción (1-8): ").strip()
        except:
            choice = "8"
        
        if choice == "1":
            adder.list_models()
        
        elif choice == "2":
            adder.add_brain_max_v2()
        
        elif choice == "3":
            adder.add_ultra_model_v2()
        
        elif choice == "4":
            adder.add_hybrid_model_v2()
        
        elif choice == "5":
            adder.add_custom_model_interactive()
        
        elif choice == "6":
            model_name = input("Nombre del modelo a eliminar: ").strip()
            adder.remove_model(model_name)
        
        elif choice == "7":
            print(f"\n📊 Estadísticas:")
            print(f"   • Modelos registrados: {len(adder.knowledge_registry)}")
            total_features = sum(len(model.get('successful_features', [])) for model in adder.knowledge_registry.values())
            total_strategies = sum(len(model.get('successful_strategies', {})) for model in adder.knowledge_registry.values())
            print(f"   • Total features: {total_features}")
            print(f"   • Total estrategias: {total_strategies}")
        
        elif choice == "8":
            print("\n👋 ¡Hasta luego!")
            break
        
        else:
            print("❌ Opción no válida")

if __name__ == "__main__":
    main() 