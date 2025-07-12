#!/usr/bin/env python3
# verify_models.py - Script para verificar modelos entrenados
"""
Script para verificar que todos los modelos se entrenaron correctamente.

Este script:
1. Verifica que los modelos existen en el directorio models/
2. Carga los modelos y verifica que funcionan
3. Ejecuta predicciones de prueba
4. Muestra métricas de rendimiento
5. Genera un reporte de verificación

Uso:
    python verify_models.py --models-dir models
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_models import AdvancedTradingAI
from rl_trading_agent import RLTradingSystem
from main import DataCollector

class ModelVerifier:
    """Clase para verificar modelos entrenados"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.verification_results = {
            'traditional_models': {},
            'rl_models': {},
            'files_exist': {},
            'predictions': {},
            'overall_status': 'UNKNOWN'
        }
    
    def check_files_exist(self):
        """Verifica que los archivos de modelos existen"""
        logger.info("📁 Verificando archivos de modelos...")
        
        expected_files = [
            'signal_classifier.pkl',
            'scaler.pkl', 
            'lstm_model.h5',
            'rl_dqn.pth',
            'rl_ppo.pth',
            'optimization_results.pkl',
            'training_summary.json'
        ]
        
        for filename in expected_files:
            filepath = self.models_dir / filename
            exists = filepath.exists()
            self.verification_results['files_exist'][filename] = exists
            
            if exists:
                size = filepath.stat().st_size
                logger.info(f"  ✅ {filename} ({size} bytes)")
            else:
                logger.warning(f"  ❌ {filename} - NO ENCONTRADO")
        
        return all(self.verification_results['files_exist'].values())
    
    def verify_traditional_models(self):
        """Verifica modelos tradicionales"""
        logger.info("🤖 Verificando modelos tradicionales...")
        
        try:
            # Cargar AI system
            ai_system = AdvancedTradingAI()
            
            # Intentar cargar modelos
            try:
                ai_system.load_models(str(self.models_dir))
                logger.info("  ✅ Modelos tradicionales cargados exitosamente")
                
                # Verificar que está entrenado
                if ai_system.is_trained:
                    logger.info("  ✅ Sistema marcado como entrenado")
                    
                    # Obtener datos de prueba
                    collector = DataCollector()
                    test_data = collector.get_market_data('AAPL', '1mo')
                    
                    if not test_data.empty:
                        # Crear features
                        features = ai_system.create_features(test_data)
                        
                        # Hacer predicción de prueba
                        if len(features) > 0:
                            # Preparar datos para predicción
                            feature_columns = [
                                'rsi', 'macd', 'momentum_5', 'momentum_10', 'momentum_20',
                                'volatility_5', 'volatility_10', 'volume_ratio', 'high_low_ratio',
                                'trend_5_20', 'trend_20_50', 'day_of_week', 'month'
                            ]
                            
                            X = features[feature_columns].fillna(0).iloc[-1:2]
                            
                            if not X.empty:
                                # Escalar features
                                X_scaled = ai_system.scaler.transform(X)
                                
                                # Hacer predicción
                                signal_pred = ai_system.signal_classifier.predict(X_scaled)
                                signal_proba = ai_system.signal_classifier.predict_proba(X_scaled)
                                
                                logger.info(f"  📊 Predicción de señal: {signal_pred[0]}")
                                logger.info(f"  📊 Probabilidades: {signal_proba[0]}")
                                
                                self.verification_results['traditional_models'] = {
                                    'loaded': True,
                                    'trained': ai_system.is_trained,
                                    'prediction_works': True,
                                    'last_prediction': {
                                        'signal': int(signal_pred[0]),
                                        'probabilities': signal_proba[0].tolist()
                                    }
                                }
                            else:
                                logger.warning("  ⚠️ No hay datos suficientes para predicción")
                                self.verification_results['traditional_models'] = {
                                    'loaded': True,
                                    'trained': ai_system.is_trained,
                                    'prediction_works': False
                                }
                        else:
                            logger.warning("  ⚠️ No se pudieron crear features")
                            self.verification_results['traditional_models'] = {
                                'loaded': True,
                                'trained': ai_system.is_trained,
                                'prediction_works': False
                            }
                    else:
                        logger.warning("  ⚠️ No se pudieron obtener datos de prueba")
                        self.verification_results['traditional_models'] = {
                            'loaded': True,
                            'trained': ai_system.is_trained,
                            'prediction_works': False
                        }
                else:
                    logger.warning("  ⚠️ Sistema no marcado como entrenado")
                    self.verification_results['traditional_models'] = {
                        'loaded': True,
                        'trained': False,
                        'prediction_works': False
                    }
                    
            except Exception as e:
                logger.error(f"  ❌ Error cargando modelos tradicionales: {e}")
                self.verification_results['traditional_models'] = {
                    'loaded': False,
                    'error': str(e)
                }
                
        except Exception as e:
            logger.error(f"  ❌ Error verificando modelos tradicionales: {e}")
            self.verification_results['traditional_models'] = {
                'loaded': False,
                'error': str(e)
            }
    
    def verify_rl_models(self):
        """Verifica modelos de Reinforcement Learning"""
        logger.info("🎮 Verificando modelos de RL...")
        
        rl_results = {}
        
        for agent_type in ['DQN', 'PPO']:
            try:
                logger.info(f"  🧠 Verificando {agent_type}...")
                
                # Crear sistema RL
                rl_system = RLTradingSystem(
                    data_source=DataCollector(),
                    agent_type=agent_type,
                    model_save_path=str(self.models_dir / f"rl_{agent_type.lower()}.pth")
                )
                
                # Intentar cargar agente
                if rl_system.load_agent():
                    logger.info(f"    ✅ {agent_type} cargado exitosamente")
                    
                    # Verificar que el agente puede hacer predicciones
                    if rl_system.agent:
                        # Crear estado de prueba
                        test_state = np.random.randn(70)  # 70 dimensiones como en el código
                        
                        if agent_type == 'DQN':
                            # Hacer predicción DQN
                            with torch.no_grad():
                                q_values = rl_system.agent.q_network(torch.FloatTensor(test_state).unsqueeze(0))
                                action = q_values.argmax().item()
                                confidence = torch.softmax(q_values, dim=1).max().item()
                        else:  # PPO
                            # Hacer predicción PPO
                            with torch.no_grad():
                                action_probs = rl_system.agent.policy_network(torch.FloatTensor(test_state).unsqueeze(0))
                                action = action_probs.argmax().item()
                                confidence = action_probs.max().item()
                        
                        logger.info(f"    📊 Predicción {agent_type}: Acción {action}, Confianza {confidence:.3f}")
                        
                        rl_results[agent_type] = {
                            'loaded': True,
                            'prediction_works': True,
                            'last_prediction': {
                                'action': action,
                                'confidence': confidence
                            }
                        }
                    else:
                        logger.warning(f"    ⚠️ Agente {agent_type} no inicializado")
                        rl_results[agent_type] = {
                            'loaded': True,
                            'prediction_works': False
                        }
                else:
                    logger.warning(f"    ⚠️ No se pudo cargar {agent_type}")
                    rl_results[agent_type] = {
                        'loaded': False,
                        'prediction_works': False
                    }
                    
            except Exception as e:
                logger.error(f"    ❌ Error verificando {agent_type}: {e}")
                rl_results[agent_type] = {
                    'loaded': False,
                    'error': str(e)
                }
        
        self.verification_results['rl_models'] = rl_results
    
    def verify_optimization_results(self):
        """Verifica resultados de optimización"""
        logger.info("🔧 Verificando resultados de optimización...")
        
        optimization_file = self.models_dir / "optimization_results.pkl"
        
        if optimization_file.exists():
            try:
                import joblib
                results = joblib.load(optimization_file)
                
                if 'best_params' in results:
                    logger.info("  ✅ Resultados de optimización cargados")
                    logger.info(f"  📊 Modelos optimizados: {list(results['best_params'].keys())}")
                    
                    self.verification_results['optimization'] = {
                        'exists': True,
                        'models_optimized': list(results['best_params'].keys()),
                        'timestamp': results.get('timestamp', 'Unknown')
                    }
                else:
                    logger.warning("  ⚠️ Archivo de optimización no tiene formato esperado")
                    self.verification_results['optimization'] = {
                        'exists': True,
                        'format_error': True
                    }
            except Exception as e:
                logger.error(f"  ❌ Error cargando optimización: {e}")
                self.verification_results['optimization'] = {
                    'exists': True,
                    'load_error': str(e)
                }
        else:
            logger.warning("  ⚠️ Archivo de optimización no encontrado")
            self.verification_results['optimization'] = {
                'exists': False
            }
    
    def verify_training_summary(self):
        """Verifica resumen de entrenamiento"""
        logger.info("📋 Verificando resumen de entrenamiento...")
        
        summary_file = self.models_dir / "training_summary.json"
        
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                
                logger.info("  ✅ Resumen de entrenamiento cargado")
                logger.info(f"  📊 Total modelos entrenados: {summary.get('total_models_trained', 'Unknown')}")
                logger.info(f"  ⏰ Timestamp: {summary.get('timestamp', 'Unknown')}")
                
                self.verification_results['training_summary'] = {
                    'exists': True,
                    'total_models': summary.get('total_models_trained', 0),
                    'timestamp': summary.get('timestamp', 'Unknown')
                }
                
            except Exception as e:
                logger.error(f"  ❌ Error cargando resumen: {e}")
                self.verification_results['training_summary'] = {
                    'exists': True,
                    'load_error': str(e)
                }
        else:
            logger.warning("  ⚠️ Resumen de entrenamiento no encontrado")
            self.verification_results['training_summary'] = {
                'exists': False
            }
    
    def run_complete_verification(self):
        """Ejecuta verificación completa"""
        logger.info("🔍 Iniciando verificación completa de modelos...")
        
        # 1. Verificar archivos
        files_ok = self.check_files_exist()
        
        # 2. Verificar modelos tradicionales
        self.verify_traditional_models()
        
        # 3. Verificar modelos RL
        self.verify_rl_models()
        
        # 4. Verificar optimización
        self.verify_optimization_results()
        
        # 5. Verificar resumen
        self.verify_training_summary()
        
        # Determinar estado general
        traditional_ok = self.verification_results['traditional_models'].get('loaded', False)
        rl_ok = any(self.verification_results['rl_models'].get(agent, {}).get('loaded', False) 
                   for agent in ['DQN', 'PPO'])
        
        if files_ok and traditional_ok and rl_ok:
            self.verification_results['overall_status'] = 'READY'
        elif files_ok and traditional_ok:
            self.verification_results['overall_status'] = 'PARTIAL'
        else:
            self.verification_results['overall_status'] = 'FAILED'
        
        # Mostrar resumen
        self.print_verification_summary()
        
        return self.verification_results['overall_status'] == 'READY'
    
    def print_verification_summary(self):
        """Imprime resumen de verificación"""
        print("\n" + "="*60)
        print("🔍 RESUMEN DE VERIFICACIÓN")
        print("="*60)
        
        # Estado general
        status = self.verification_results['overall_status']
        if status == 'READY':
            print("✅ ESTADO: LISTO PARA INTEGRACIÓN")
        elif status == 'PARTIAL':
            print("⚠️ ESTADO: PARCIALMENTE LISTO")
        else:
            print("❌ ESTADO: NO LISTO")
        
        print()
        
        # Archivos
        files = self.verification_results['files_exist']
        files_ok = sum(files.values())
        files_total = len(files)
        print(f"📁 Archivos: {files_ok}/{files_total} encontrados")
        
        # Modelos tradicionales
        traditional = self.verification_results['traditional_models']
        if traditional.get('loaded'):
            print("🤖 Modelos Tradicionales: ✅ Cargados")
            if traditional.get('prediction_works'):
                print("   📊 Predicciones: ✅ Funcionando")
            else:
                print("   📊 Predicciones: ⚠️ No probadas")
        else:
            print("🤖 Modelos Tradicionales: ❌ No cargados")
        
        # Modelos RL
        rl = self.verification_results['rl_models']
        rl_loaded = sum(1 for agent in rl.values() if agent.get('loaded'))
        rl_total = len(rl)
        print(f"🎮 Modelos RL: {rl_loaded}/{rl_total} cargados")
        
        # Optimización
        optimization = self.verification_results.get('optimization', {})
        if optimization.get('exists'):
            print("🔧 Optimización: ✅ Resultados encontrados")
        else:
            print("🔧 Optimización: ⚠️ No encontrada")
        
        # Resumen de entrenamiento
        summary = self.verification_results.get('training_summary', {})
        if summary.get('exists'):
            print("📋 Resumen: ✅ Encontrado")
        else:
            print("📋 Resumen: ⚠️ No encontrado")
        
        print("="*60)
        
        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        
        if status == 'READY':
            print("✅ Todos los modelos están listos para integración")
            print("✅ Puedes proceder con las mejoras del frontend")
        elif status == 'PARTIAL':
            print("⚠️ Algunos modelos están listos, otros necesitan entrenamiento")
            print("⚠️ Considera ejecutar train_all_models.py nuevamente")
        else:
            print("❌ Los modelos no están listos")
            print("❌ Ejecuta train_all_models.py primero")
        
        print("="*60)

def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Verificar modelos entrenados')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directorio de modelos')
    
    args = parser.parse_args()
    
    # Crear verificador
    verifier = ModelVerifier(models_dir=args.models_dir)
    
    # Ejecutar verificación
    success = verifier.run_complete_verification()
    
    if success:
        print("\n🎉 ¡Verificación exitosa!")
        print("✅ Los modelos están listos para integración")
        sys.exit(0)
    else:
        print("\n❌ Verificación falló")
        print("❌ Los modelos necesitan entrenamiento")
        sys.exit(1)

if __name__ == "__main__":
    main() 