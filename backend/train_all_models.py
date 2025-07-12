#!/usr/bin/env python3
# train_all_models.py - Script completo de entrenamiento de modelos
"""
Script para entrenar todos los modelos de IA antes de la integración con el frontend.

Este script:
1. Entrena modelos tradicionales (Random Forest, XGBoost, LSTM)
2. Entrena agentes de Reinforcement Learning (DQN, PPO)
3. Optimiza hiperparámetros
4. Configura auto-training system
5. Guarda todos los modelos entrenados

Uso:
    python train_all_models.py --symbols AAPL,MSFT,GOOGL --episodes 1000
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Añadir el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ai_models import AdvancedTradingAI
from rl_trading_agent import RLTradingSystem
from hyperparameter_optimization import AdvancedHyperparameterOptimizer
from auto_training_system import AutoTrainingManager
from main import DataCollector

class ModelTrainer:
    """Clase principal para entrenar todos los modelos"""
    
    def __init__(self, models_dir="models", logs_dir="logs"):
        self.models_dir = Path(models_dir)
        self.logs_dir = Path(logs_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Inicializar componentes
        self.data_collector = DataCollector()
        self.ai_system = AdvancedTradingAI()
        self.training_manager = AutoTrainingManager(models_dir=str(self.models_dir))
        
        # Resultados de entrenamiento
        self.training_results = {
            'traditional_models': {},
            'rl_models': {},
            'optimization': {},
            'auto_training': {}
        }
    
    def prepare_training_data(self, symbols, period="2y"):
        """Prepara datos de entrenamiento para todos los símbolos"""
        logger.info(f"📊 Preparando datos de entrenamiento para {len(symbols)} símbolos...")
        
        all_data = []
        for symbol in symbols:
            try:
                logger.info(f"  📈 Obteniendo datos para {symbol}...")
                data = self.data_collector.get_market_data(symbol, period)
                
                if not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)
                    logger.info(f"  ✅ {symbol}: {len(data)} registros")
                else:
                    logger.warning(f"  ⚠️ No se obtuvieron datos para {symbol}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error obteniendo datos para {symbol}: {e}")
        
        if not all_data:
            logger.error("❌ No se pudieron obtener datos para ningún símbolo")
            return None
        
        # Combinar todos los datos
        combined_data = pd.concat(all_data, ignore_index=True)
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)
        
        logger.info(f"✅ Datos preparados: {len(combined_data)} registros totales")
        return combined_data
    
    def train_traditional_models(self, data, symbols):
        """Entrena modelos tradicionales (Random Forest, LSTM)"""
        logger.info("🤖 Entrenando modelos tradicionales...")
        
        results = {}
        
        # Entrenar para cada símbolo individualmente
        for symbol in symbols:
            symbol_data = data[data['symbol'] == symbol]
            
            if len(symbol_data) < 100:
                logger.warning(f"⚠️ Datos insuficientes para {symbol}: {len(symbol_data)} registros")
                continue
            
            logger.info(f"  📊 Entrenando {symbol}...")
            
            try:
                # Entrenar clasificador de señales
                signal_success = self.ai_system.train_signal_classifier(symbol_data)
                
                # Entrenar LSTM
                lstm_success = self.ai_system.train_lstm_predictor(symbol_data)
                
                results[symbol] = {
                    'signal_classifier': signal_success,
                    'lstm_model': lstm_success,
                    'data_points': len(symbol_data)
                }
                
                if signal_success and lstm_success:
                    logger.info(f"  ✅ {symbol} entrenado exitosamente")
                else:
                    logger.warning(f"  ⚠️ {symbol}: Signal={signal_success}, LSTM={lstm_success}")
                    
            except Exception as e:
                logger.error(f"  ❌ Error entrenando {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        # Guardar modelos entrenados
        try:
            self.ai_system.save_models(str(self.models_dir))
            logger.info("💾 Modelos tradicionales guardados")
        except Exception as e:
            logger.error(f"❌ Error guardando modelos: {e}")
        
        self.training_results['traditional_models'] = results
        return results
    
    def train_rl_models(self, data, symbols, episodes=1000):
        """Entrena agentes de Reinforcement Learning"""
        logger.info("🎮 Entrenando agentes de Reinforcement Learning...")
        
        results = {}
        
        # Usar el primer símbolo para entrenamiento RL (en producción, combinar todos)
        primary_symbol = symbols[0]
        symbol_data = data[data['symbol'] == primary_symbol]
        
        if len(symbol_data) < 100:
            logger.error(f"❌ Datos insuficientes para RL: {len(symbol_data)} registros")
            return results
        
        # Entrenar DQN
        try:
            logger.info("  🧠 Entrenando agente DQN...")
            dqn_system = RLTradingSystem(
                data_source=self.data_collector,
                agent_type='DQN',
                model_save_path=str(self.models_dir / "rl_dqn.pth")
            )
            
            dqn_system.initialize_environment(symbol_data)
            dqn_system.train_agent(episodes=episodes)
            dqn_system.save_agent()
            
            results['DQN'] = {
                'success': True,
                'episodes': episodes,
                'performance': dqn_system.performance_metrics
            }
            logger.info("  ✅ DQN entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"  ❌ Error entrenando DQN: {e}")
            results['DQN'] = {'success': False, 'error': str(e)}
        
        # Entrenar PPO
        try:
            logger.info("  🧠 Entrenando agente PPO...")
            ppo_system = RLTradingSystem(
                data_source=self.data_collector,
                agent_type='PPO',
                model_save_path=str(self.models_dir / "rl_ppo.pth")
            )
            
            ppo_system.initialize_environment(symbol_data)
            ppo_system.train_agent(episodes=episodes)
            ppo_system.save_agent()
            
            results['PPO'] = {
                'success': True,
                'episodes': episodes,
                'performance': ppo_system.performance_metrics
            }
            logger.info("  ✅ PPO entrenado exitosamente")
            
        except Exception as e:
            logger.error(f"  ❌ Error entrenando PPO: {e}")
            results['PPO'] = {'success': False, 'error': str(e)}
        
        self.training_results['rl_models'] = results
        return results
    
    def optimize_hyperparameters(self, data, symbols):
        """Optimiza hiperparámetros de todos los modelos"""
        logger.info("🔧 Optimizando hiperparámetros...")
        
        try:
            # Preparar datos para optimización
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            
            # Usar el primer símbolo para optimización
            primary_symbol = symbols[0]
            symbol_data = data[data['symbol'] == primary_symbol]
            
            # Crear features para optimización
            features = self.ai_system.create_features(symbol_data)
            signals = self.ai_system.create_signals(symbol_data)
            
            # Preparar X, y
            feature_columns = [
                'rsi', 'macd', 'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_5', 'volatility_10', 'volume_ratio', 'high_low_ratio',
                'trend_5_20', 'trend_20_50', 'day_of_week', 'month'
            ]
            
            X = features[feature_columns].fillna(0)
            y = signals
            
            # Filtrar datos válidos
            valid_mask = ~X.isin([np.inf, -np.inf]).any(axis=1)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 100:
                logger.error("❌ Datos insuficientes para optimización")
                return {}
            
            # Inicializar optimizador
            optimizer = AdvancedHyperparameterOptimizer(
                data_source=self.data_collector,
                max_trials=50,  # Reducido para entrenamiento rápido
                cv_folds=3
            )
            
            # Optimizar todos los modelos
            optimized_params = optimizer.optimize_all_models(X, y)
            
            # Crear modelos optimizados
            optimized_models = optimizer.create_optimized_models(X, y)
            
            # Guardar resultados
            optimizer.save_optimization_results(str(self.models_dir / "optimization_results.pkl"))
            
            logger.info("✅ Optimización de hiperparámetros completada")
            
            self.training_results['optimization'] = {
                'success': True,
                'best_params': optimized_params,
                'models_created': len(optimized_models)
            }
            
            return optimized_params
            
        except Exception as e:
            logger.error(f"❌ Error en optimización: {e}")
            self.training_results['optimization'] = {'success': False, 'error': str(e)}
            return {}
    
    async def setup_auto_training(self, data, symbols):
        """Configura el sistema de auto-entrenamiento"""
        logger.info("🔄 Configurando sistema de auto-entrenamiento...")
        
        try:
            # Ejecutar primer auto-entrenamiento
            success = await self.training_manager.auto_retrain(data, symbols)
            
            if success:
                logger.info("✅ Auto-entrenamiento configurado exitosamente")
                self.training_results['auto_training'] = {'success': True}
            else:
                logger.warning("⚠️ Auto-entrenamiento falló")
                self.training_results['auto_training'] = {'success': False}
                
        except Exception as e:
            logger.error(f"❌ Error configurando auto-entrenamiento: {e}")
            self.training_results['auto_training'] = {'success': False, 'error': str(e)}
    
    def save_training_summary(self):
        """Guarda un resumen del entrenamiento"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'results': self.training_results,
            'models_dir': str(self.models_dir),
            'total_models_trained': sum([
                len([r for r in self.training_results['traditional_models'].values() if r.get('signal_classifier')]),
                len([r for r in self.training_results['rl_models'].values() if r.get('success')]),
                self.training_results['optimization'].get('models_created', 0)
            ])
        }
        
        summary_path = self.models_dir / "training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"📋 Resumen de entrenamiento guardado en {summary_path}")
    
    async def run_complete_training(self, symbols, episodes=1000, optimize=True):
        """Ejecuta el entrenamiento completo"""
        logger.info("🚀 Iniciando entrenamiento completo de modelos...")
        logger.info(f"📊 Símbolos: {symbols}")
        logger.info(f"🎮 Episodios RL: {episodes}")
        logger.info(f"🔧 Optimización: {'Sí' if optimize else 'No'}")
        
        start_time = datetime.now()
        
        try:
            # 1. Preparar datos
            data = self.prepare_training_data(symbols)
            if data is None:
                return False
            
            # 2. Entrenar modelos tradicionales
            traditional_results = self.train_traditional_models(data, symbols)
            
            # 3. Entrenar modelos RL
            rl_results = self.train_rl_models(data, symbols, episodes)
            
            # 4. Optimizar hiperparámetros (opcional)
            if optimize:
                optimization_results = self.optimize_hyperparameters(data, symbols)
            
            # 5. Configurar auto-training
            await self.setup_auto_training(data, symbols)
            
            # 6. Guardar resumen
            self.save_training_summary()
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            logger.info("🎉 Entrenamiento completo finalizado!")
            logger.info(f"⏱️ Duración total: {duration}")
            
            # Mostrar resumen
            self.print_training_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento completo: {e}")
            return False
    
    def print_training_summary(self):
        """Imprime un resumen del entrenamiento"""
        print("\n" + "="*60)
        print("📊 RESUMEN DE ENTRENAMIENTO")
        print("="*60)
        
        # Modelos tradicionales
        traditional = self.training_results['traditional_models']
        successful_traditional = sum(1 for r in traditional.values() 
                                   if r.get('signal_classifier') and r.get('lstm_model'))
        print(f"🤖 Modelos Tradicionales: {successful_traditional}/{len(traditional)} exitosos")
        
        # Modelos RL
        rl = self.training_results['rl_models']
        successful_rl = sum(1 for r in rl.values() if r.get('success'))
        print(f"🎮 Modelos RL: {successful_rl}/{len(rl)} exitosos")
        
        # Optimización
        optimization = self.training_results['optimization']
        if optimization.get('success'):
            print(f"🔧 Optimización: ✅ Completada")
        else:
            print(f"🔧 Optimización: ❌ Falló")
        
        # Auto-training
        auto_training = self.training_results['auto_training']
        if auto_training.get('success'):
            print(f"🔄 Auto-training: ✅ Configurado")
        else:
            print(f"🔄 Auto-training: ❌ Falló")
        
        print("="*60)
        print(f"📁 Modelos guardados en: {self.models_dir}")
        print(f"📋 Resumen detallado: {self.models_dir}/training_summary.json")
        print("="*60)

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description='Entrenar todos los modelos de IA')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL,TSLA,NVDA',
                       help='Símbolos para entrenamiento (separados por coma)')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Número de episodios para RL')
    parser.add_argument('--no-optimize', action='store_true',
                       help='Saltar optimización de hiperparámetros')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directorio para guardar modelos')
    
    args = parser.parse_args()
    
    # Parsear símbolos
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Crear trainer
    trainer = ModelTrainer(models_dir=args.models_dir)
    
    # Ejecutar entrenamiento
    success = asyncio.run(trainer.run_complete_training(
        symbols=symbols,
        episodes=args.episodes,
        optimize=not args.no_optimize
    ))
    
    if success:
        print("\n🎉 ¡Entrenamiento completado exitosamente!")
        print("✅ Ahora puedes proceder con la integración al frontend")
    else:
        print("\n❌ El entrenamiento falló. Revisa los logs para más detalles.")
        sys.exit(1)

if __name__ == "__main__":
    main() 