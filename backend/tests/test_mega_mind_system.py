#!/usr/bin/env python3
"""
Test Script para el Sistema MegaMind
====================================
Prueba completa de todas las funcionalidades del sistema de Cerebros Colaborativos
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Agregar el directorio src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(current_dir)
src_dir = os.path.join(backend_dir, 'src')
services_dir = os.path.join(backend_dir, 'services')

sys.path.insert(0, src_dir)
sys.path.insert(0, services_dir)
sys.path.insert(0, backend_dir)

print(f"Python path configurado:")
print(f"  - Current dir: {current_dir}")
print(f"  - Backend dir: {backend_dir}")
print(f"  - Src dir: {src_dir}")
print(f"  - Services dir: {services_dir}")
print(f"  - Sys.path: {sys.path[:3]}")

try:
    from services.mega_mind_service import MegaMindService
    print("✅ Importación exitosa de MegaMindService")
except ImportError as e:
    print(f"❌ Error importando MegaMindService: {e}")
    print("Intentando importación alternativa...")
    try:
        # Intentar importación directa
        import mega_mind_service
        MegaMindService = mega_mind_service.MegaMindService
        print("✅ Importación alternativa exitosa")
    except ImportError as e2:
        print(f"❌ Error en importación alternativa: {e2}")
        sys.exit(1)

class MegaMindTester:
    """Clase para probar todas las funcionalidades de MegaMind"""
    
    def __init__(self):
        try:
            self.service = MegaMindService()
            self.test_results = []
            print("✅ MegaMindService inicializado correctamente")
        except Exception as e:
            print(f"❌ Error inicializando MegaMindService: {e}")
            raise
    
    async def run_all_tests(self):
        """Ejecutar todas las pruebas"""
        print("🧠 Iniciando pruebas del Sistema MegaMind...")
        print("=" * 60)
        
        # Pruebas básicas
        await self.test_basic_functionality()
        
        # Pruebas de predicciones
        await self.test_predictions()
        
        # Pruebas de colaboración
        await self.test_collaboration()
        
        # Pruebas de arena
        await self.test_arena()
        
        # Pruebas de evolución
        await self.test_evolution()
        
        # Pruebas de orquestación
        await self.test_orchestration()
        
        # Pruebas de configuración
        await self.test_configuration()
        
        # Pruebas de entrenamiento
        await self.test_training()
        
        # Pruebas de rendimiento
        await self.test_performance()
        
        # Mostrar resultados
        self.print_results()
    
    async def test_basic_functionality(self):
        """Probar funcionalidad básica"""
        print("\n📋 Probando funcionalidad básica...")
        
        try:
            # Verificar que el servicio se inicializó correctamente
            assert self.service is not None
            assert hasattr(self.service, 'brain_collaboration')
            assert hasattr(self.service, 'brain_fusion')
            assert hasattr(self.service, 'brain_arena')
            assert hasattr(self.service, 'brain_evolution')
            assert hasattr(self.service, 'brain_orchestration')
            assert hasattr(self.service, 'brain_gamification')
            assert hasattr(self.service, 'brain_personalization')
            
            # Verificar estados de cerebros
            assert 'brain_max' in self.service.brain_states
            assert 'brain_ultra' in self.service.brain_states
            assert 'brain_predictor' in self.service.brain_states
            
            # Verificar configuraciones
            assert 'fusion_weights' in self.service.__dict__
            assert 'collaboration_config' in self.service.__dict__
            assert 'evolution_config' in self.service.__dict__
            assert 'gamification_config' in self.service.__dict__
            
            self.test_results.append({
                'test': 'basic_functionality',
                'status': 'PASS',
                'message': 'Funcionalidad básica correcta'
            })
            print("✅ Funcionalidad básica: PASS")
            
        except Exception as e:
            self.test_results.append({
                'test': 'basic_functionality',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Funcionalidad básica: FAIL - {str(e)}")
    
    async def test_predictions(self):
        """Probar sistema de predicciones"""
        print("\n🎯 Probando sistema de predicciones...")
        
        try:
            # Probar predicciones MEGA MIND
            predictions = await self.service.get_mega_mind_predictions(
                pair="EURUSD",
                style="day_trading",
                limit=5
            )
            
            # Verificar estructura de predicciones
            assert len(predictions) > 0
            for pred in predictions:
                assert 'pair' in pred
                assert 'direction' in pred
                assert 'confidence' in pred
                assert 'target_price' in pred
                assert 'brain_type' in pred
                assert pred['brain_type'] == 'mega_mind'
                assert 'fusion_method' in pred
                assert 'collaboration_score' in pred
            
            self.test_results.append({
                'test': 'predictions',
                'status': 'PASS',
                'message': f'Generadas {len(predictions)} predicciones correctamente'
            })
            print(f"✅ Predicciones: PASS - {len(predictions)} predicciones generadas")
            
        except Exception as e:
            self.test_results.append({
                'test': 'predictions',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Predicciones: FAIL - {str(e)}")
    
    async def test_collaboration(self):
        """Probar sistema de colaboración"""
        print("\n🤝 Probando sistema de colaboración...")
        
        try:
            # Probar análisis de colaboración
            collaboration = await self.service.get_brain_collaboration("EURUSD")
            
            # Verificar estructura
            assert 'pair' in collaboration
            assert 'collaboration_score' in collaboration
            assert 'consensus_level' in collaboration
            assert 'brain_synergy' in collaboration
            assert 'conflict_resolution' in collaboration
            assert 'performance_metrics' in collaboration
            
            # Verificar valores
            assert 0 <= collaboration['collaboration_score'] <= 1
            assert 0 <= collaboration['consensus_level'] <= 1
            
            self.test_results.append({
                'test': 'collaboration',
                'status': 'PASS',
                'message': f'Colaboración score: {collaboration["collaboration_score"]:.2f}'
            })
            print(f"✅ Colaboración: PASS - Score: {collaboration['collaboration_score']:.2f}")
            
        except Exception as e:
            self.test_results.append({
                'test': 'collaboration',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Colaboración: FAIL - {str(e)}")
    
    async def test_arena(self):
        """Probar sistema de arena"""
        print("\n🏆 Probando sistema de arena...")
        
        try:
            # Probar resultados de arena
            arena_results = await self.service.get_brain_arena_results("EURUSD")
            
            # Verificar estructura
            assert 'pair' in arena_results
            assert 'competition_round' in arena_results
            assert 'arena_results' in arena_results
            assert 'champion' in arena_results
            assert 'overall_performance' in arena_results
            
            # Verificar resultados de cerebros
            brain_results = arena_results['arena_results']
            assert 'brain_max' in brain_results
            assert 'brain_ultra' in brain_results
            assert 'brain_predictor' in brain_results
            
            self.test_results.append({
                'test': 'arena',
                'status': 'PASS',
                'message': f'Campeón: {arena_results["champion"]}'
            })
            print(f"✅ Arena: PASS - Campeón: {arena_results['champion']}")
            
        except Exception as e:
            self.test_results.append({
                'test': 'arena',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Arena: FAIL - {str(e)}")
    
    async def test_evolution(self):
        """Probar sistema de evolución"""
        print("\n🧬 Probando sistema de evolución...")
        
        try:
            # Probar estado de evolución
            evolution_status = await self.service.get_brain_evolution_status()
            
            # Verificar estructura
            assert 'evolution_phase' in evolution_status
            assert 'generation' in evolution_status
            assert 'improvement_rate' in evolution_status
            assert 'evolution_metrics' in evolution_status
            assert 'next_evolution_trigger' in evolution_status
            
            # Verificar métricas
            metrics = evolution_status['evolution_metrics']
            assert 'fitness_scores' in metrics
            assert 'mutation_count' in metrics
            assert 'crossover_count' in metrics
            
            self.test_results.append({
                'test': 'evolution',
                'status': 'PASS',
                'message': f'Generación: {evolution_status["generation"]}'
            })
            print(f"✅ Evolución: PASS - Generación: {evolution_status['generation']}")
            
        except Exception as e:
            self.test_results.append({
                'test': 'evolution',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Evolución: FAIL - {str(e)}")
    
    async def test_orchestration(self):
        """Probar sistema de orquestación"""
        print("\n🎼 Probando sistema de orquestación...")
        
        try:
            # Probar estado de orquestación
            orchestration_status = await self.service.get_brain_orchestration_status()
            
            # Verificar estructura
            assert 'orchestration_mode' in orchestration_status
            assert 'coordination_score' in orchestration_status
            assert 'orchestration_metrics' in orchestration_status
            assert 'active_strategies' in orchestration_status
            
            # Verificar valores
            assert 0 <= orchestration_status['coordination_score'] <= 1
            assert orchestration_status['active_strategies'] > 0
            
            self.test_results.append({
                'test': 'orchestration',
                'status': 'PASS',
                'message': f'Modo: {orchestration_status["orchestration_mode"]}'
            })
            print(f"✅ Orquestación: PASS - Modo: {orchestration_status['orchestration_mode']}")
            
        except Exception as e:
            self.test_results.append({
                'test': 'orchestration',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Orquestación: FAIL - {str(e)}")
    
    async def test_configuration(self):
        """Probar configuración de cerebros"""
        print("\n⚙️ Probando configuración de cerebros...")
        
        try:
            # Configuración de prueba
            test_config = {
                'trading_params': {
                    'stop_loss': 0.02,
                    'take_profit': 0.04,
                    'lot_size': 0.1,
                    'max_drawdown': 0.15
                },
                'market_preferences': {
                    'markets': ['EURUSD', 'GBPUSD'],
                    'timeframes': ['15m', '1h'],
                    'risk_profile': 'moderate'
                },
                'specializations': {
                    'indicators': ['RSI', 'MACD'],
                    'strategies': ['trend_following'],
                    'custom_indicators': []
                },
                'consensus_weight': 0.33
            }
            
            # Configurar cerebro
            result = await self.service.configure_brain('brain_max', test_config)
            
            # Verificar resultado
            assert result['config_applied'] == True
            assert result['brain_type'] == 'brain_max'
            assert 'config_version' in result
            
            self.test_results.append({
                'test': 'configuration',
                'status': 'PASS',
                'message': f'Configuración aplicada: v{result["config_version"]}'
            })
            print(f"✅ Configuración: PASS - Versión: {result['config_version']}")
            
        except Exception as e:
            self.test_results.append({
                'test': 'configuration',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Configuración: FAIL - {str(e)}")
    
    async def test_training(self):
        """Probar entrenamiento de cerebros"""
        print("\n🎓 Probando entrenamiento de cerebros...")
        
        try:
            # Datos de entrenamiento de prueba
            training_data = {
                'training_data': {
                    'historical_data': 'mock_data',
                    'market_conditions': 'mock_conditions',
                    'performance_metrics': 'mock_metrics'
                },
                'training_params': {
                    'epochs': 50,
                    'learning_rate': 0.001,
                    'batch_size': 32
                }
            }
            
            # Entrenar cerebro
            result = await self.service.train_brain('brain_max', training_data)
            
            # Verificar resultado
            assert result['training_completed'] == True
            assert result['brain_type'] == 'brain_max'
            assert 'training_metrics' in result
            assert 'new_accuracy' in result
            
            self.test_results.append({
                'test': 'training',
                'status': 'PASS',
                'message': f'Entrenamiento completado - Nueva precisión: {result["new_accuracy"]:.1f}%'
            })
            print(f"✅ Entrenamiento: PASS - Nueva precisión: {result['new_accuracy']:.1f}%")
            
        except Exception as e:
            self.test_results.append({
                'test': 'training',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Entrenamiento: FAIL - {str(e)}")
    
    async def test_performance(self):
        """Probar métricas de rendimiento"""
        print("\n📊 Probando métricas de rendimiento...")
        
        try:
            # Obtener métricas de rendimiento
            performance = await self.service.get_mega_mind_performance()
            
            # Verificar métricas básicas
            assert 'overall_accuracy' in performance
            assert 'prediction_success_rate' in performance
            assert 'risk_adjusted_returns' in performance
            assert 'sharpe_ratio' in performance
            assert 'win_rate' in performance
            
            # Verificar métricas de cerebros
            assert 'brain_levels' in performance
            assert 'achievements' in performance
            
            # Verificar valores
            assert 0 <= performance['overall_accuracy'] <= 100
            assert 0 <= performance['win_rate'] <= 1
            
            self.test_results.append({
                'test': 'performance',
                'status': 'PASS',
                'message': f'Precisión general: {performance["overall_accuracy"]:.1f}%'
            })
            print(f"✅ Rendimiento: PASS - Precisión: {performance['overall_accuracy']:.1f}%")
            
        except Exception as e:
            self.test_results.append({
                'test': 'performance',
                'status': 'FAIL',
                'message': f'Error: {str(e)}'
            })
            print(f"❌ Rendimiento: FAIL - {str(e)}")
    
    def print_results(self):
        """Mostrar resultados de las pruebas"""
        print("\n" + "=" * 60)
        print("📋 RESULTADOS DE PRUEBAS MEGAMIND")
        print("=" * 60)
        
        passed = 0
        failed = 0
        
        for result in self.test_results:
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{status_icon} {result['test']}: {result['status']}")
            print(f"   {result['message']}")
            print()
            
            if result['status'] == 'PASS':
                passed += 1
            else:
                failed += 1
        
        print("=" * 60)
        print(f"📊 RESUMEN: {passed} PASS, {failed} FAIL")
        
        if failed == 0:
            print("🎉 ¡TODAS LAS PRUEBAS PASARON! El sistema MegaMind está funcionando correctamente.")
        else:
            print("⚠️  Algunas pruebas fallaron. Revisar los errores arriba.")
        
        print("=" * 60)
        
        # Guardar resultados en archivo
        self.save_results()
    
    def save_results(self):
        """Guardar resultados en archivo JSON"""
        timestamp = datetime.now().isoformat()
        results_data = {
            'timestamp': timestamp,
            'total_tests': len(self.test_results),
            'passed': len([r for r in self.test_results if r['status'] == 'PASS']),
            'failed': len([r for r in self.test_results if r['status'] == 'FAIL']),
            'results': self.test_results
        }
        
        filename = f"megamind_test_results_{timestamp[:19].replace(':', '-')}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados guardados en: {filename}")
        except Exception as e:
            print(f"⚠️  Error guardando resultados: {str(e)}")

async def main():
    """Función principal"""
    print("🧠 MEGAMIND SYSTEM TESTER")
    print("Sistema de Cerebros Colaborativos - Pruebas Completas")
    print("=" * 60)
    
    try:
        tester = MegaMindTester()
        await tester.run_all_tests()
    except Exception as e:
        print(f"❌ Error crítico: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 