"""
Servicio de Estrategia de Scalping para Trading Automático
Integra la estrategia scalping_24h.py con el sistema de trading automático
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

# Agregar el path para importar los modelos
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'strategies'))

try:
    from strategies.scalping_24h import Scalping24HourSimulator
    from Modelo_AI_Ultra import EURUSDMultiStrategyAI, generate_eurusd_data
except ImportError as e:
    print(f"⚠️ Error importando módulos de estrategia: {e}")
    # Fallback para desarrollo
    class Scalping24HourSimulator:
        def __init__(self):
            self.initial_balance = 10000
            self.current_balance = self.initial_balance
            self.positions = []
            self.closed_trades = []
            self.signals_history = []
            self.total_signals = 0
            self.executed_trades = 0
            self.winning_trades = 0
            self.losing_trades = 0

class ScalpingStrategyService:
    """
    Servicio para integrar la estrategia de scalping con el trading automático
    """
    
    def __init__(self):
        self.simulator = Scalping24HourSimulator()
        self.active_strategies: Dict[str, Dict] = {}
        self.strategy_configs = {
            'scalping_eurusd': {
                'name': 'Scalping EURUSD 24h',
                'description': 'Estrategia de scalping optimizada para EURUSD en sesiones de 24 horas',
                'pair': 'EURUSD',
                'timeframe': '1M',
                'parameters': {
                    'position_size': 0.2,
                    'stop_loss_pips': 2,
                    'take_profit_pips': 4,
                    'min_confidence': 80,
                    'min_volatility': 0.0003,
                    'max_spread': 0.0003,
                    'risk_per_trade': 0.015
                },
                'filters': {
                    'sessions': ['london', 'new_york', 'tokyo'],
                    'min_volume': 1000,
                    'max_daily_trades': 50
                }
            },
            'scalping_gbpusd': {
                'name': 'Scalping GBPUSD 24h',
                'description': 'Estrategia de scalping optimizada para GBPUSD',
                'pair': 'GBPUSD',
                'timeframe': '1M',
                'parameters': {
                    'position_size': 0.15,
                    'stop_loss_pips': 3,
                    'take_profit_pips': 6,
                    'min_confidence': 85,
                    'min_volatility': 0.0004,
                    'max_spread': 0.0004,
                    'risk_per_trade': 0.012
                },
                'filters': {
                    'sessions': ['london', 'new_york'],
                    'min_volume': 1200,
                    'max_daily_trades': 40
                }
            }
        }
    
    async def create_strategy(self, strategy_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea una nueva estrategia de scalping
        """
        try:
            if strategy_type not in self.strategy_configs:
                raise ValueError(f"Tipo de estrategia no válido: {strategy_type}")
            
            strategy_id = f"{strategy_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Combinar configuración base con personalizada
            base_config = self.strategy_configs[strategy_type].copy()
            base_config['parameters'].update(config.get('parameters', {}))
            base_config['filters'].update(config.get('filters', {}))
            
            strategy = {
                'id': strategy_id,
                'type': strategy_type,
                'name': base_config['name'],
                'description': base_config['description'],
                'pair': base_config['pair'],
                'timeframe': base_config['timeframe'],
                'parameters': base_config['parameters'],
                'filters': base_config['filters'],
                'status': 'created',
                'created_at': datetime.now().isoformat(),
                'last_signal': None,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_pnl': 0.0,
                'current_price': 0.0,
                'balance': config.get('initial_balance', 10000),
                'equity': config.get('initial_balance', 10000)
            }
            
            self.active_strategies[strategy_id] = strategy
            
            return {
                'success': True,
                'strategy_id': strategy_id,
                'strategy': strategy,
                'message': f'Estrategia {strategy_type} creada exitosamente'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error creando estrategia: {str(e)}'
            }
    
    async def start_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Inicia una estrategia de scalping
        """
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            strategy = self.active_strategies[strategy_id]
            strategy['status'] = 'active'
            strategy['started_at'] = datetime.now().isoformat()
            
            # Inicializar simulador para esta estrategia
            self.simulator.initial_balance = strategy['balance']
            self.simulator.current_balance = strategy['balance']
            self.simulator.positions = []
            self.simulator.closed_trades = []
            
            # Configurar parámetros del simulador
            params = strategy['parameters']
            self.simulator.position_size = params['position_size']
            self.simulator.stop_loss_pips = params['stop_loss_pips']
            self.simulator.take_profit_pips = params['take_profit_pips']
            self.simulator.min_confidence = params['min_confidence']
            self.simulator.min_volatility = params['min_volatility']
            self.simulator.max_spread = params['max_spread']
            
            return {
                'success': True,
                'strategy_id': strategy_id,
                'message': f'Estrategia {strategy["name"]} iniciada exitosamente'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error iniciando estrategia: {str(e)}'
            }
    
    async def stop_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Detiene una estrategia de scalping
        """
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            strategy = self.active_strategies[strategy_id]
            strategy['status'] = 'stopped'
            strategy['stopped_at'] = datetime.now().isoformat()
            
            # Cerrar posiciones abiertas
            if self.simulator.positions:
                # Simular cierre de posiciones
                for position in self.simulator.positions:
                    position['exit_time'] = datetime.now()
                    position['exit_price'] = strategy['current_price']
                    position['status'] = 'closed'
                    self.simulator.closed_trades.append(position)
                
                self.simulator.positions = []
            
            return {
                'success': True,
                'strategy_id': strategy_id,
                'message': f'Estrategia {strategy["name"]} detenida exitosamente'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error deteniendo estrategia: {str(e)}'
            }
    
    async def generate_signal(self, strategy_id: str, market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera una señal de trading para la estrategia
        """
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            strategy = self.active_strategies[strategy_id]
            
            if strategy['status'] != 'active':
                return {
                    'success': False,
                    'signal': 'HOLD',
                    'reason': 'Estrategia no activa'
                }
            
            # Verificar condiciones de mercado
            if len(market_data) < 30:
                return {
                    'success': False,
                    'signal': 'HOLD',
                    'reason': 'Datos insuficientes'
                }
            
            # Generar señal usando el simulador
            current_idx = len(market_data) - 1
            current_price = market_data['close'].iloc[-1]
            
            # Verificar condiciones de mercado
            market_ok, market_reason = self.simulator.check_market_conditions_24h(market_data, current_idx)
            
            if not market_ok:
                return {
                    'success': True,
                    'signal': 'HOLD',
                    'reason': market_reason
                }
            
            # Generar señal usando el modelo AI
            try:
                signals = self.simulator.eurusd_ai.generate_signals_strategy(market_data, 'scalping')
                
                if signals and len(signals) > 0:
                    latest_signal = signals[-1]
                    
                    # Verificar confianza mínima
                    if latest_signal['confidence'] < strategy['parameters']['min_confidence']:
                        return {
                            'success': True,
                            'signal': 'HOLD',
                            'reason': f'Confianza insuficiente: {latest_signal["confidence"]:.1f}%'
                        }
                    
                    # Verificar si hay posiciones abiertas
                    if len(self.simulator.positions) > 0:
                        return {
                            'success': True,
                            'signal': 'HOLD',
                            'reason': 'Posición abierta'
                        }
                    
                    # Actualizar estrategia
                    strategy['last_signal'] = latest_signal
                    strategy['current_price'] = current_price
                    
                    return {
                        'success': True,
                        'signal': latest_signal['signal'],
                        'confidence': latest_signal['confidence'],
                        'reason': 'Señal generada',
                        'price': current_price,
                        'timestamp': datetime.now().isoformat()
                    }
                
            except Exception as e:
                return {
                    'success': False,
                    'signal': 'HOLD',
                    'reason': f'Error generando señal: {str(e)}'
                }
            
            return {
                'success': True,
                'signal': 'HOLD',
                'reason': 'No hay señales disponibles'
            }
            
        except Exception as e:
            return {
                'success': False,
                'signal': 'HOLD',
                'reason': f'Error en generación de señal: {str(e)}'
            }
    
    async def execute_trade(self, strategy_id: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta una operación de trading
        """
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            strategy = self.active_strategies[strategy_id]
            
            if signal['signal'] == 'HOLD':
                return {
                    'success': True,
                    'action': 'HOLD',
                    'reason': signal['reason']
                }
            
            # Ejecutar operación usando el simulador
            position = self.simulator.execute_trade_24h(
                signal,
                signal['price'],
                datetime.now(),
                True,
                'Condiciones favorables'
            )
            
            if position:
                strategy['total_trades'] += 1
                
                return {
                    'success': True,
                    'action': 'EXECUTED',
                    'position': position,
                    'message': f'Operación ejecutada: {signal["signal"]} {strategy["pair"]}'
                }
            else:
                return {
                    'success': True,
                    'action': 'REJECTED',
                    'reason': 'Operación rechazada por condiciones de mercado'
                }
                
        except Exception as e:
            return {
                'success': False,
                'action': 'ERROR',
                'reason': f'Error ejecutando operación: {str(e)}'
            }
    
    async def get_strategy_status(self, strategy_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado actual de una estrategia
        """
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            strategy = self.active_strategies[strategy_id]
            
            # Calcular estadísticas
            win_rate = 0
            if strategy['total_trades'] > 0:
                win_rate = (strategy['winning_trades'] / strategy['total_trades']) * 100
            
            return {
                'success': True,
                'strategy': {
                    **strategy,
                    'win_rate': win_rate,
                    'open_positions': len(self.simulator.positions),
                    'closed_positions': len(self.simulator.closed_trades)
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_all_strategies(self) -> Dict[str, Any]:
        """
        Obtiene todas las estrategias activas
        """
        try:
            strategies = []
            
            for strategy_id, strategy in self.active_strategies.items():
                # Calcular estadísticas
                win_rate = 0
                if strategy['total_trades'] > 0:
                    win_rate = (strategy['winning_trades'] / strategy['total_trades']) * 100
                
                strategies.append({
                    **strategy,
                    'win_rate': win_rate,
                    'open_positions': len(self.simulator.positions),
                    'closed_positions': len(self.simulator.closed_trades)
                })
            
            return {
                'success': True,
                'strategies': strategies,
                'total_strategies': len(strategies)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    async def delete_strategy(self, strategy_id: str) -> Dict[str, Any]:
        """
        Elimina una estrategia
        """
        try:
            if strategy_id not in self.active_strategies:
                raise ValueError(f"Estrategia no encontrada: {strategy_id}")
            
            strategy = self.active_strategies.pop(strategy_id)
            
            return {
                'success': True,
                'strategy_id': strategy_id,
                'message': f'Estrategia {strategy["name"]} eliminada exitosamente'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Error eliminando estrategia: {str(e)}'
            }

# Instancia global del servicio
scalping_service = ScalpingStrategyService() 