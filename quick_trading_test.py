#!/usr/bin/env python3
"""
Prueba rápida de trading para validar la efectividad de los modelos Brain Max
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.brain_trader_service import BrainTraderService
from utils.model_loader import ModelLoader
import logging
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuickTradingTest:
    def __init__(self, initial_balance=10000, lot_size=0.1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lot_size = lot_size
        self.trades = []
        self.brain_service = BrainTraderService()
        
        # Configuración de trading
        self.sl_pips = 15  # Stop Loss en pips
        self.tp_pips = 25  # Take Profit en pips
        self.pip_value = 0.0001  # Valor de un pip para EURUSD
        
    def get_historical_data(self, pair="EURUSD", days=7):
        """Obtener datos históricos recientes"""
        try:
            ticker = yf.Ticker(f"{pair}=X")
            data = ticker.history(period=f"{days}d", interval="15m")
            return data
        except Exception as e:
            logger.error(f"Error obteniendo datos históricos: {e}")
            return None
    
    def simulate_trade(self, entry_price, direction, confidence, entry_time, exit_price, exit_time):
        """Simular una operación con datos históricos"""
        
        # Calcular Stop Loss y Take Profit
        if direction == 'up':
            sl_price = entry_price - self.calculate_pip_value(entry_price, self.sl_pips)
            tp_price = entry_price + self.calculate_pip_value(entry_price, self.tp_pips)
        else:  # down
            sl_price = entry_price + self.calculate_pip_value(entry_price, self.sl_pips)
            tp_price = entry_price - self.calculate_pip_value(entry_price, self.tp_pips)
        
        # Determinar si se alcanzó SL o TP
        if direction == 'up':
            if exit_price <= sl_price:
                final_price = sl_price
                reason = 'sl'
            elif exit_price >= tp_price:
                final_price = tp_price
                reason = 'tp'
            else:
                final_price = exit_price
                reason = 'manual'
        else:  # down
            if exit_price >= sl_price:
                final_price = sl_price
                reason = 'sl'
            elif exit_price <= tp_price:
                final_price = tp_price
                reason = 'tp'
            else:
                final_price = exit_price
                reason = 'manual'
        
        # Calcular P&L
        if direction == 'up':
            pips = (final_price - entry_price) / self.pip_value
        else:
            pips = (entry_price - final_price) / self.pip_value
        
        pnl = pips * self.lot_size * 10  # $10 por pip para lot_size 0.1
        
        # Crear trade
        trade = {
            'id': len(self.trades) + 1,
            'pair': 'EURUSD',
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': final_price,
            'entry_time': entry_time,
            'exit_time': exit_time,
            'confidence': confidence,
            'pips': pips,
            'pnl': pnl,
            'reason': reason,
            'duration': exit_time - entry_time
        }
        
        self.trades.append(trade)
        self.balance += pnl
        
        return trade
    
    def calculate_pip_value(self, price, pips):
        """Calcular valor en pips"""
        return price * (pips * self.pip_value)
    
    async def generate_prediction_for_data(self, data, index, pair="EURUSD", style="day_trading"):
        """Generar predicción para un punto específico en los datos históricos"""
        try:
            # Usar el precio de cierre en ese momento
            current_price = data['Close'].iloc[index]
            
            # Generar predicción usando el servicio
            prediction = await self.brain_service._get_brain_max_prediction(pair, style, current_price)
            
            if prediction:
                return {
                    'direction': prediction.get('direction'),
                    'confidence': prediction.get('confidence', 0),
                    'target_price': prediction.get('target_price', 0),
                    'current_price': current_price,
                    'timestamp': data.index[index]
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error generando predicción: {e}")
            return None
    
    async def run_quick_test(self, test_periods=10):
        """Ejecutar prueba rápida con datos históricos"""
        print("🚀 PRUEBA RÁPIDA DE TRADING - BRAIN MAX")
        print("=" * 60)
        print(f"💰 Balance inicial: ${self.initial_balance:,.2f}")
        print(f"📊 Períodos de prueba: {test_periods}")
        print(f"🎯 Stop Loss: {self.sl_pips} pips")
        print(f"🎯 Take Profit: {self.tp_pips} pips")
        print(f"📦 Tamaño de lote: {self.lot_size}")
        print("-" * 60)
        
        # Obtener datos históricos
        print("📊 Obteniendo datos históricos...")
        data = self.get_historical_data(days=7)
        
        if data is None or len(data) < test_periods:
            print("❌ No se pudieron obtener suficientes datos históricos")
            return
        
        print(f"✅ Datos obtenidos: {len(data)} registros")
        print(f"   Período: {data.index[0]} a {data.index[-1]}")
        
        # Generar predicciones y simular trades
        predictions_made = 0
        trades_executed = 0
        
        # Tomar muestras espaciadas en el tiempo
        step = max(1, len(data) // test_periods)
        
        for i in range(0, len(data) - 1, step):
            if predictions_made >= test_periods:
                break
                
            try:
                current_time = data.index[i]
                print(f"\n⏰ {current_time.strftime('%Y-%m-%d %H:%M')} - Generando predicción #{predictions_made + 1}")
                
                # Generar predicción
                prediction = await self.generate_prediction_for_data(data, i)
                predictions_made += 1
                
                if prediction:
                    direction = prediction['direction']
                    confidence = prediction['confidence']
                    entry_price = prediction['current_price']
                    entry_time = prediction['timestamp']
                    
                    print(f"🧠 Predicción:")
                    print(f"   Dirección: {direction.upper()}")
                    print(f"   Confianza: {confidence:.2f}%")
                    print(f"   Precio entrada: {entry_price:.5f}")
                    print(f"   Precio objetivo: {prediction['target_price']:.5f}")
                    
                    # Simular trade si confianza es alta
                    if confidence >= 70:  # Umbral de confianza
                        # Buscar precio de salida (15 minutos después)
                        exit_index = min(i + 1, len(data) - 1)  # Siguiente período
                        exit_price = data['Close'].iloc[exit_index]
                        exit_time = data.index[exit_index]
                        
                        # Simular la operación
                        trade = self.simulate_trade(
                            entry_price, direction, confidence, 
                            entry_time, exit_price, exit_time
                        )
                        trades_executed += 1
                        
                        # Mostrar resultado
                        if trade['pnl'] > 0:
                            print(f"   ✅ Trade: +{trade['pips']:.1f} pips (+${trade['pnl']:.2f}) - {trade['reason'].upper()}")
                        else:
                            print(f"   ❌ Trade: {trade['pips']:.1f} pips (${trade['pnl']:.2f}) - {trade['reason'].upper()}")
                    else:
                        print(f"   ⚠️  Confianza insuficiente ({confidence:.2f}% < 70%)")
                else:
                    print("   ❌ No se pudo generar predicción")
                    
            except Exception as e:
                logger.error(f"Error en período {i}: {e}")
                continue
        
        # Mostrar resultados
        self.show_results(predictions_made, trades_executed)
    
    def show_results(self, predictions_made, trades_executed):
        """Mostrar resultados de la prueba"""
        print("\n" + "=" * 60)
        print("🏁 RESULTADOS DE LA PRUEBA RÁPIDA")
        print("=" * 60)
        
        print(f"📊 Resumen:")
        print(f"   Predicciones generadas: {predictions_made}")
        print(f"   Operaciones ejecutadas: {trades_executed}")
        print(f"   Tasa de ejecución: {(trades_executed/predictions_made)*100:.1f}%" if predictions_made > 0 else "   Tasa de ejecución: 0%")
        
        if len(self.trades) == 0:
            print("❌ No se realizaron operaciones")
            return
        
        # Calcular estadísticas
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len(self.trades) - winning_trades
        win_rate = (winning_trades / len(self.trades)) * 100
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pips = sum(t['pips'] for t in self.trades)
        avg_pips = total_pips / len(self.trades)
        
        max_profit = max(t['pnl'] for t in self.trades)
        max_loss = min(t['pnl'] for t in self.trades)
        
        print(f"\n💰 Resultados financieros:")
        print(f"   Balance inicial: ${self.initial_balance:,.2f}")
        print(f"   Balance final: ${self.balance:,.2f}")
        print(f"   P&L total: ${total_pnl:+.2f}")
        print(f"   Retorno: {((self.balance / self.initial_balance) - 1) * 100:+.2f}%")
        
        print(f"\n📊 Operaciones:")
        print(f"   Total: {len(self.trades)}")
        print(f"   Ganadoras: {winning_trades}")
        print(f"   Perdedoras: {losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        
        print(f"\n📏 Pips:")
        print(f"   Total: {total_pips:.1f}")
        print(f"   Promedio: {avg_pips:.1f}")
        print(f"   Máximo ganancia: {max_profit:.2f}")
        print(f"   Máximo pérdida: {max_loss:.2f}")
        
        # Mostrar detalles de cada operación
        print(f"\n📋 DETALLES DE OPERACIONES:")
        print("-" * 60)
        for i, trade in enumerate(self.trades, 1):
            status = "✅" if trade['pnl'] > 0 else "❌"
            print(f"{status} Trade #{i}: {trade['direction'].upper()} EURUSD")
            print(f"   Entrada: {trade['entry_price']:.5f} | Salida: {trade['exit_price']:.5f}")
            print(f"   Pips: {trade['pips']:.1f} | P&L: ${trade['pnl']:.2f}")
            print(f"   Confianza: {trade['confidence']:.1f}% | Razón: {trade['reason'].upper()}")
            print(f"   Duración: {trade['duration']}")
            print()
        
        # Análisis de confianza
        confidences = [t['confidence'] for t in self.trades]
        avg_confidence = np.mean(confidences)
        print(f"📈 Análisis de confianza:")
        print(f"   Promedio: {avg_confidence:.2f}%")
        print(f"   Mínima: {min(confidences):.2f}%")
        print(f"   Máxima: {max(confidences):.2f}%")
        
        # Evaluación del modelo
        print(f"\n🎯 EVALUACIÓN DEL MODELO:")
        if win_rate >= 60:
            print(f"   ✅ Win Rate bueno ({win_rate:.1f}% >= 60%)")
        elif win_rate >= 50:
            print(f"   ⚠️  Win Rate aceptable ({win_rate:.1f}% >= 50%)")
        else:
            print(f"   ❌ Win Rate bajo ({win_rate:.1f}% < 50%)")
        
        if avg_confidence > 90:
            print(f"   ⚠️  Confianza muy alta ({avg_confidence:.1f}% > 90%) - posible overfitting")
        elif avg_confidence < 50:
            print(f"   ⚠️  Confianza muy baja ({avg_confidence:.1f}% < 50%) - posible underfitting")
        else:
            print(f"   ✅ Confianza en rango normal ({avg_confidence:.1f}%)")
        
        if total_pnl > 0:
            print(f"   ✅ P&L positivo (${total_pnl:.2f})")
        else:
            print(f"   ❌ P&L negativo (${total_pnl:.2f})")
        
        # Guardar resultados
        self.save_results()
    
    def save_results(self):
        """Guardar resultados en archivo JSON"""
        results = {
            'timestamp': datetime.now().isoformat(),
            'initial_balance': self.initial_balance,
            'final_balance': self.balance,
            'total_pnl': self.balance - self.initial_balance,
            'return_percentage': ((self.balance / self.initial_balance) - 1) * 100,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
            'win_rate': (len([t for t in self.trades if t['pnl'] > 0]) / len(self.trades)) * 100 if self.trades else 0,
            'trades': [
                {
                    'id': t['id'],
                    'pair': t['pair'],
                    'direction': t['direction'],
                    'entry_price': t['entry_price'],
                    'exit_price': t['exit_price'],
                    'pips': t['pips'],
                    'pnl': t['pnl'],
                    'confidence': t['confidence'],
                    'reason': t['reason'],
                    'duration': str(t['duration']),
                    'entry_time': t['entry_time'].isoformat(),
                    'exit_time': t['exit_time'].isoformat()
                }
                for t in self.trades
            ]
        }
        
        filename = f"quick_trading_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Resultados guardados en: {filename}")

async def main():
    """Función principal"""
    print("🧪 PRUEBA RÁPIDA DE TRADING - BRAIN MAX")
    print("=" * 60)
    
    # Configuración de la prueba
    initial_balance = 10000  # $10,000
    lot_size = 0.1  # 0.1 lotes
    test_periods = 10  # 10 períodos de prueba
    
    # Crear prueba
    test = QuickTradingTest(
        initial_balance=initial_balance,
        lot_size=lot_size
    )
    
    # Ejecutar prueba
    await test.run_quick_test(test_periods=test_periods)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 