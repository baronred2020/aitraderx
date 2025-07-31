#!/usr/bin/env python3
"""
Simulación de Trading para probar la efectividad de los modelos Brain Max
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
import time

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingSimulator:
    def __init__(self, initial_balance=10000, lot_size=0.1):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.lot_size = lot_size
        self.trades = []
        self.current_trade = None
        self.brain_service = BrainTraderService()
        self.model_loader = ModelLoader()
        
        # Configuración de trading
        self.sl_pips = 15  # Stop Loss en pips
        self.tp_pips = 25  # Take Profit en pips
        self.pip_value = 0.0001  # Valor de un pip para EURUSD
        
    def get_current_price(self, pair="EURUSD"):
        """Obtener precio actual en tiempo real"""
        try:
            ticker = yf.Ticker(f"{pair}=X")
            current_price = ticker.info.get('regularMarketPrice')
            if current_price is None:
                # Fallback: obtener último precio de datos históricos
                data = ticker.history(period="1d", interval="1m")
                if len(data) > 0:
                    current_price = data['Close'].iloc[-1]
                else:
                    current_price = 1.1550  # Precio por defecto
            return current_price
        except Exception as e:
            logger.error(f"Error obteniendo precio actual: {e}")
            return 1.1550
    
    def calculate_pip_value(self, price, pips):
        """Calcular valor en pips"""
        return price * (pips * self.pip_value)
    
    def open_trade(self, pair, direction, entry_price, confidence, prediction_id=None):
        """Abrir una nueva operación"""
        if self.current_trade is not None:
            logger.warning("Ya hay una operación abierta")
            return False
        
        # Calcular Stop Loss y Take Profit
        if direction == 'up':
            sl_price = entry_price - self.calculate_pip_value(entry_price, self.sl_pips)
            tp_price = entry_price + self.calculate_pip_value(entry_price, self.tp_pips)
        else:  # down
            sl_price = entry_price + self.calculate_pip_value(entry_price, self.sl_pips)
            tp_price = entry_price - self.calculate_pip_value(entry_price, self.tp_pips)
        
        # Calcular riesgo por operación
        risk_amount = abs(entry_price - sl_price) * self.lot_size * 100000  # 100000 para lotes estándar
        
        self.current_trade = {
            'id': len(self.trades) + 1,
            'pair': pair,
            'direction': direction,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'prediction_id': prediction_id,
            'status': 'open',
            'risk_amount': risk_amount
        }
        
        logger.info(f"🟢 Operación abierta: {direction.upper()} {pair} @ {entry_price:.5f}")
        logger.info(f"   SL: {sl_price:.5f}, TP: {tp_price:.5f}, Confianza: {confidence:.2f}%")
        
        return True
    
    def check_trade_status(self, current_price):
        """Verificar si la operación actual debe cerrarse"""
        if self.current_trade is None:
            return
        
        trade = self.current_trade
        direction = trade['direction']
        entry_price = trade['entry_price']
        sl_price = trade['sl_price']
        tp_price = trade['tp_price']
        
        # Verificar si se alcanzó Stop Loss o Take Profit
        if direction == 'up':
            if current_price <= sl_price:
                self.close_trade(current_price, 'sl')
            elif current_price >= tp_price:
                self.close_trade(current_price, 'tp')
        else:  # down
            if current_price >= sl_price:
                self.close_trade(current_price, 'sl')
            elif current_price <= tp_price:
                self.close_trade(current_price, 'tp')
    
    def close_trade(self, exit_price, reason):
        """Cerrar la operación actual"""
        if self.current_trade is None:
            return
        
        trade = self.current_trade
        entry_price = trade['entry_price']
        direction = trade['direction']
        
        # Calcular P&L
        if direction == 'up':
            pips = (exit_price - entry_price) / self.pip_value
        else:
            pips = (entry_price - exit_price) / self.pip_value
        
        pnl = pips * self.lot_size * 10  # $10 por pip para lot_size 0.1
        
        # Actualizar balance
        self.balance += pnl
        
        # Actualizar trade
        trade['exit_price'] = exit_price
        trade['exit_time'] = datetime.now()
        trade['reason'] = reason
        trade['pips'] = pips
        trade['pnl'] = pnl
        trade['status'] = 'closed'
        trade['duration'] = trade['exit_time'] - trade['entry_time']
        
        # Agregar a historial
        self.trades.append(trade)
        
        # Log del resultado
        if pnl > 0:
            logger.info(f"✅ Trade cerrado: +{pips:.1f} pips (+${pnl:.2f}) - {reason.upper()}")
        else:
            logger.warning(f"❌ Trade cerrado: {pips:.1f} pips (${pnl:.2f}) - {reason.upper()}")
        
        logger.info(f"💰 Balance actual: ${self.balance:.2f}")
        
        # Limpiar trade actual
        self.current_trade = None
    
    def generate_prediction(self, pair="EURUSD", style="day_trading"):
        """Generar predicción usando Brain Max"""
        try:
            current_price = self.get_current_price(pair)
            prediction = self.brain_service._get_brain_max_prediction(pair, style, current_price)
            
            if prediction:
                return {
                    'direction': prediction.get('direction'),
                    'confidence': prediction.get('confidence', 0),
                    'target_price': prediction.get('target_price', 0),
                    'current_price': current_price,
                    'reasoning': prediction.get('reasoning', 'N/A')
                }
            else:
                return None
        except Exception as e:
            logger.error(f"Error generando predicción: {e}")
            return None
    
    def run_simulation(self, duration_minutes=60, check_interval=5):
        """Ejecutar simulación de trading"""
        print("🚀 INICIANDO SIMULACIÓN DE TRADING")
        print("=" * 60)
        print(f"💰 Balance inicial: ${self.initial_balance:,.2f}")
        print(f"📊 Duración: {duration_minutes} minutos")
        print(f"⏱️  Intervalo de verificación: {check_interval} minutos")
        print(f"🎯 Stop Loss: {self.sl_pips} pips")
        print(f"🎯 Take Profit: {self.tp_pips} pips")
        print(f"📦 Tamaño de lote: {self.lot_size}")
        print("-" * 60)
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        trades_count = 0
        predictions_count = 0
        
        while datetime.now() < end_time:
            try:
                current_time = datetime.now()
                print(f"\n⏰ {current_time.strftime('%H:%M:%S')} - Verificando mercado...")
                
                # Verificar operación actual si existe
                if self.current_trade is not None:
                    current_price = self.get_current_price()
                    self.check_trade_status(current_price)
                
                # Generar nueva predicción si no hay operación abierta
                if self.current_trade is None:
                    prediction = self.generate_prediction()
                    predictions_count += 1
                    
                    if prediction:
                        direction = prediction['direction']
                        confidence = prediction['confidence']
                        current_price = prediction['current_price']
                        
                        print(f"🧠 Predicción #{predictions_count}:")
                        print(f"   Dirección: {direction.upper()}")
                        print(f"   Confianza: {confidence:.2f}%")
                        print(f"   Precio actual: {current_price:.5f}")
                        print(f"   Precio objetivo: {prediction['target_price']:.5f}")
                        
                        # Abrir operación si confianza es alta
                        if confidence >= 70:  # Umbral de confianza
                            if self.open_trade("EURUSD", direction, current_price, confidence, predictions_count):
                                trades_count += 1
                        else:
                            print(f"   ⚠️  Confianza insuficiente ({confidence:.2f}% < 70%)")
                    else:
                        print("   ❌ No se pudo generar predicción")
                
                # Mostrar estadísticas
                if len(self.trades) > 0:
                    winning_trades = len([t for t in self.trades if t['pnl'] > 0])
                    total_trades = len(self.trades)
                    win_rate = (winning_trades / total_trades) * 100
                    
                    print(f"\n📊 Estadísticas:")
                    print(f"   Trades totales: {total_trades}")
                    print(f"   Trades ganadores: {winning_trades}")
                    print(f"   Win Rate: {win_rate:.1f}%")
                    print(f"   Balance: ${self.balance:.2f}")
                    print(f"   P&L: ${self.balance - self.initial_balance:+.2f}")
                
                # Esperar hasta la siguiente verificación
                if datetime.now() < end_time:
                    print(f"⏳ Esperando {check_interval} minutos...")
                    time.sleep(check_interval * 60)
                
            except KeyboardInterrupt:
                print("\n⏹️  Simulación interrumpida por el usuario")
                break
            except Exception as e:
                logger.error(f"Error en simulación: {e}")
                time.sleep(60)  # Esperar 1 minuto antes de continuar
        
        # Cerrar operación abierta si existe
        if self.current_trade is not None:
            current_price = self.get_current_price()
            self.close_trade(current_price, 'manual')
        
        # Mostrar resultados finales
        self.show_final_results()
    
    def show_final_results(self):
        """Mostrar resultados finales de la simulación"""
        print("\n" + "=" * 60)
        print("🏁 RESULTADOS FINALES DE LA SIMULACIÓN")
        print("=" * 60)
        
        total_trades = len(self.trades)
        if total_trades == 0:
            print("❌ No se realizaron operaciones")
            return
        
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pips = sum(t['pips'] for t in self.trades)
        avg_pips = total_pips / total_trades
        
        max_profit = max(t['pnl'] for t in self.trades) if self.trades else 0
        max_loss = min(t['pnl'] for t in self.trades) if self.trades else 0
        
        print(f"💰 Balance inicial: ${self.initial_balance:,.2f}")
        print(f"💰 Balance final: ${self.balance:,.2f}")
        print(f"💰 P&L total: ${total_pnl:+.2f}")
        print(f"📈 Retorno: {((self.balance / self.initial_balance) - 1) * 100:+.2f}%")
        print()
        print(f"📊 Operaciones:")
        print(f"   Total: {total_trades}")
        print(f"   Ganadoras: {winning_trades}")
        print(f"   Perdedoras: {losing_trades}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print()
        print(f"📏 Pips:")
        print(f"   Total: {total_pips:.1f}")
        print(f"   Promedio: {avg_pips:.1f}")
        print(f"   Máximo ganancia: {max_profit:.2f}")
        print(f"   Máximo pérdida: {max_loss:.2f}")
        print()
        
        # Mostrar detalles de cada operación
        print("📋 DETALLES DE OPERACIONES:")
        print("-" * 60)
        for i, trade in enumerate(self.trades, 1):
            status = "✅" if trade['pnl'] > 0 else "❌"
            print(f"{status} Trade #{i}: {trade['direction'].upper()} {trade['pair']}")
            print(f"   Entrada: {trade['entry_price']:.5f} | Salida: {trade['exit_price']:.5f}")
            print(f"   Pips: {trade['pips']:.1f} | P&L: ${trade['pnl']:.2f}")
            print(f"   Confianza: {trade['confidence']:.1f}% | Razón: {trade['reason'].upper()}")
            print(f"   Duración: {trade['duration']}")
            print()
        
        # Guardar resultados en archivo
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
        
        filename = f"trading_simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Resultados guardados en: {filename}")

async def main():
    """Función principal"""
    print("🧪 SIMULADOR DE TRADING - BRAIN MAX")
    print("=" * 60)
    
    # Configuración de la simulación
    initial_balance = 10000  # $10,000
    lot_size = 0.1  # 0.1 lotes
    duration_minutes = 120  # 2 horas
    check_interval = 5  # Verificar cada 5 minutos
    
    # Crear simulador
    simulator = TradingSimulator(
        initial_balance=initial_balance,
        lot_size=lot_size
    )
    
    # Ejecutar simulación
    simulator.run_simulation(
        duration_minutes=duration_minutes,
        check_interval=check_interval
    )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 