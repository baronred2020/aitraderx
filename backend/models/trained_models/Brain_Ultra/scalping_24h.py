# scalping_24h.py - Simulación de scalping EURUSD por 24 horas
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from Modelo_AI_Ultra import EURUSDMultiStrategyAI, generate_eurusd_data

class Scalping24HourSimulator:
    """
    Simulador de scalping EURUSD para 24 horas completas
    """
    
    def __init__(self):
        self.eurusd_ai = EURUSDMultiStrategyAI()
        self.initial_balance = 10000
        self.current_balance = self.initial_balance
        self.positions = []
        self.closed_trades = []
        self.signals_history = []
        
        # PARÁMETROS OPTIMIZADOS PARA 24 HORAS
        self.position_size = 0.2  # 0.2 lot
        self.stop_loss_pips = 2   # 2 pips
        self.take_profit_pips = 4 # 4 pips (ratio 2:1)
        self.min_confidence = 80  # 80%
        self.pip_value = 0.0001
        self.pip_value_usd = self.position_size * 10  # $2 por pip
        
        # FILTROS DE MERCADO PARA 24 HORAS
        self.min_volatility = 0.0003
        self.max_spread = 0.0003
        
        # ESTADÍSTICAS DETALLADAS
        self.total_signals = 0
        self.executed_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.hourly_stats = {}
        
    def calculate_position_size_24h(self, balance, risk_per_trade=0.015):
        """Calcula tamaño de posición para 24 horas"""
        risk_amount = balance * risk_per_trade
        max_loss_usd = self.stop_loss_pips * self.pip_value_usd
        
        if max_loss_usd > 0:
            position_size = risk_amount / max_loss_usd
            return min(position_size, 0.3)  # Máximo 0.3 lot
        return 0.2
    
    def check_market_conditions_24h(self, data, current_idx):
        """Verifica condiciones de mercado para 24 horas"""
        if current_idx < 10:
            return False, "Datos insuficientes"
        
        # Calcular volatilidad
        recent_prices = data['close'].iloc[current_idx-10:current_idx]
        volatility = recent_prices.pct_change().std()
        
        if volatility < self.min_volatility:
            return False, f"Volatilidad baja: {volatility:.6f}"
        
        # Calcular spread
        current_high = data['high'].iloc[current_idx]
        current_low = data['low'].iloc[current_idx]
        spread = current_high - current_low
        
        if spread > self.max_spread:
            return False, f"Spread alto: {spread:.5f}"
        
        # Verificar sesión de trading (24 horas)
        current_time = data.index[current_idx] if hasattr(data, 'index') else current_idx
        if hasattr(current_time, 'hour'):
            hour = current_time.hour
            # Operar en todas las sesiones principales
            if hour < 0 or hour > 23:
                return False, f"Hora fuera de rango: {hour}:00"
        
        return True, "Condiciones favorables"
    
    def execute_trade_24h(self, signal, current_price, timestamp, market_ok, market_reason):
        """Ejecuta operación para 24 horas"""
        if signal['signal'] == 'HOLD':
            return None
        
        # Verificar condiciones de mercado
        if not market_ok:
            return None
        
        # Verificar confianza mínima
        if signal['confidence'] < self.min_confidence:
            return None
        
        # Calcular tamaño de posición
        position_size = self.calculate_position_size_24h(self.current_balance)
        
        # Crear posición
        position = {
            'entry_time': timestamp,
            'entry_price': current_price,
            'signal': signal['signal'],
            'confidence': signal['confidence'],
            'position_size': position_size,
            'take_profit': current_price + (self.take_profit_pips * self.pip_value) if signal['signal'] == 'BUY' else current_price - (self.take_profit_pips * self.pip_value),
            'stop_loss': current_price - (self.stop_loss_pips * self.pip_value) if signal['signal'] == 'BUY' else current_price + (self.stop_loss_pips * self.pip_value),
            'target_pips': self.take_profit_pips,
            'stop_loss_pips': self.stop_loss_pips,
            'risk_reward_ratio': self.take_profit_pips / self.stop_loss_pips,
            'market_condition': market_reason
        }
        
        self.positions.append(position)
        
        print(f"📈 {signal['signal']} EURUSD @ ${current_price:.5f}")
        print(f"   🎯 TP: ${position['take_profit']:.5f} | SL: ${position['stop_loss']:.5f}")
        print(f"   📊 Confianza: {signal['confidence']:.1f}% | Tamaño: {position_size:.2f} lot")
        
        return position
    
    def check_exit_conditions_24h(self, current_price, timestamp):
        """Verifica salidas para 24 horas"""
        closed_positions = []
        
        for position in self.positions[:]:
            exit_reason = None
            exit_price = None
            pnl = 0
            
            if position['signal'] == 'BUY':
                # Verificar take profit
                if current_price >= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = position['take_profit']
                    pips_gained = (exit_price - position['entry_price']) / self.pip_value
                # Verificar stop loss
                elif current_price <= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    exit_price = position['stop_loss']
                    pips_lost = (position['entry_price'] - exit_price) / self.pip_value
                # Verificar tiempo máximo (10 minutos para 24 horas)
                elif (timestamp - position['entry_time']).total_seconds() > 600:  # 10 minutos
                    exit_reason = 'TIME_EXIT'
                    exit_price = current_price
                    pips_gained = (exit_price - position['entry_price']) / self.pip_value
                    
            elif position['signal'] == 'SELL':
                # Verificar take profit
                if current_price <= position['take_profit']:
                    exit_reason = 'TAKE_PROFIT'
                    exit_price = position['take_profit']
                    pips_gained = (position['entry_price'] - exit_price) / self.pip_value
                # Verificar stop loss
                elif current_price >= position['stop_loss']:
                    exit_reason = 'STOP_LOSS'
                    exit_price = position['stop_loss']
                    pips_lost = (position['entry_price'] - exit_price) / self.pip_value
                # Verificar tiempo máximo (10 minutos para 24 horas)
                elif (timestamp - position['entry_time']).total_seconds() > 600:  # 10 minutos
                    exit_reason = 'TIME_EXIT'
                    exit_price = current_price
                    pips_gained = (position['entry_price'] - exit_price) / self.pip_value
            
            if exit_reason:
                # Calcular P&L
                if 'pips_gained' in locals():
                    pnl = pips_gained * self.pip_value_usd * position['position_size']
                    self.winning_trades += 1
                elif 'pips_lost' in locals():
                    pnl = -pips_lost * self.pip_value_usd * position['position_size']
                    self.losing_trades += 1
                
                # Actualizar balance
                self.current_balance += pnl
                
                # Crear trade cerrado
                closed_trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'signal': position['signal'],
                    'exit_reason': exit_reason,
                    'pips': pips_gained if 'pips_gained' in locals() else -pips_lost,
                    'pnl': pnl,
                    'position_size': position['position_size'],
                    'confidence': position['confidence'],
                    'market_condition': position['market_condition']
                }
                
                self.closed_trades.append(closed_trade)
                self.positions.remove(position)
                closed_positions.append(closed_trade)
                
                # Mostrar resultado
                emoji = "🟢" if pnl > 0 else "🔴"
                print(f"{emoji} {exit_reason}: {position['signal']} @ ${exit_price:.5f}")
                print(f"   💰 P&L: ${pnl:.2f} | Pips: {closed_trade['pips']:.1f}")
                print(f"   💵 Balance: ${self.current_balance:.2f}")
        
        return closed_positions
    
    def run_24h_simulation(self, duration_hours=24):
        """Ejecuta simulación de 24 horas"""
        duration_minutes = duration_hours * 60
        
        print("🚀 SIMULACIÓN DE SCALPING EURUSD - 24 HORAS")
        print("=" * 60)
        print(f"⏰ Duración: {duration_hours} horas ({duration_minutes} minutos)")
        print(f"💰 Balance inicial: ${self.initial_balance:.2f}")
        print(f"📊 Parámetros:")
        print(f"   📊 Tamaño: {self.position_size} lot")
        print(f"   📏 Stop Loss: {self.stop_loss_pips} pips")
        print(f"   🎯 Take Profit: {self.take_profit_pips} pips")
        print(f"   🎯 Confianza mínima: {self.min_confidence}%")
        print(f"   ⚖️ Ratio R:R: {self.take_profit_pips/self.stop_loss_pips:.1f}:1")
        print("=" * 60)
        
        # 1. Generar datos de entrenamiento
        print("\n1️⃣ Generando datos de entrenamiento...")
        training_data = generate_eurusd_data('1T', 5000)  # Más datos para entrenar
        
        # 2. Entrenar modelo
        print("\n2️⃣ Entrenando modelo...")
        self.eurusd_ai.train_strategy(training_data, 'scalping')
        
        # 3. Generar datos de simulación de 24 horas
        print("\n3️⃣ Generando datos de simulación de 24 horas...")
        simulation_data = self.generate_24h_data(duration_minutes)
        
        # 4. Ejecutar simulación de 24 horas
        print("\n4️⃣ Ejecutando simulación de 24 horas...")
        print("-" * 60)
        
        ignored_signals = 0
        hourly_pnl = {}
        
        for i, (timestamp, row) in enumerate(simulation_data.iterrows()):
            current_price = row['close']
            current_hour = timestamp.hour if hasattr(timestamp, 'hour') else i // 60
            
            # Verificar salidas de posiciones existentes
            closed_positions = self.check_exit_conditions_24h(current_price, timestamp)
            
            # Actualizar estadísticas por hora
            if current_hour not in hourly_pnl:
                hourly_pnl[current_hour] = {'trades': 0, 'pnl': 0, 'signals': 0}
            
            # Generar señal para el período actual
            current_data = simulation_data.iloc[:i+1]
            
            if len(current_data) >= 30:
                signals = self.eurusd_ai.generate_signals_strategy(current_data, 'scalping')
                
                if signals and len(signals) > 0:
                    latest_signal = signals[-1]
                    self.signals_history.append(latest_signal)
                    self.total_signals += 1
                    hourly_pnl[current_hour]['signals'] += 1
                    
                    # Verificar condiciones de mercado
                    market_ok, market_reason = self.check_market_conditions_24h(current_data, i)
                    
                    # Solo ejecutar si no hay posiciones abiertas y cumple criterios
                    if (len(self.positions) == 0 and 
                        latest_signal['signal'] != 'HOLD'):
                        
                        position = self.execute_trade_24h(latest_signal, current_price, timestamp, market_ok, market_reason)
                        if position:
                            self.executed_trades += 1
                            hourly_pnl[current_hour]['trades'] += 1
                        else:
                            ignored_signals += 1
            
            # Mostrar progreso cada hora
            if (i + 1) % 60 == 0:
                hour_num = (i + 1) // 60
                print(f"⏰ Hora {hour_num:2d}:00 | "
                      f"💰 ${current_price:.5f} | "
                      f"📊 Posiciones: {len(self.positions)} | "
                      f"💵 Balance: ${self.current_balance:.2f}")
        
        # Cerrar posiciones restantes
        final_price = simulation_data['close'].iloc[-1]
        final_timestamp = simulation_data.index[-1]
        self.check_exit_conditions_24h(final_price, final_timestamp)
        
        # 5. Análisis de resultados de 24 horas
        print("\n5️⃣ ANÁLISIS DE RESULTADOS - 24 HORAS")
        print("=" * 60)
        
        self.analyze_24h_results(ignored_signals, hourly_pnl)
        
        return {
            'total_signals': self.total_signals,
            'executed_trades': self.executed_trades,
            'ignored_signals': ignored_signals,
            'closed_trades': len(self.closed_trades),
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'final_balance': self.current_balance,
            'pnl': self.current_balance - self.initial_balance,
            'return_percentage': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100,
            'hourly_stats': hourly_pnl
        }
    
    def generate_24h_data(self, duration_minutes=1440):
        """Genera datos de 24 horas con patrones realistas"""
        print(f"📊 Generando datos de 24 horas ({duration_minutes} minutos)...")
        
        # Generar datos base
        data = generate_eurusd_data('1T', duration_minutes)
        
        # Agregar patrones de sesiones de trading
        start_time = datetime.now().replace(second=0, microsecond=0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(duration_minutes)]
        data['timestamp'] = timestamps
        data.index = timestamps
        
        # Ajustar volatilidad por sesiones
        for i, timestamp in enumerate(timestamps):
            hour = timestamp.hour
            
            # Aumentar volatilidad en sesiones principales
            if 8 <= hour <= 16:  # Londres
                volatility_multiplier = 1.5
            elif 13 <= hour <= 21:  # Nueva York
                volatility_multiplier = 1.8
            elif 13 <= hour <= 16:  # Overlap
                volatility_multiplier = 2.0
            elif 23 <= hour or hour <= 7:  # Asia
                volatility_multiplier = 1.2
            else:
                volatility_multiplier = 1.0
            
            if i > 0:
                price_change = data['close'].iloc[i] - data['close'].iloc[i-1]
                data.loc[data.index[i], 'close'] = data['close'].iloc[i-1] + (price_change * volatility_multiplier)
                
                # Ajustar high y low
                data.loc[data.index[i], 'high'] = max(data['close'].iloc[i], data['high'].iloc[i])
                data.loc[data.index[i], 'low'] = min(data['close'].iloc[i], data['low'].iloc[i])
        
        print(f"✅ Datos de 24 horas generados: {len(data)} períodos")
        print(f"   📅 Rango: {data.index[0]} a {data.index[-1]}")
        print(f"   💰 Precio inicial: ${data['close'].iloc[0]:.5f}")
        print(f"   💰 Precio final: ${data['close'].iloc[-1]:.5f}")
        
        return data
    
    def analyze_24h_results(self, ignored_signals, hourly_pnl):
        """Analiza los resultados de 24 horas"""
        print(f"\n📊 ESTADÍSTICAS DE 24 HORAS")
        print("-" * 40)
        
        # Balance y P&L
        pnl = self.current_balance - self.initial_balance
        return_pct = (pnl / self.initial_balance) * 100
        
        print(f"💰 Balance inicial: ${self.initial_balance:.2f}")
        print(f"💰 Balance final: ${self.current_balance:.2f}")
        print(f"📈 P&L: ${pnl:.2f}")
        print(f"📊 Retorno: {return_pct:.2f}%")
        
        # Estadísticas de señales
        if self.signals_history:
            buy_signals = sum(1 for s in self.signals_history if s['signal'] == 'BUY')
            sell_signals = sum(1 for s in self.signals_history if s['signal'] == 'SELL')
            hold_signals = sum(1 for s in self.signals_history if s['signal'] == 'HOLD')
            avg_confidence = np.mean([s['confidence'] for s in self.signals_history])
            
            print(f"\n🎯 SEÑALES GENERADAS")
            print(f"   📊 Total: {self.total_signals}")
            print(f"   🟢 BUY: {buy_signals}")
            print(f"   🔴 SELL: {sell_signals}")
            print(f"   ⚪ HOLD: {hold_signals}")
            print(f"   🎯 Confianza promedio: {avg_confidence:.1f}%")
            print(f"   ⚠️ Señales ignoradas: {ignored_signals}")
            print(f"   📈 Trades ejecutados: {self.executed_trades}")
        
        # Estadísticas de trades
        if self.closed_trades:
            total_trades = len(self.closed_trades)
            win_rate = (self.winning_trades / total_trades) * 100 if total_trades > 0 else 0
            avg_win = np.mean([t['pnl'] for t in self.closed_trades if t['pnl'] > 0]) if self.winning_trades > 0 else 0
            avg_loss = np.mean([t['pnl'] for t in self.closed_trades if t['pnl'] < 0]) if self.losing_trades > 0 else 0
            
            total_pips = sum(t['pips'] for t in self.closed_trades)
            avg_pips = np.mean([t['pips'] for t in self.closed_trades])
            
            print(f"\n📈 TRADES EJECUTADOS")
            print(f"   📊 Total: {total_trades}")
            print(f"   🟢 Ganadores: {self.winning_trades}")
            print(f"   🔴 Perdedores: {self.losing_trades}")
            print(f"   📊 Win Rate: {win_rate:.1f}%")
            print(f"   💰 Ganancia promedio: ${avg_win:.2f}")
            print(f"   💸 Pérdida promedio: ${avg_loss:.2f}")
            print(f"   📏 Pips totales: {total_pips:.1f}")
            print(f"   📏 Pips promedio: {avg_pips:.1f}")
            
            # Análisis por tipo de salida
            exit_reasons = {}
            for trade in self.closed_trades:
                reason = trade['exit_reason']
                if reason not in exit_reasons:
                    exit_reasons[reason] = {'count': 0, 'pnl': 0}
                exit_reasons[reason]['count'] += 1
                exit_reasons[reason]['pnl'] += trade['pnl']
            
            print(f"\n🎯 ANÁLISIS POR TIPO DE SALIDA")
            for reason, stats in exit_reasons.items():
                avg_pnl = stats['pnl'] / stats['count']
                print(f"   {reason}: {stats['count']} trades, P&L promedio: ${avg_pnl:.2f}")
        
        # Análisis por hora
        print(f"\n⏰ ANÁLISIS POR HORA")
        print("-" * 30)
        
        profitable_hours = 0
        total_hourly_pnl = 0
        
        for hour in range(24):
            if hour in hourly_pnl:
                stats = hourly_pnl[hour]
                if stats['trades'] > 0:
                    profitable_hours += 1 if stats['pnl'] > 0 else 0
                    total_hourly_pnl += stats['pnl']
                    print(f"   Hora {hour:2d}:00 | Trades: {stats['trades']:2d} | P&L: ${stats['pnl']:6.2f} | Señales: {stats['signals']:2d}")
        
        print(f"\n📊 Resumen por hora:")
        print(f"   🕐 Horas con trades: {profitable_hours}/24")
        print(f"   💰 P&L total por hora: ${total_hourly_pnl:.2f}")
        print(f"   📈 Promedio por hora: ${total_hourly_pnl/24:.2f}")
        
        # Proyecciones
        print(f"\n🚀 PROYECCIONES")
        print("-" * 20)
        
        if total_trades > 0:
            trades_per_day = total_trades
            pnl_per_day = pnl
            pnl_per_week = pnl_per_day * 5
            pnl_per_month = pnl_per_day * 22
            
            print(f"   📊 Trades por día: {trades_per_day:.1f}")
            print(f"   💰 P&L por día: ${pnl_per_day:.2f}")
            print(f"   📈 P&L por semana: ${pnl_per_week:.2f}")
            print(f"   📅 P&L por mes: ${pnl_per_month:.2f}")
            print(f"   📊 Retorno anual: {return_pct * 365:.1f}%")

def main():
    """Función principal de la simulación de 24 horas"""
    print("🚀 SIMULADOR DE SCALPING EURUSD - 24 HORAS")
    print("=" * 60)
    
    # Crear simulador de 24 horas
    simulator = Scalping24HourSimulator()
    
    # Ejecutar simulación de 24 horas
    results = simulator.run_24h_simulation(duration_hours=24)
    
    # Mostrar resumen final
    print("\n" + "=" * 60)
    print("🎉 SIMULACIÓN DE 24 HORAS COMPLETADA")
    print("=" * 60)
    print(f"📊 Señales generadas: {results['total_signals']}")
    print(f"⚠️ Señales ignoradas: {results['ignored_signals']}")
    print(f"📈 Trades ejecutados: {results['executed_trades']}")
    print(f"📋 Trades cerrados: {results['closed_trades']}")
    print(f"🟢 Trades ganadores: {results['winning_trades']}")
    print(f"🔴 Trades perdedores: {results['losing_trades']}")
    print(f"💰 P&L final: ${results['pnl']:.2f}")
    print(f"📊 Retorno: {results['return_percentage']:.2f}%")
    
    if results['closed_trades'] > 0:
        win_rate = (results['winning_trades'] / results['closed_trades']) * 100
        print(f"🎯 Win Rate: {win_rate:.1f}%")
    
    print("\n💡 CARACTERÍSTICAS DE LA SIMULACIÓN:")
    print("   1. ✅ 24 horas de trading continuo")
    print("   2. ✅ Patrones de sesiones realistas")
    print("   3. ✅ Gestión de riesgo conservadora")
    print("   4. ✅ Filtros de mercado activos")
    print("   5. ✅ Análisis detallado por hora")
    print("   6. ✅ Proyecciones de rentabilidad")

if __name__ == "__main__":
    main() 