"""
Monitor de Consumo de API - Twelve Data
=======================================
Script para monitorear el consumo de API y verificar límites
"""

import time
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class APIMonitor:
    def __init__(self):
        self.stats_file = "logs/api_usage.json"
        self.daily_limit = 800
        self.current_usage = 0
        self.last_reset = datetime.now()
        self.load_stats()
    
    def load_stats(self):
        """Cargar estadísticas de uso desde archivo"""
        try:
            if os.path.exists(self.stats_file):
                with open(self.stats_file, 'r') as f:
                    data = json.load(f)
                    self.current_usage = data.get('current_usage', 0)
                    self.last_reset = datetime.fromisoformat(data.get('last_reset', datetime.now().isoformat()))
            else:
                self.reset_daily_stats()
        except Exception as e:
            print(f"Error cargando estadísticas: {e}")
            self.reset_daily_stats()
    
    def save_stats(self):
        """Guardar estadísticas de uso"""
        try:
            os.makedirs(os.path.dirname(self.stats_file), exist_ok=True)
            with open(self.stats_file, 'w') as f:
                json.dump({
                    'current_usage': self.current_usage,
                    'last_reset': self.last_reset.isoformat(),
                    'daily_limit': self.daily_limit,
                    'usage_percentage': (self.current_usage / self.daily_limit) * 100
                }, f, indent=2)
        except Exception as e:
            print(f"Error guardando estadísticas: {e}")
    
    def reset_daily_stats(self):
        """Resetear estadísticas diarias"""
        self.current_usage = 0
        self.last_reset = datetime.now()
        self.save_stats()
    
    def check_daily_reset(self):
        """Verificar si necesitamos resetear estadísticas diarias"""
        now = datetime.now()
        if (now - self.last_reset).days >= 1:
            self.reset_daily_stats()
            return True
        return False
    
    def increment_usage(self, calls: int = 1):
        """Incrementar contador de uso"""
        self.check_daily_reset()
        self.current_usage += calls
        self.save_stats()
    
    def can_make_call(self) -> bool:
        """Verificar si podemos hacer una llamada"""
        self.check_daily_reset()
        return self.current_usage < self.daily_limit
    
    def get_usage_stats(self) -> Dict:
        """Obtener estadísticas de uso"""
        self.check_daily_reset()
        return {
            'current_usage': self.current_usage,
            'daily_limit': self.daily_limit,
            'remaining_calls': self.daily_limit - self.current_usage,
            'usage_percentage': (self.current_usage / self.daily_limit) * 100,
            'last_reset': self.last_reset.isoformat(),
            'next_reset': (self.last_reset + timedelta(days=1)).isoformat()
        }
    
    def get_usage_warning(self) -> Tuple[bool, str]:
        """Obtener advertencia de uso"""
        stats = self.get_usage_stats()
        percentage = stats['usage_percentage']
        
        if percentage >= 90:
            return True, f"⚠️ CRÍTICO: {percentage:.1f}% del límite usado ({self.current_usage}/{self.daily_limit})"
        elif percentage >= 75:
            return True, f"⚠️ ALTO: {percentage:.1f}% del límite usado ({self.current_usage}/{self.daily_limit})"
        elif percentage >= 50:
            return True, f"ℹ️ MEDIO: {percentage:.1f}% del límite usado ({self.current_usage}/{self.daily_limit})"
        else:
            return False, f"✅ NORMAL: {percentage:.1f}% del límite usado ({self.current_usage}/{self.daily_limit})"

def calculate_expected_usage() -> Dict:
    """Calcular uso esperado basado en configuración actual"""
    
    # Configuración actual
    symbols = 3  # EURUSD, GBPUSD, USDJPY
    price_cache_minutes = 15
    candle_cache_minutes = 30
    
    # Cálculos
    minutes_per_day = 24 * 60
    price_calls_per_symbol = minutes_per_day / price_cache_minutes
    candle_calls_per_symbol = minutes_per_day / candle_cache_minutes
    total_calls_per_symbol = price_calls_per_symbol + candle_calls_per_symbol
    total_daily_calls = total_calls_per_symbol * symbols
    
    return {
        'symbols': symbols,
        'price_cache_minutes': price_cache_minutes,
        'candle_cache_minutes': candle_cache_minutes,
        'price_calls_per_symbol': int(price_calls_per_symbol),
        'candle_calls_per_symbol': int(candle_calls_per_symbol),
        'total_calls_per_symbol': int(total_calls_per_symbol),
        'total_daily_calls': int(total_daily_calls),
        'daily_limit': 800,
        'usage_percentage': (total_daily_calls / 800) * 100,
        'remaining_calls': 800 - total_daily_calls
    }

def print_usage_report():
    """Imprimir reporte de uso"""
    monitor = APIMonitor()
    stats = monitor.get_usage_stats()
    expected = calculate_expected_usage()
    
    print("=" * 60)
    print("📊 REPORTE DE CONSUMO DE API - TWELVE DATA")
    print("=" * 60)
    
    print(f"\n📈 USO ACTUAL:")
    print(f"   • Llamadas usadas: {stats['current_usage']}")
    print(f"   • Límite diario: {stats['daily_limit']}")
    print(f"   • Llamadas restantes: {stats['remaining_calls']}")
    print(f"   • Porcentaje usado: {stats['usage_percentage']:.1f}%")
    
    warning, message = monitor.get_usage_warning()
    print(f"\n⚠️  ESTADO: {message}")
    
    print(f"\n📋 CONFIGURACIÓN ACTUAL:")
    print(f"   • Símbolos: {expected['symbols']}")
    print(f"   • Cache precios: {expected['price_cache_minutes']} minutos")
    print(f"   • Cache velas: {expected['candle_cache_minutes']} minutos")
    
    print(f"\n🎯 USO ESPERADO:")
    print(f"   • Llamadas por símbolo: {expected['total_calls_per_symbol']}")
    print(f"   • Total diario esperado: {expected['total_daily_calls']}")
    print(f"   • Porcentaje esperado: {expected['usage_percentage']:.1f}%")
    print(f"   • Margen disponible: {expected['remaining_calls']} llamadas")
    
    print(f"\n🔄 RESET:")
    print(f"   • Último reset: {stats['last_reset']}")
    print(f"   • Próximo reset: {stats['next_reset']}")
    
    print("=" * 60)

if __name__ == "__main__":
    print_usage_report() 