#!/usr/bin/env python3
"""
Script para diagnosticar el score de calidad de señales
"""

import sys
import os

# Agregar el directorio backend/src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'src'))

def diagnose_score():
    """Diagnosticar por qué el score es 55%"""
    
    print("🔍 Diagnosticando score de calidad...\n")
    
    try:
        from services.brain_trader_service import BrainTraderService
        from services.confidence_calculator import confidence_calculator
        
        service = BrainTraderService()
        
        # Simular indicadores típicos
        indicators = {
            'rsi': 45,  # RSI neutral
            'macd': 0.001,  # MACD débil
            'macd_signal': 0.0005,
            'adx': 20,  # ADX bajo (tendencia débil)
            'sma_20': 1.0920,
            'sma_50': 1.0915
        }
        
        print("📊 Indicadores simulados:")
        for key, value in indicators.items():
            print(f"  {key}: {value}")
        
        # Calcular confianza real
        real_confidence = confidence_calculator.calculate_real_confidence(indicators)
        print(f"\n🎯 Confianza real calculada: {real_confidence:.1f}%")
        
        # Calcular score de calidad
        quality_score = service._calculate_signal_quality('buy', 'medium', 75.0, indicators)
        print(f"📊 Score de calidad: {quality_score:.1f}%")
        
        # Analizar componentes
        print(f"\n🔍 Análisis de componentes:")
        print(f"  - Confianza real: {real_confidence:.1f}%")
        print(f"  - Peso de confianza (80%): {real_confidence * 0.8:.1f}%")
        print(f"  - Fuerza de señal (medium): 15%")
        print(f"  - Score total: {quality_score:.1f}%")
        
        # Probar con indicadores más fuertes
        print(f"\n🚀 Probando con indicadores más fuertes:")
        strong_indicators = {
            'rsi': 25,  # RSI sobreventa
            'macd': 0.005,  # MACD fuerte
            'macd_signal': 0.002,
            'adx': 35,  # ADX alto (tendencia fuerte)
            'sma_20': 1.0920,
            'sma_50': 1.0915
        }
        
        strong_confidence = confidence_calculator.calculate_real_confidence(strong_indicators)
        strong_score = service._calculate_signal_quality('buy', 'strong', 85.0, strong_indicators)
        
        print(f"  - Confianza fuerte: {strong_confidence:.1f}%")
        print(f"  - Score fuerte: {strong_score:.1f}%")
        
        # Recomendaciones
        print(f"\n💡 Recomendaciones para mejorar el score:")
        print(f"  1. RSI más extremo (< 30 o > 70): +15-20%")
        print(f"  2. MACD más fuerte (> 0.003): +10-15%")
        print(f"  3. ADX más alto (> 30): +10-15%")
        print(f"  4. Fuerza de señal 'strong': +5%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = diagnose_score()
    
    if success:
        print(f"\n✅ Diagnóstico completado")
    else:
        print(f"\n❌ Diagnóstico falló")