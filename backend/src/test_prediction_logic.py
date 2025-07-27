#!/usr/bin/env python3
"""
Script de prueba para verificar la lógica de predicción corregida
"""

import asyncio
import sys
import os

# Agregar el directorio actual al path
sys.path.append(os.path.dirname(__file__))

from services.brain_trader_service import BrainTraderService

async def test_prediction_logic():
    """Probar la lógica de predicción corregida"""
    print("🧪 **PRUEBA DE LÓGICA DE PREDICCIÓN**")
    print("=" * 50)
    
    # Crear instancia del servicio
    service = BrainTraderService()
    
    # Precio actual de ejemplo (similar al que viste en la UI)
    current_price = 1.1761
    
    print(f"💰 **Precio Actual:** ${current_price}")
    print()
    
    # Probar diferentes estilos de trading
    styles = ['day_trading', 'scalping', 'swing_trading', 'position_trading']
    
    for style in styles:
        print(f"📊 **Probando estilo: {style}**")
        
        # Probar predicción Brain Max
        try:
            prediction_data = await service._get_brain_max_prediction('EURUSD', style, current_price)
            
            direction = prediction_data['direction']
            confidence = prediction_data['confidence']
            target_price = prediction_data['target_price']
            reasoning = prediction_data['reasoning']
            
            print(f"  🎯 Dirección: {direction.upper()}")
            print(f"  📈 Confianza: {confidence:.1f}%")
            print(f"  💰 Precio Objetivo: ${target_price:.5f}")
            print(f"  📝 Razón: {reasoning}")
            
            # Verificar consistencia
            if direction == 'up':
                if target_price > current_price:
                    print(f"  ✅ **CONSISTENTE:** Precio objetivo ({target_price:.5f}) > Precio actual ({current_price:.5f})")
                else:
                    print(f"  ❌ **INCONSISTENTE:** Precio objetivo ({target_price:.5f}) <= Precio actual ({current_price:.5f})")
            elif direction == 'down':
                if target_price < current_price:
                    print(f"  ✅ **CONSISTENTE:** Precio objetivo ({target_price:.5f}) < Precio actual ({current_price:.5f})")
                else:
                    print(f"  ❌ **INCONSISTENTE:** Precio objetivo ({target_price:.5f}) >= Precio actual ({current_price:.5f})")
            else:  # sideways
                print(f"  🔄 **LATERAL:** Precio objetivo ({target_price:.5f}) cerca del precio actual ({current_price:.5f})")
            
            # Calcular diferencia porcentual
            diff_percent = ((target_price - current_price) / current_price) * 100
            print(f"  📊 Diferencia: {diff_percent:+.3f}%")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        print()
    
    print("🎯 **RESUMEN DE LA PRUEBA**")
    print("=" * 50)
    print("✅ La lógica corregida debería mostrar:")
    print("   - Para predicciones 'UP': precio objetivo > precio actual")
    print("   - Para predicciones 'DOWN': precio objetivo < precio actual")
    print("   - Para predicciones 'SIDEWAYS': precio objetivo ≈ precio actual")
    print()
    print("📊 **Rangos esperados por estilo:**")
    print("   - Day Trading (15min): 0.1% a 0.5%")
    print("   - Scalping (5min): 0.05% a 0.2%")
    print("   - Swing Trading (1h): 0.2% a 1%")
    print("   - Position Trading (4h): 0.5% a 2%")

if __name__ == "__main__":
    asyncio.run(test_prediction_logic()) 