#!/usr/bin/env python3
"""
Script para probar el sistema de límites de señales
"""

import sys
import os

# Agregar el directorio backend/src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend', 'src'))

from services.signal_service import SignalService
import asyncio

async def test_signal_limits():
    """Probar el sistema de límites de señales"""
    
    print("🧪 Probando sistema de límites de señales...")
    
    # Crear instancia del servicio
    signal_service = SignalService()
    
    # Usuario de prueba
    test_user_id = "0bb94f45-4299-4506-b8c4-9d12d438c79c"
    test_plan = "starter"
    test_style = "day_trading"
    
    print(f"\n📋 Usuario de prueba: {test_user_id}")
    print(f"📋 Plan: {test_plan}")
    print(f"📋 Estilo: {test_style}")
    
    # 1. Verificar límites iniciales
    print("\n🔍 1. Verificando límites iniciales...")
    limits = await signal_service.can_generate_signal(test_user_id, test_style, test_plan)
    
    print(f"   ✅ Puede generar: {limits['can_generate']}")
    print(f"   ✅ Señales restantes: {limits['remaining_signals']}")
    print(f"   ✅ Máximo por día: {limits['max_signals_per_day']}")
    print(f"   ✅ Plan: {limits['plan_type']}")
    
    # 2. Generar primera señal
    print("\n🚀 2. Generando primera señal...")
    success = signal_service.increment_signal_usage(test_user_id)
    print(f"   ✅ Incremento exitoso: {success}")
    
    # 3. Verificar límites después de primera señal
    print("\n🔍 3. Verificando límites después de primera señal...")
    limits_after = await signal_service.can_generate_signal(test_user_id, test_style, test_plan)
    
    print(f"   ✅ Puede generar: {limits_after['can_generate']}")
    print(f"   ✅ Señales restantes: {limits_after['remaining_signals']}")
    print(f"   ✅ Máximo por día: {limits_after['max_signals_per_day']}")
    
    # 4. Generar señales hasta agotar límite
    print("\n🚀 4. Generando señales hasta agotar límite...")
    
    remaining = limits_after['remaining_signals']
    for i in range(remaining):
        print(f"   🔄 Generando señal {i + 1}/{remaining}...")
        success = signal_service.increment_signal_usage(test_user_id)
        if not success:
            print(f"   ❌ Error al generar señal {i + 1}")
            break
    
    # 5. Verificar límites finales
    print("\n🔍 5. Verificando límites finales...")
    final_limits = await signal_service.can_generate_signal(test_user_id, test_style, test_plan)
    
    print(f"   ✅ Puede generar: {final_limits['can_generate']}")
    print(f"   ✅ Señales restantes: {final_limits['remaining_signals']}")
    print(f"   ✅ Máximo por día: {final_limits['max_signals_per_day']}")
    
    # 6. Intentar generar una señal más (debería fallar)
    print("\n🚫 6. Intentando generar señal adicional (debería fallar)...")
    if final_limits['can_generate']:
        print("   ⚠️  Aún puede generar señales (posible error)")
    else:
        print("   ✅ Correcto: No puede generar más señales")
    
    # 7. Probar con plan ilimitado
    print("\n🔍 7. Probando con plan institucional (ilimitado)...")
    unlimited_limits = await signal_service.can_generate_signal(test_user_id, test_style, "institutional")
    
    print(f"   ✅ Puede generar: {unlimited_limits['can_generate']}")
    print(f"   ✅ Señales restantes: {unlimited_limits['remaining_signals']}")
    print(f"   ✅ Es ilimitado: {unlimited_limits['has_unlimited']}")
    
    print("\n🎉 Prueba completada!")

if __name__ == "__main__":
    asyncio.run(test_signal_limits()) 