#!/usr/bin/env python3
"""
Script para verificar la autenticidad de las métricas de Precision y Win Rate
en las predicciones del sistema de trading.
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify_trading_results_authenticity():
    """
    Verifica la autenticidad de los trading_results en los metadatos de los modelos
    """
    print("🔍 VERIFICACIÓN DE AUTENTICIDAD DE MÉTRICAS DE TRADING")
    print("=" * 80)
    
    # Directorio de modelos entrenados
    models_dir = "backend/models/trained_models/Brain_Max"
    
    if not os.path.exists(models_dir):
        print(f"❌ Directorio de modelos no encontrado: {models_dir}")
        return
    
    issues_found = []
    authentic_models = []
    
    # Revisar todos los pares y estilos
    for pair in os.listdir(models_dir):
        pair_dir = os.path.join(models_dir, pair)
        if not os.path.isdir(pair_dir):
            continue
            
        print(f"\n📊 Verificando par: {pair}")
        
        for style in os.listdir(pair_dir):
            style_dir = os.path.join(pair_dir, style)
            if not os.path.isdir(style_dir):
                continue
                
            metadata_path = os.path.join(style_dir, "metadata.json")
            
            if not os.path.exists(metadata_path):
                print(f"  ❌ {style}: Metadata no encontrada")
                issues_found.append(f"{pair}/{style}: Metadata no encontrada")
                continue
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                trading_results = metadata.get('trading_results', {})
                
                print(f"  📈 {style}:")
                
                # Verificar que trading_results existe
                if not trading_results:
                    print(f"    ❌ No hay trading_results")
                    issues_found.append(f"{pair}/{style}: No hay trading_results")
                    continue
                
                # Extraer métricas
                win_rate = trading_results.get('win_rate', 0)
                total_trades = trading_results.get('total_trades', 0)
                winning_trades = trading_results.get('winning_trades', 0)
                initial_balance = trading_results.get('initial_balance', 0)
                final_balance = trading_results.get('final_balance', 0)
                
                print(f"    - Win Rate: {win_rate:.2%}")
                print(f"    - Total Trades: {total_trades}")
                print(f"    - Winning Trades: {winning_trades}")
                print(f"    - Balance Inicial: ${initial_balance:,.2f}")
                print(f"    - Balance Final: ${final_balance:,.2f}")
                
                # Verificaciones de autenticidad
                authenticity_score = 0
                max_score = 5
                
                # 1. Verificar que win_rate es consistente con winning_trades/total_trades
                if total_trades > 0:
                    calculated_win_rate = winning_trades / total_trades
                    if abs(calculated_win_rate - win_rate) < 0.01:
                        print(f"    ✅ Win Rate consistente")
                        authenticity_score += 1
                    else:
                        print(f"    ❌ Win Rate inconsistente: calculado={calculated_win_rate:.2%}, reportado={win_rate:.2%}")
                        issues_found.append(f"{pair}/{style}: Win Rate inconsistente")
                else:
                    print(f"    ⚠️  No hay trades para verificar")
                
                # 2. Verificar que los números de trades son realistas
                if 0 <= total_trades <= 1000 and 0 <= winning_trades <= total_trades:
                    print(f"    ✅ Números de trades realistas")
                    authenticity_score += 1
                else:
                    print(f"    ❌ Números de trades no realistas")
                    issues_found.append(f"{pair}/{style}: Números de trades no realistas")
                
                # 3. Verificar que el win_rate está en rango realista
                if 0.3 <= win_rate <= 0.9:
                    print(f"    ✅ Win Rate en rango realista")
                    authenticity_score += 1
                else:
                    print(f"    ⚠️  Win Rate fuera de rango típico: {win_rate:.2%}")
                    issues_found.append(f"{pair}/{style}: Win Rate fuera de rango típico")
                
                # 4. Verificar que el balance final es realista
                if initial_balance > 0 and final_balance > 0:
                    return_pct = (final_balance - initial_balance) / initial_balance
                    if -0.5 <= return_pct <= 2.0:  # -50% a +200%
                        print(f"    ✅ Retorno realista: {return_pct:.2%}")
                        authenticity_score += 1
                    else:
                        print(f"    ⚠️  Retorno extremo: {return_pct:.2%}")
                        issues_found.append(f"{pair}/{style}: Retorno extremo")
                else:
                    print(f"    ❌ Balances inválidos")
                    issues_found.append(f"{pair}/{style}: Balances inválidos")
                
                # 5. Verificar que hay suficientes trades para ser estadísticamente significativo
                if total_trades >= 10:
                    print(f"    ✅ Suficientes trades para significancia estadística")
                    authenticity_score += 1
                else:
                    print(f"    ⚠️  Pocos trades para significancia estadística")
                    issues_found.append(f"{pair}/{style}: Pocos trades para significancia estadística")
                
                # Calcular score de autenticidad
                authenticity_percentage = (authenticity_score / max_score) * 100
                print(f"    📊 Score de Autenticidad: {authenticity_score}/{max_score} ({authenticity_percentage:.1f}%)")
                
                if authenticity_percentage >= 80:
                    print(f"    ✅ MODELO AUTÉNTICO")
                    authentic_models.append(f"{pair}/{style}")
                elif authenticity_percentage >= 60:
                    print(f"    ⚠️  MODELO PARCIALMENTE AUTÉNTICO")
                else:
                    print(f"    ❌ MODELO POTENCIALMENTE NO AUTÉNTICO")
                
            except Exception as e:
                print(f"    ❌ Error leyendo metadata: {e}")
                issues_found.append(f"{pair}/{style}: Error leyendo metadata - {e}")
    
    # Resumen final
    print(f"\n" + "=" * 80)
    print(f"📋 RESUMEN DE VERIFICACIÓN")
    print(f"=" * 80)
    
    print(f"✅ Modelos Auténticos: {len(authentic_models)}")
    for model in authentic_models:
        print(f"   - {model}")
    
    print(f"\n❌ Problemas Encontrados: {len(issues_found)}")
    for issue in issues_found:
        print(f"   - {issue}")
    
    return authentic_models, issues_found

def verify_calculation_methods():
    """
    Verifica los métodos de cálculo de precision y win_rate en el código
    """
    print(f"\n🔧 VERIFICACIÓN DE MÉTODOS DE CÁLCULO")
    print("=" * 80)
    
    # Verificar brain_trader_service.py
    service_file = "backend/src/services/brain_trader_service.py"
    
    if os.path.exists(service_file):
        print(f"📄 Revisando: {service_file}")
        
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar métodos de cálculo
        if 'trading_results' in content and 'win_rate' in content:
            print("  ✅ Usa trading_results del modelo")
        else:
            print("  ❌ No usa trading_results del modelo")
        
        if 'precision = min(win_rate * (confidence / 100.0), 95.0)' in content:
            print("  ✅ Calcula precision basada en win_rate y confidence")
        else:
            print("  ❌ No calcula precision correctamente")
        
        if 'fallback' in content and 'confidence * 0.7' in content:
            print("  ✅ Tiene fallback para cuando no hay trading_results")
        else:
            print("  ❌ No tiene fallback apropiado")
    else:
        print(f"❌ Archivo no encontrado: {service_file}")
    
    # Verificar Modelo_Brain_Max.py
    model_file = "Modelo_Brain_Max.py"
    
    if os.path.exists(model_file):
        print(f"\n📄 Revisando: {model_file}")
        
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar función de simulación
        if 'def simulate_style_trading(' in content:
            print("  ✅ Función simulate_style_trading encontrada")
            
            # Verificar si usa datos reales o simulados
            if 'winning_trades = int(total_trades * 0.6)' in content:
                print("  ⚠️  Usa win_rate fijo de 60% (posiblemente simulado)")
            else:
                print("  ✅ Calcula win_rate dinámicamente")
        else:
            print("  ❌ Función simulate_style_trading no encontrada")
        
        # Verificar función avanzada de simulación
        if 'def simulate_trading_signals(' in content:
            print("  ✅ Función simulate_trading_signals encontrada")
            
            if 'winning_trades = len([t for t in trades if t[\'pnl\'] > 0])' in content:
                print("  ✅ Calcula winning_trades basado en PnL real")
            else:
                print("  ❌ No calcula winning_trades correctamente")
        else:
            print("  ❌ Función simulate_trading_signals no encontrada")
    else:
        print(f"❌ Archivo no encontrado: {model_file}")

def check_data_sources():
    """
    Verifica las fuentes de datos utilizadas
    """
    print(f"\n📊 VERIFICACIÓN DE FUENTES DE DATOS")
    print("=" * 80)
    
    # Verificar si se usan datos reales de Yahoo Finance
    yahoo_usage = False
    simulated_data_usage = False
    
    # Revisar Modelo_Brain_Max.py
    model_file = "Modelo_Brain_Max.py"
    if os.path.exists(model_file):
        with open(model_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'yfinance' in content or 'yahoo' in content.lower():
            print("✅ Usa datos de Yahoo Finance")
            yahoo_usage = True
        else:
            print("❌ No usa datos de Yahoo Finance")
        
        if 'create_simulated_data' in content:
            print("⚠️  Tiene función para crear datos simulados")
            simulated_data_usage = True
        else:
            print("✅ No usa datos simulados")
    
    # Revisar brain_trader_service.py
    service_file = "backend/src/services/brain_trader_service.py"
    if os.path.exists(service_file):
        with open(service_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'yfinance' in content or 'yahoo' in content.lower():
            print("✅ Servicio usa datos de Yahoo Finance")
            yahoo_usage = True
        else:
            print("❌ Servicio no usa datos de Yahoo Finance")
    
    # Conclusión sobre fuentes de datos
    print(f"\n📋 CONCLUSIÓN SOBRE FUENTES DE DATOS:")
    if yahoo_usage and not simulated_data_usage:
        print("✅ Sistema usa datos reales de Yahoo Finance")
    elif yahoo_usage and simulated_data_usage:
        print("⚠️  Sistema usa datos reales pero también tiene capacidad de simulación")
    else:
        print("❌ Sistema no usa datos reales de Yahoo Finance")

def main():
    """
    Función principal de verificación
    """
    print("🎯 VERIFICADOR DE AUTENTICIDAD DE MÉTRICAS DE TRADING")
    print("=" * 80)
    print(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 1. Verificar autenticidad de trading_results
    authentic_models, issues = verify_trading_results_authenticity()
    
    # 2. Verificar métodos de cálculo
    verify_calculation_methods()
    
    # 3. Verificar fuentes de datos
    check_data_sources()
    
    # 4. Resumen final
    print(f"\n" + "=" * 80)
    print(f"🏁 RESUMEN FINAL")
    print(f"=" * 80)
    
    total_models = len(authentic_models) + len([i for i in issues if 'trading_results' in i])
    
    print(f"📊 Total de modelos revisados: {total_models}")
    print(f"✅ Modelos auténticos: {len(authentic_models)}")
    print(f"❌ Problemas encontrados: {len(issues)}")
    
    if len(authentic_models) > len(issues):
        print(f"\n🎉 RESULTADO: La mayoría de las métricas parecen ser auténticas")
    else:
        print(f"\n⚠️  RESULTADO: Se encontraron problemas significativos en las métricas")
    
    print(f"\n💡 RECOMENDACIONES:")
    if len(issues) > 0:
        print("   - Revisar y corregir los problemas identificados")
        print("   - Implementar validaciones adicionales")
        print("   - Documentar mejor los métodos de cálculo")
    else:
        print("   - Las métricas parecen estar en buen estado")
        print("   - Continuar monitoreando la calidad de los datos")

if __name__ == "__main__":
    main() 