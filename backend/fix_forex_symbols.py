#!/usr/bin/env python3
"""
Script para arreglar símbolos de forex incorrectos
=================================================
Cambia los símbolos de forex de formato yfinance a formato correcto
"""

import os
import re
import glob

def fix_forex_symbols():
    """Arreglar símbolos de forex en todos los archivos Python"""
    
    # Mapeo de símbolos incorrectos a correctos
    symbol_mapping = {
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD', 
        'USDJPY': 'USDJPY',
        'AUDUSD': 'AUDUSD',
        'USDCAD': 'USDCAD',
        '$EURUSD': 'EURUSD',
        '$GBPUSD': 'GBPUSD',
        '$USDJPY': 'USDJPY',
        '$AUDUSD': 'AUDUSD',
        '$USDCAD': 'USDCAD'
    }
    
    # Buscar todos los archivos Python
    python_files = []
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    print(f"🔍 Encontrados {len(python_files)} archivos Python")
    
    fixed_files = 0
    total_replacements = 0
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            replacements = 0
            
            # Reemplazar símbolos incorrectos
            for old_symbol, new_symbol in symbol_mapping.items():
                if old_symbol in content:
                    content = content.replace(old_symbol, new_symbol)
                    replacements += content.count(new_symbol) - original_content.count(new_symbol)
            
            # Si hubo cambios, guardar el archivo
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                fixed_files += 1
                total_replacements += replacements
                print(f"✅ Arreglado: {file_path} ({replacements} reemplazos)")
        
        except Exception as e:
            print(f"❌ Error procesando {file_path}: {e}")
    
    print(f"\n📊 RESUMEN:")
    print(f"   Archivos arreglados: {fixed_files}")
    print(f"   Total de reemplazos: {total_replacements}")
    
    return fixed_files, total_replacements

def create_cmd_test_script():
    """Crear script CMD para ejecutar pruebas"""
    
    script_content = """@echo off
echo ========================================
echo MEGAMIND SYSTEM TESTER - CMD VERSION
echo Sistema de Cerebros Colaborativos
echo ========================================
echo.

cd /d "%~dp0"
echo Directorio actual: %CD%
echo.

echo Configurando Python path...
set PYTHONPATH=%CD%\src;%PYTHONPATH%
echo PYTHONPATH: %PYTHONPATH%
echo.

echo Ejecutando pruebas de MegaMind...
python tests/test_mega_mind_system.py

echo.
echo ========================================
echo Pruebas completadas
echo ========================================
pause
"""
    
    with open('test_megamind_fixed.cmd', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Script CMD creado: test_megamind_fixed.cmd")

def create_powershell_test_script():
    """Crear script PowerShell para ejecutar pruebas"""
    
    script_content = """# PowerShell Script para MegaMind Tester
Write-Host "========================================" -ForegroundColor Green
Write-Host "MEGAMIND SYSTEM TESTER - PowerShell" -ForegroundColor Green
Write-Host "Sistema de Cerebros Colaborativos" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""

# Configurar directorio
Set-Location $PSScriptRoot
Write-Host "Directorio actual: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""

# Configurar Python path
$env:PYTHONPATH = "$(Get-Location)\src;$env:PYTHONPATH"
Write-Host "PYTHONPATH: $env:PYTHONPATH" -ForegroundColor Yellow
Write-Host ""

# Ejecutar pruebas
Write-Host "Ejecutando pruebas de MegaMind..." -ForegroundColor Cyan
python tests/test_mega_mind_system.py

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "Pruebas completadas" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
"""
    
    with open('test_megamind.ps1', 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ Script PowerShell creado: test_megamind.ps1")

if __name__ == "__main__":
    print("🔧 ARREGLANDO SÍMBOLOS DE FOREX")
    print("=" * 50)
    
    # Arreglar símbolos
    fixed_files, total_replacements = fix_forex_symbols()
    
    print("\n🔧 CREANDO SCRIPTS DE PRUEBA")
    print("=" * 50)
    
    # Crear scripts de prueba
    create_cmd_test_script()
    create_powershell_test_script()
    
    print("\n✅ PROCESO COMPLETADO")
    print("=" * 50)
    print(f"📊 Archivos arreglados: {fixed_files}")
    print(f"📊 Total de reemplazos: {total_replacements}")
    print("\n🎯 Ahora puedes ejecutar:")
    print("   - test_megamind_fixed.cmd (en CMD)")
    print("   - .\\test_megamind.ps1 (en PowerShell)") 