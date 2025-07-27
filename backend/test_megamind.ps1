# PowerShell Script para MegaMind Tester
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
