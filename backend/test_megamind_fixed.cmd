@echo off
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
