@echo off
echo Configurando tarea programada para reinicio diario de predicciones...

REM Obtener la ruta actual
set CURRENT_DIR=%~dp0
set PYTHON_PATH=python
set SCRIPT_PATH=%CURRENT_DIR%backend\reset_daily_predictions.py

REM Crear la tarea programada para ejecutar a las 00:00 todos los días
schtasks /create /tn "AI Trader Daily Reset" /tr "%PYTHON_PATH% %SCRIPT_PATH%" /sc daily /st 00:00 /f

if %ERRORLEVEL% EQU 0 (
    echo ✅ Tarea programada creada exitosamente
    echo 📅 Se ejecutará todos los días a las 00:00
    echo 🔍 Para verificar: schtasks /query /tn "AI Trader Daily Reset"
) else (
    echo ❌ Error creando la tarea programada
    echo 💡 Asegúrate de ejecutar como administrador
)

pause