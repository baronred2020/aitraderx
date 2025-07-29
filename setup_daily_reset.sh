#!/bin/bash

echo "Configurando cron job para reinicio diario de predicciones..."

# Obtener la ruta actual
CURRENT_DIR=$(pwd)
PYTHON_PATH=$(which python3)
SCRIPT_PATH="$CURRENT_DIR/backend/reset_daily_predictions.py"

# Crear el cron job para ejecutar a las 00:00 todos los días
CRON_JOB="0 0 * * * $PYTHON_PATH $SCRIPT_PATH"

# Agregar al crontab
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

if [ $? -eq 0 ]; then
    echo "✅ Cron job creado exitosamente"
    echo "📅 Se ejecutará todos los días a las 00:00"
    echo "🔍 Para verificar: crontab -l"
else
    echo "❌ Error creando el cron job"
fi