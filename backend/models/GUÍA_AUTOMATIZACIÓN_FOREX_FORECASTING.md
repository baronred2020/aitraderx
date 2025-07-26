# GUÍA DE AUTOMATIZACIÓN: Sistema de Forecasting Forex

## 🚀 ¿Qué vas a automatizar?
- **Entrenamiento diario de modelos** (1 vez al día)
- **Monitoreo inteligente** (cada hora, o antes de eventos económicos importantes)

---

## 1. Estructura del Script

El script principal es:
```
backend/forex_forecasting_system.py
```

### Modos de uso:
- **Entrenamiento:**
  ```bash
  python forex_forecasting_system.py --train
  ```
- **Monitoreo inteligente:**
  ```bash
  python forex_forecasting_system.py --monitor
  ```
- **Flujo completo (manual):**
  ```bash
  python forex_forecasting_system.py
  ```

---

## 2. Automatización en **Windows** (Task Scheduler)

### **Entrenamiento diario**
1. Abre el **Programador de tareas** (Task Scheduler).
2. Crea una **nueva tarea básica**.
3. Ponle nombre: `Entrenamiento Forex AI`
4. Programa la tarea para que se ejecute **diariamente** a las 3:00 AM.
5. En "Acción", selecciona **Iniciar un programa**.
6. Programa:
   - **Programa:** `python`
   - **Argumentos:** `forex_forecasting_system.py --train`
   - **Iniciar en:** `C:\ruta\a\aitraderx\backend` (ajusta la ruta a tu carpeta)
7. Guarda y activa la tarea.

### **Monitoreo inteligente cada hora**
1. Repite los pasos 1-3, pero nombra la tarea: `Monitoreo Forex AI`
2. Programa la tarea para que se ejecute **cada hora** (elige "Diariamente" y en "Repetir cada: 1 hora").
3. En "Acción":
   - **Programa:** `python`
   - **Argumentos:** `forex_forecasting_system.py --monitor`
   - **Iniciar en:** `C:\ruta\a\aitraderx\backend`
4. Guarda y activa la tarea.

---

## 3. Automatización en **Linux** (cron)

### **Entrenamiento diario**
1. Abre la terminal y ejecuta:
   ```bash
   crontab -e
   ```
2. Agrega la siguiente línea para entrenar a las 3:00 AM:
   ```bash
   0 3 * * * cd /ruta/a/aitraderx/backend && python3 forex_forecasting_system.py --train
   ```

### **Monitoreo inteligente cada hora**
1. En el mismo archivo `crontab`, agrega:
   ```bash
   0 * * * * cd /ruta/a/aitraderx/backend && python3 forex_forecasting_system.py --monitor
   ```

---

## 4. Recomendaciones
- **No entrenes más de 1 vez al día** para evitar sobrecargar las APIs y tu equipo.
- **El monitoreo puede ejecutarse cada hora**: el script solo hará análisis si hay eventos importantes en la próxima hora.
- **Revisa los logs** o la consola para ver resultados y posibles errores.
- **Asegúrate de tener Python y las dependencias instaladas** (`requirements.txt`).
- **Ajusta las rutas** según la ubicación real de tu proyecto.

---

## 5. Troubleshooting (Solución de problemas)
- **No se encuentran modelos entrenados:**
  - Ejecuta primero el entrenamiento manualmente: `python forex_forecasting_system.py --train`
- **Error de permisos:**
  - En Linux, asegúrate de que el usuario del cron tenga permisos en la carpeta.
- **No se ejecuta el script:**
  - Verifica la ruta de Python (`python` vs `python3`).
  - Prueba ejecutar el comando manualmente en la terminal.
- **Problemas con las APIs:**
  - Revisa los límites de uso de Alpha Vantage y otras APIs.
  - Si hay errores de conexión, revisa tu red o las claves API.

---

## 6. Personalización avanzada
- Puedes ajustar la hora de entrenamiento y la frecuencia de monitoreo según tu estrategia.
- Puedes agregar notificaciones por email/SMS si lo deseas (requiere desarrollo extra).
- Si quieres logs automáticos, redirige la salida a un archivo:
  ```bash
  python forex_forecasting_system.py --monitor >> monitoreo.log 2>&1
  ```

---

## 7. Resumen visual

| Tarea                | Comando                                      | Frecuencia      |
|----------------------|----------------------------------------------|-----------------|
| Entrenamiento diario | python forex_forecasting_system.py --train    | 1 vez al día    |
| Monitoreo inteligente| python forex_forecasting_system.py --monitor  | Cada hora       |

---

¿Dudas? ¿Problemas? ¡Consulta este documento o pide ayuda! 