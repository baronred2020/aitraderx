# 🤖 Sistema de Automatización de Señales - Brain Trader

## 📋 Índice
1. [Introducción](#introducción)
2. [Opciones de Automatización](#opciones-de-automatización)
3. [Arquitectura del Sistema](#arquitectura-del-sistema)
4. [Componentes Necesarios](#componentes-necesarios)
5. [Base de Datos](#base-de-datos)
6. [Flujo Automático](#flujo-automático)
7. [Configuración por Usuario](#configuración-por-usuario)
8. [Implementación Sugerida](#implementación-sugerida)

---

## 🎯 Introducción

Actualmente, las señales en Brain Trader se generan **manualmente** cuando el usuario presiona el botón "Generar Señal Manual". Para hacer el sistema completamente automático, necesitamos implementar un **sistema de programación de tareas** que genere señales en los intervalos apropiados según el estilo de trading seleccionado.

### **Estado Actual vs Automatización**

| Aspecto | Manual | Automático |
|---------|--------|------------|
| **Activación** | Usuario presiona botón | Scheduler automático |
| **Frecuencia** | Cuando el usuario quiera | Según timeframe del estilo |
| **Validación** | Tiempo válido + Score > 70% | Misma lógica automática |
| **Notificación** | Mostrar en UI | Email, Push, Webhook |
| **Almacenamiento** | Temporal en sesión | Base de datos persistente |

---

## 🔄 Opciones de Automatización

### **1. Celery + Redis/RabbitMQ (Recomendado para Producción)**

#### **Ventajas:**
- ✅ Escalable y robusto
- ✅ Manejo avanzado de errores
- ✅ Monitoreo detallado
- ✅ Distribución de carga
- ✅ Persistencia de tareas

#### **Componentes:**
- **Celery**: Framework de tareas asíncronas
- **Redis/RabbitMQ**: Message broker
- **Celery Beat**: Scheduler integrado
- **Flower**: Dashboard de monitoreo

#### **Arquitectura:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Celery    │───▶│   Redis     │───▶│   Workers   │
│   Beat      │    │   Broker    │    │   (Tasks)   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### **2. APScheduler (Recomendado para Desarrollo)**

#### **Ventajas:**
- ✅ Fácil de implementar
- ✅ No requiere servicios externos
- ✅ Integración directa con FastAPI
- ✅ Configuración flexible

#### **Componentes:**
- **APScheduler**: Scheduler de Python
- **Background Threads**: Ejecución de tareas
- **SQLAlchemy**: Persistencia opcional

#### **Arquitectura:**
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ FastAPI     │───▶│ APScheduler │───▶│ Background  │
│ App         │    │ (Integrated)│    │ Tasks       │
└─────────────┘    └─────────────┘    └─────────────┘
```

### **3. Cron Jobs (Sistema Operativo)**

#### **Ventajas:**
- ✅ Simple y confiable
- ✅ No requiere código adicional
- ✅ Gestión del sistema operativo

#### **Desventajas:**
- ❌ Menos flexible
- ❌ Difícil de monitorear
- ❌ No integrado con la aplicación

---

## ⏰ Lógica de Programación

### **Intervalos por Estilo de Trading:**

| Estilo | Timeframe | Intervalo | Ejemplo de Horarios |
|--------|-----------|-----------|-------------------|
| **Scalping** | 5M | Cada 5 minutos | 00:00, 00:05, 00:10... |
| **Day Trading** | 15M | Cada 15 minutos | 00:00, 00:15, 00:30... |
| **Swing Trading** | 1H | Cada 1 hora | 00:00, 01:00, 02:00... |
| **Position Trading** | 1D | Cada 1 día | 00:00 (diario) |

### **Validaciones Automáticas:**

```python
# Pseudocódigo de validación
def should_generate_signal(style, current_time):
    # 1. Verificar si es tiempo válido
    if not _is_valid_signal_time(style, current_time):
        return False
    
    # 2. Verificar si el usuario tiene automatización activa
    if not user_automation_enabled:
        return False
    
    # 3. Verificar si ya se generó una señal recientemente
    if signal_generated_recently(style, current_time):
        return False
    
    return True
```

### **Criterios de Calidad:**

- ✅ **Score mínimo**: 70% (configurable por usuario)
- ✅ **Confianza del modelo**: > 60%
- ✅ **Indicadores técnicos**: Mínimo 2 confirmando
- ✅ **Tendencia**: ADX > 25 (opcional)

---

## 🏗️ Arquitectura del Sistema

### **Diagrama de Componentes:**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Scheduler     │───▶│   Signal        │───▶│   Notification  │
│   Service       │    │   Generator     │    │   Service       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Time          │    │   Quality       │    │   Email/Push    │
│   Validation    │    │   Check         │    │   Webhook       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User          │    │   Database      │    │   External      │
│   Settings      │    │   Storage       │    │   Integrations  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### **Flujo de Datos:**

1. **Scheduler** verifica cada minuto las tareas pendientes
2. **Time Validation** confirma si es momento de generar señal
3. **Signal Generator** ejecuta `generate_quality_signal`
4. **Quality Check** valida si score > umbral configurado
5. **Notification** envía alerta al usuario si es señal de calidad
6. **Database** registra la señal generada

---

## 🔧 Componentes Necesarios

### **1. Scheduler Service**

```python
class SignalScheduler:
    def __init__(self):
        self.scheduler = APScheduler()
        self.scheduler.start()
    
    def schedule_user_signals(self, user_id, brain_type, pair, style):
        # Programar tareas según configuración del usuario
        pass
    
    def cancel_user_signals(self, user_id):
        # Cancelar tareas del usuario
        pass
```

**Responsabilidades:**
- Programar tareas según timeframe
- Manejar diferentes estilos de trading
- Gestionar timezone del usuario
- Cancelar tareas cuando se desactiva

### **2. Signal Automation Service**

```python
class SignalAutomationService:
    def __init__(self):
        self.brain_trader_service = BrainTraderService()
    
    async def generate_automated_signal(self, user_id, brain_type, pair, style):
        # Lógica de generación automática
        pass
    
    def validate_signal_quality(self, signal_result):
        # Validar calidad de la señal
        pass
```

**Responsabilidades:**
- Llamar a `generate_quality_signal`
- Validar score de calidad
- Almacenar señales generadas
- Manejar errores de generación

### **3. Notification Service**

```python
class NotificationService:
    def __init__(self):
        self.email_service = EmailService()
        self.push_service = PushService()
    
    async def send_signal_notification(self, user_id, signal_data):
        # Enviar notificaciones
        pass
```

**Responsabilidades:**
- Enviar notificaciones por email
- Push notifications
- Webhooks para integración externa
- Templates de notificación

### **4. Configuration Service**

```python
class AutomationConfigService:
    def get_user_automation_settings(self, user_id):
        # Obtener configuración del usuario
        pass
    
    def update_user_automation_settings(self, user_id, settings):
        # Actualizar configuración
        pass
```

**Responsabilidades:**
- Configurar intervalos por usuario
- Activar/desactivar automatización
- Preferencias de notificación
- Gestión de timezones

---

## 📊 Base de Datos

### **Tabla: `automated_signals`**

```sql
CREATE TABLE automated_signals (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    brain_type VARCHAR(50) NOT NULL,
    pair VARCHAR(20) NOT NULL,
    style VARCHAR(50) NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- 'buy' or 'sell'
    quality_score DECIMAL(5,2) NOT NULL,
    entry_price DECIMAL(10,5) NOT NULL,
    stop_loss DECIMAL(10,5),
    take_profit DECIMAL(10,5),
    reasoning TEXT,
    indicators_used JSON,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'sent', 'failed'
    notification_sent_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Tabla: `user_automation_settings`**

```sql
CREATE TABLE user_automation_settings (
    id SERIAL PRIMARY KEY,
    user_id INTEGER UNIQUE NOT NULL,
    brain_type VARCHAR(50) NOT NULL DEFAULT 'brain_max',
    pair VARCHAR(20) NOT NULL DEFAULT 'EURUSD',
    style VARCHAR(50) NOT NULL DEFAULT 'day_trading',
    is_active BOOLEAN DEFAULT FALSE,
    min_quality_score DECIMAL(5,2) DEFAULT 70.0,
    notification_email BOOLEAN DEFAULT TRUE,
    notification_push BOOLEAN DEFAULT FALSE,
    notification_webhook VARCHAR(255),
    timezone VARCHAR(50) DEFAULT 'UTC',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **Tabla: `signal_notifications`**

```sql
CREATE TABLE signal_notifications (
    id SERIAL PRIMARY KEY,
    signal_id INTEGER REFERENCES automated_signals(id),
    notification_type VARCHAR(20) NOT NULL, -- 'email', 'push', 'webhook'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'sent', 'failed'
    sent_at TIMESTAMP,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔄 Flujo Automático

### **1. Inicialización del Sistema**

```python
# Al iniciar la aplicación
def initialize_automation_system():
    # 1. Cargar configuraciones de usuarios
    user_settings = load_all_user_automation_settings()
    
    # 2. Programar tareas para cada usuario activo
    for user_setting in user_settings:
        if user_setting.is_active:
            schedule_user_signals(user_setting)
    
    # 3. Iniciar scheduler
    scheduler.start()
```

### **2. Programación de Tareas**

```python
def schedule_user_signals(user_setting):
    # Obtener intervalo según estilo
    interval_minutes = get_interval_for_style(user_setting.style)
    
    # Programar tarea
    scheduler.add_job(
        func=generate_automated_signal,
        trigger='interval',
        minutes=interval_minutes,
        args=[user_setting.user_id, user_setting.brain_type, 
              user_setting.pair, user_setting.style],
        id=f"signal_{user_setting.user_id}_{user_setting.style}",
        replace_existing=True
    )
```

### **3. Generación de Señales**

```python
async def generate_automated_signal(user_id, brain_type, pair, style):
    try:
        # 1. Validar tiempo
        if not is_valid_signal_time(style):
            logger.info(f"Tiempo no válido para {style}")
            return
        
        # 2. Generar señal
        signal_result = await brain_trader_service.generate_quality_signal(
            brain_type, pair, style
        )
        
        # 3. Validar calidad
        if signal_result['quality_score'] < get_min_quality_score(user_id):
            logger.info(f"Señal de baja calidad: {signal_result['quality_score']}%")
            return
        
        # 4. Almacenar señal
        signal_id = await store_automated_signal(user_id, signal_result)
        
        # 5. Enviar notificaciones
        await send_signal_notifications(user_id, signal_id, signal_result)
        
        logger.info(f"Señal automática generada: {signal_id}")
        
    except Exception as e:
        logger.error(f"Error generando señal automática: {e}")
```

### **4. Notificaciones**

```python
async def send_signal_notifications(user_id, signal_id, signal_result):
    user_settings = get_user_automation_settings(user_id)
    
    # Email
    if user_settings.notification_email:
        await send_email_notification(user_id, signal_result)
    
    # Push notification
    if user_settings.notification_push:
        await send_push_notification(user_id, signal_result)
    
    # Webhook
    if user_settings.notification_webhook:
        await send_webhook_notification(user_settings.notification_webhook, signal_result)
    
    # Registrar notificaciones
    await log_notifications(signal_id, user_settings)
```

---

## ⚙️ Configuración por Usuario

### **Interfaz de Configuración:**

```typescript
interface AutomationSettings {
    // Configuración básica
    isActive: boolean;
    brainType: string;
    pair: string;
    style: string;
    
    // Calidad
    minQualityScore: number;
    
    // Notificaciones
    notificationEmail: boolean;
    notificationPush: boolean;
    notificationWebhook?: string;
    
    // Tiempo
    timezone: string;
}
```

### **Opciones de Configuración:**

#### **Activación/Desactivación:**
- ✅ Activar automatización
- ❌ Desactivar automatización

#### **Configuración de Señales:**
- 🧠 **Brain Type**: brain_max, brain_ultra, brain_predictor
- 💱 **Par de Divisas**: EURUSD, GBPUSD, USDJPY, etc.
- 📊 **Estilo de Trading**: scalping, day_trading, swing_trading, position_trading

#### **Calidad Mínima:**
- 📈 **Score mínimo**: 50% - 90% (configurable)
- ⚡ **Frecuencia**: Según estilo seleccionado

#### **Notificaciones:**
- 📧 **Email**: Activar/desactivar
- 📱 **Push**: Activar/desactivar
- 🔗 **Webhook**: URL personalizada
- ⏰ **Timezone**: Zona horaria del usuario

### **Ejemplo de Configuración:**

```json
{
    "isActive": true,
    "brainType": "brain_max",
    "pair": "EURUSD",
    "style": "day_trading",
    "minQualityScore": 75.0,
    "notificationEmail": true,
    "notificationPush": false,
    "notificationWebhook": "https://api.trading.com/webhook",
    "timezone": "America/New_York"
}
```

---

## 🚀 Implementación Sugerida

### **Fase 1: APScheduler (Desarrollo)**

1. **Instalar dependencias:**
   ```bash
   pip install apscheduler sqlalchemy
   ```

2. **Crear servicios básicos:**
   - `SignalScheduler`
   - `SignalAutomationService`
   - `NotificationService`

3. **Implementar base de datos:**
   - Tablas de configuración
   - Tablas de señales automáticas
   - Tablas de notificaciones

4. **Integrar con FastAPI:**
   - Endpoints de configuración
   - Dashboard de monitoreo
   - API de gestión

### **Fase 2: Celery (Producción)**

1. **Migrar a Celery:**
   - Configurar Redis/RabbitMQ
   - Implementar workers
   - Configurar Celery Beat

2. **Monitoreo avanzado:**
   - Flower dashboard
   - Métricas de rendimiento
   - Alertas de sistema

3. **Escalabilidad:**
   - Múltiples workers
   - Load balancing
   - High availability

### **Fase 3: Características Avanzadas**

1. **Machine Learning:**
   - Ajuste automático de umbrales
   - Predicción de calidad
   - Optimización de parámetros

2. **Integraciones:**
   - APIs de brokers
   - Webhooks personalizados
   - Notificaciones avanzadas

3. **Analytics:**
   - Métricas de rendimiento
   - Análisis de señales
   - Reportes automáticos

---

## 📈 Métricas y Monitoreo

### **Métricas Clave:**

- **Señales Generadas**: Total por día/semana/mes
- **Tasa de Éxito**: Señales con score > 70%
- **Tiempo de Respuesta**: Latencia de generación
- **Notificaciones Enviadas**: Email, push, webhook
- **Errores**: Fallos en generación o notificación

### **Dashboard de Monitoreo:**

```
┌─────────────────────────────────────────────────────────┐
│                    Brain Trader Automation              │
├─────────────────────────────────────────────────────────┤
│ 📊 Señales Hoy: 1,234 │ ✅ Éxito: 89% │ ⚠️ Errores: 11% │
├─────────────────────────────────────────────────────────┤
│ 🧠 brain_max: 567 │ 🧠 brain_ultra: 234 │ 🧠 brain_predictor: 433 │
├─────────────────────────────────────────────────────────┤
│ 📧 Email: 1,100 │ 📱 Push: 89 │ 🔗 Webhook: 45 │
└─────────────────────────────────────────────────────────┘
```

---

## 🔒 Seguridad y Consideraciones

### **Seguridad:**

- ✅ **Autenticación**: Verificar permisos de usuario
- ✅ **Rate Limiting**: Limitar generación de señales
- ✅ **Validación**: Verificar parámetros de entrada
- ✅ **Logging**: Registrar todas las acciones
- ✅ **Backup**: Respaldo de configuraciones

### **Consideraciones:**

- ⚠️ **Recursos**: Monitorear uso de CPU/memoria
- ⚠️ **API Limits**: Respetar límites de Yahoo Finance
- ⚠️ **Timezone**: Manejar diferentes zonas horarias
- ⚠️ **Escalabilidad**: Preparar para múltiples usuarios
- ⚠️ **Mantenimiento**: Actualizaciones sin interrumpir

---

## 📝 Conclusión

La automatización del sistema de señales transformará Brain Trader de una herramienta manual a un **sistema de trading automático** que:

- 🎯 **Genera señales** en los momentos óptimos
- 📊 **Valida calidad** automáticamente
- 📧 **Notifica al usuario** cuando encuentra oportunidades
- 💾 **Almacena historial** de todas las señales
- ⚙️ **Se adapta** a las preferencias de cada usuario

La implementación sugerida con **APScheduler** para desarrollo y **Celery** para producción proporcionará la flexibilidad y robustez necesarias para un sistema de trading automático profesional.

---

*Documento creado para el sistema Brain Trader - AI Trading Platform* 