# 🎯 LÍMITES DE CONSULTAS POR SUSCRIPCIÓN

## 📋 **RESUMEN EJECUTIVO**

Este documento identifica **exactamente qué consultas/APIs** deben tener límites por suscripción, basado en el consumo de recursos y la criticidad de cada operación.

### **🎯 OBJETIVO:**
Definir qué consultas consumen más recursos y deben ser limitadas para controlar costos y garantizar un servicio equitativo.

---

## 🔍 **CONSULTAS QUE DEBEN TENER LÍMITES**

### **🚨 1. CONSULTAS DE ALTO CONSUMO (CRÍTICAS)**

#### **🧠 Brain Trader Predictions**
```python
# Endpoints que consumen más recursos:
- GET /api/v1/brain-trader/predictions/{brain_type}
- GET /api/v1/brain-trader/predictions/{brain_type}/with-intervals
- GET /api/v1/brain-trader/signals/{brain_type}
- POST /api/v1/brain-trader/signals/{brain_type}/generate
```

**🔧 Razón del límite:**
- **Procesamiento intensivo** de modelos de IA
- **Cálculos complejos** de predicciones
- **Uso de GPU/CPU** para inferencia
- **Tiempo de respuesta** alto (2-5 segundos)

#### **🌟 Mega Mind Predictions**
```python
# Endpoints de fusión de cerebros:
- GET /api/v1/mega-mind/predictions
- GET /api/v1/mega-mind/collaboration
- GET /api/v1/mega-mind/arena
```

**🔧 Razón del límite:**
- **Fusión de múltiples modelos** IA
- **Procesamiento paralelo** intensivo
- **Análisis colaborativo** entre cerebros
- **Cálculos de confianza** complejos

#### **📊 Generación de Predicciones**
```python
# Endpoints de predicciones:
- POST /api/v1/predictions/generate
- GET /api/v1/predictions/active
```

**🔧 Razón del límite:**
- **Ejecución de modelos** de machine learning
- **Análisis de datos** históricos
- **Cálculos de indicadores** técnicos
- **Generación de señales** de trading

---

### **⚠️ 2. CONSULTAS DE CONSUMO MEDIO**

#### **📈 Backtesting**
```python
# Endpoints de backtesting:
- POST /api/backtest
- GET /api/backtest/results
- POST /api/backtest/optimize
```

**🔧 Razón del límite:**
- **Simulación de estrategias** históricas
- **Procesamiento de datos** masivos
- **Cálculos de métricas** de rendimiento
- **Optimización de parámetros**

#### **🔔 Alertas y Monitoreo**
```python
# Endpoints de alertas:
- POST /api/alerts
- GET /api/alerts/check
- POST /api/alerts/configure
```

**🔧 Razón del límite:**
- **Monitoreo continuo** de precios
- **Verificación de condiciones** en tiempo real
- **Envío de notificaciones** push/email
- **Almacenamiento** de configuraciones

#### **📊 Análisis Técnico Avanzado**
```python
# Endpoints de análisis:
- GET /api/technical-analysis/{symbol}
- GET /api/fundamental-analysis/{symbol}
- POST /api/analysis/custom
```

**🔧 Razón del límite:**
- **Cálculo de indicadores** complejos
- **Análisis de patrones** de mercado
- **Procesamiento de datos** fundamentales
- **Generación de reportes** personalizados

---

### **📋 3. CONSULTAS DE CONSUMO BAJO**

#### **💰 Wallet y Transacciones**
```python
# Endpoints de wallet:
- POST /api/v1/wallet/trade
- GET /api/v1/wallet/transactions
- POST /api/v1/wallet/recharge
```

**🔧 Razón del límite:**
- **Operaciones de base de datos** simples
- **Validaciones** de saldo
- **Registro de transacciones**
- **Cálculos** de P&L

#### **👤 Gestión de Usuario**
```python
# Endpoints de usuario:
- GET /api/auth/me
- PUT /api/auth/profile
- GET /api/subscriptions/me
```

**🔧 Razón del límite:**
- **Consultas simples** a base de datos
- **Validaciones** básicas
- **Actualizaciones** de perfil
- **Verificación** de suscripción

---

## 📊 **LÍMITES RECOMENDADOS POR PLAN**

### **🆓 STARTER (Gratuito)**
```python
LIMITS = {
    "daily_requests": 100,
    "brain_trader_predictions": 5,
    "mega_mind_predictions": 0,  # No disponible
    "prediction_generation": 3,
    "backtests": 2,
    "alerts": 3,
    "technical_analysis": 10,
    "wallet_operations": 20,
    "user_operations": 50
}
```

### **💼 TRADER ($29/mes)**
```python
LIMITS = {
    "daily_requests": 500,
    "brain_trader_predictions": 20,
    "mega_mind_predictions": 5,
    "prediction_generation": 10,
    "backtests": 10,
    "alerts": 15,
    "technical_analysis": 50,
    "wallet_operations": 100,
    "user_operations": 200
}
```

### **🚀 EXPERT ($99/mes)**
```python
LIMITS = {
    "daily_requests": 2000,
    "brain_trader_predictions": 100,
    "mega_mind_predictions": 50,
    "prediction_generation": 50,
    "backtests": 50,
    "alerts": 100,
    "technical_analysis": 200,
    "wallet_operations": 500,
    "user_operations": 1000
}
```

### **💎 PREMIUM ($299/mes)**
```python
LIMITS = {
    "daily_requests": 10000,
    "brain_trader_predictions": 500,
    "mega_mind_predictions": 200,
    "prediction_generation": 200,
    "backtests": 200,
    "alerts": 500,
    "technical_analysis": 1000,
    "wallet_operations": 2000,
    "user_operations": 5000
}
```

### **🏢 INSTITUTIONAL ($1,199/mes)**
```python
LIMITS = {
    "daily_requests": 50000,
    "brain_trader_predictions": 2000,
    "mega_mind_predictions": 1000,
    "prediction_generation": 1000,
    "backtests": 1000,
    "alerts": 2000,
    "technical_analysis": 5000,
    "wallet_operations": 10000,
    "user_operations": 25000
}
```

---

## 🔧 **CONSULTAS SIN LÍMITES**

### **✅ CONSULTAS PÚBLICAS (Siempre disponibles)**
```python
# Endpoints públicos sin límites:
- GET /api/market-data (datos básicos de precios)
- GET /api/candles (datos históricos básicos)
- GET /api/market-status (estado del mercado)
- GET /health (health check)
- GET /docs (documentación)
```

### **✅ CONSULTAS DE BAJO IMPACTO**
```python
# Endpoints de bajo impacto:
- GET /api/auth/login (autenticación)
- GET /api/subscriptions/plans (listado de planes)
- GET /api/v1/brain-trader/available-brains (cerebros disponibles)
- GET /api/v1/brain-trader/health (estado del servicio)
```

---

## 🛡️ **IMPLEMENTACIÓN DE LÍMITES**

### **📊 Sistema de Tracking**
```python
# Métricas a trackear por usuario:
TRACKING_METRICS = {
    "api_requests": "Contador general de requests",
    "brain_trader_predictions": "Predicciones de Brain Trader",
    "mega_mind_predictions": "Predicciones de Mega Mind",
    "prediction_generation": "Generación de predicciones",
    "backtests": "Ejecución de backtests",
    "alerts": "Creación de alertas",
    "technical_analysis": "Análisis técnico",
    "wallet_operations": "Operaciones de wallet",
    "user_operations": "Operaciones de usuario"
}
```

### **🔄 Reset Diario**
```python
# Reset automático diario:
DAILY_RESET = {
    "time": "00:00 UTC",
    "metrics": [
        "api_requests",
        "brain_trader_predictions", 
        "mega_mind_predictions",
        "prediction_generation",
        "technical_analysis",
        "wallet_operations",
        "user_operations"
    ]
}

# Reset mensual:
MONTHLY_RESET = {
    "time": "Primer día del mes 00:00 UTC",
    "metrics": [
        "backtests",
        "alerts"
    ]
}
```

### **⚡ Rate Limiting en Tiempo Real**
```python
# Límites por minuto/hora:
RATE_LIMITS = {
    "brain_trader_predictions": {
        "per_minute": 2,
        "per_hour": 10,
        "per_day": "plan_limit"
    },
    "mega_mind_predictions": {
        "per_minute": 1,
        "per_hour": 5,
        "per_day": "plan_limit"
    },
    "prediction_generation": {
        "per_minute": 1,
        "per_hour": 5,
        "per_day": "plan_limit"
    }
}
```

---

## 📈 **MONITOREO Y ALERTAS**

### **🔔 Alertas de Límites**
```python
# Alertas cuando se alcanzan límites:
ALERTS = {
    "80_percent": "Usuario al 80% del límite diario",
    "90_percent": "Usuario al 90% del límite diario",
    "95_percent": "Usuario al 95% del límite diario",
    "limit_reached": "Usuario alcanzó el límite diario"
}
```

### **📊 Dashboard de Uso**
```python
# Métricas para mostrar al usuario:
USER_DASHBOARD = {
    "current_usage": "Uso actual vs límite",
    "remaining_requests": "Requests restantes",
    "usage_by_category": "Uso por categoría",
    "reset_time": "Tiempo hasta reset",
    "upgrade_suggestions": "Sugerencias de upgrade"
}
```

---

## 💡 **RECOMENDACIONES**

### **✅ IMPLEMENTACIÓN INMEDIATA:**
1. **Límites críticos:** Brain Trader, Mega Mind, Predicciones
2. **Rate limiting:** Por minuto y hora
3. **Tracking:** Métricas en tiempo real
4. **Alertas:** Notificaciones al usuario

### **🔄 IMPLEMENTACIÓN FASE 2:**
1. **Backtesting:** Límites mensuales
2. **Alertas:** Límites por usuario
3. **Análisis técnico:** Límites por complejidad
4. **Wallet:** Límites por operación

### **🚀 OPTIMIZACIONES FUTURAS:**
1. **Cache inteligente:** Reducir consumo
2. **Priorización:** Por tipo de usuario
3. **Escalado dinámico:** Basado en carga
4. **Métricas avanzadas:** Análisis de patrones

---

## 📚 **REFERENCIAS**

### **🔗 Archivos del Sistema:**
- `backend/src/middleware/subscription_middleware.py`
- `backend/src/config/subscription_config.py`
- `backend/src/services/subscription_service.py`
- `docs/APIS_COMPLETAS_SISTEMA.md`

### **📊 Métricas de Consumo:**
- **Brain Trader:** 2-5 segundos por request
- **Mega Mind:** 3-8 segundos por request
- **Backtesting:** 10-60 segundos por request
- **Market Data:** 0.1-0.5 segundos por request

---

*Documento creado: Enero 2025*
*Última actualización: Enero 2025*
*Versión: 1.0* 