# Sistema de Suscripciones - AI Trading Platform

## 📋 Descripción General

El sistema de suscripciones implementa un modelo de negocio con 4 niveles de planes que diferencian el acceso a las capacidades de IA según el nivel del usuario. El sistema incluye gestión de usuarios, verificación de permisos, métricas de uso y APIs completas.

## 🎯 Planes de Suscripción

### 1. FREEMIUM ($0/mes)
**Objetivo:** Onboarding y conversión

**Capacidades:**
- ✅ AI Tradicional básica (señales simples)
- ✅ 1 indicador técnico (RSI)
- ✅ Predicciones limitadas (3 días)
- ✅ Backtesting básico (30 días)
- ✅ 1 par de trading (EUR/USD)
- ✅ 3 alertas básicas

**Límites:**
- 100 requests diarios
- 10 predicciones por día
- 5 backtests por mes

### 2. BÁSICO ($29/mes)
**Objetivo:** Usuarios serios

**Capacidades:**
- ✅ AI Tradicional completa
- ✅ 3 indicadores técnicos (RSI, MACD, Bollinger)
- ✅ Predicciones mejoradas (7 días)
- ✅ Backtesting avanzado (90 días)
- ✅ 5 pares de trading
- ✅ 10 alertas avanzadas
- ✅ Gráficos avanzados

**Límites:**
- 500 requests diarios
- 50 predicciones por día
- 20 backtests por mes

### 3. PRO ($99/mes)
**Objetivo:** Traders profesionales

**Capacidades:**
- ✅ AI Tradicional Premium + LSTM
- ✅ Reinforcement Learning (DQN)
- ✅ Ensemble AI (Tradicional + RL)
- ✅ Todos los indicadores técnicos
- ✅ Predicciones avanzadas (14 días)
- ✅ Backtesting profesional
- ✅ Todos los pares de trading
- ✅ Risk Management básico
- ✅ Portfolio Optimization básico
- ✅ Integración MT4 básica
- ✅ RL Dashboard
- ✅ AI Monitor

**Límites:**
- 2000 requests diarios
- 200 predicciones por día
- 100 backtests por mes

### 4. ELITE ($299/mes)
**Objetivo:** Traders institucionales

**Capacidades:**
- ✅ AI Tradicional Elite + máxima precisión
- ✅ Reinforcement Learning completo (DQN + PPO)
- ✅ Ensemble AI avanzado optimizado
- ✅ Predicciones elite (30 días)
- ✅ Backtesting institucional
- ✅ Todos los instrumentos (Forex, Stocks, Crypto)
- ✅ Risk Management avanzado
- ✅ Portfolio Optimization avanzado
- ✅ Auto-Trading con AI
- ✅ Custom Models personalizados
- ✅ Integración MT4 completa
- ✅ API personalizada
- ✅ Soporte prioritario 24/7

**Límites:**
- 10000 requests diarios
- 1000 predicciones por día
- 500 backtests por mes

## 🏗️ Arquitectura del Sistema

### Estructura de Archivos

```
backend/src/
├── models/
│   └── subscription.py          # Modelos de datos
├── services/
│   └── subscription_service.py  # Lógica de negocio
├── api/
│   └── subscription_routes.py   # Endpoints de API
├── middleware/
│   └── subscription_middleware.py # Middleware de verificación
├── config/
│   └── subscription_config.py   # Configuraciones
├── tests/
│   └── test_subscription_system.py # Tests unitarios
└── demo_subscription_system.py  # Script de demostración
```

### Componentes Principales

#### 1. Modelos de Datos (`models/subscription.py`)
- `SubscriptionPlan`: Define planes y sus características
- `UserSubscription`: Gestiona suscripciones de usuarios
- `UsageMetrics`: Trackea métricas de uso
- `AICapabilities`: Capacidades de IA por plan
- `APILimits`: Límites de API por plan
- `UIFeatures`: Características de UI por plan

#### 2. Servicio de Suscripciones (`services/subscription_service.py`)
- Gestión de planes y usuarios
- Verificación de permisos
- Métricas de uso
- Upgrades y cancelaciones
- Persistencia de datos

#### 3. API Routes (`api/subscription_routes.py`)
- Endpoints para gestión de planes
- Endpoints para usuarios
- Endpoints para permisos
- Endpoints para métricas
- Endpoints de administración

#### 4. Middleware (`middleware/subscription_middleware.py`)
- Verificación automática de permisos
- Actualización de métricas de uso
- Control de acceso por endpoint

## 🔌 API Endpoints

### Planes
```
GET    /api/subscriptions/plans              # Obtener todos los planes
GET    /api/subscriptions/plans/{plan_type}  # Obtener plan específico
```

### Usuarios
```
POST   /api/subscriptions/users/{user_id}/subscribe    # Crear suscripción
GET    /api/subscriptions/users/{user_id}/subscription # Obtener suscripción
GET    /api/subscriptions/users/{user_id}/plan         # Obtener plan de usuario
POST   /api/subscriptions/users/{user_id}/upgrade      # Upgrade de suscripción
DELETE /api/subscriptions/users/{user_id}/subscription # Cancelar suscripción
```

### Permisos
```
POST   /api/subscriptions/users/{user_id}/permissions/check    # Verificar permisos
GET    /api/subscriptions/users/{user_id}/permissions/features # Obtener características
```

### Métricas
```
GET    /api/subscriptions/users/{user_id}/usage        # Obtener métricas de uso
POST   /api/subscriptions/users/{user_id}/usage/update # Actualizar métricas
```

### Administración
```
GET    /api/subscriptions/admin/stats                  # Estadísticas del sistema
POST   /api/subscriptions/admin/maintenance/check-expiry # Verificar expiración
POST   /api/subscriptions/admin/maintenance/reset-usage  # Reset métricas
GET    /api/subscriptions/health                       # Health check
```

## 🔐 Sistema de Permisos

### Características Verificadas

#### AI Capabilities
- `traditional_ai`: AI Tradicional (Random Forest, etc.)
- `reinforcement_learning`: RL (DQN, PPO)
- `ensemble_ai`: Ensemble de modelos
- `lstm_predictions`: Predicciones LSTM
- `custom_models`: Modelos personalizados
- `auto_training`: Auto-entrenamiento

#### UI Features
- `advanced_charts`: Gráficos avanzados
- `multiple_timeframes`: Múltiples timeframes
- `rl_dashboard`: Dashboard de RL
- `ai_monitor`: Monitor de IA
- `mt4_integration`: Integración MT4
- `api_access`: Acceso a API
- `custom_reports`: Reportes personalizados
- `priority_support`: Soporte prioritario

#### API Limits
- `api_requests`: Requests diarios
- `predictions`: Predicciones por día
- `backtests`: Backtests por mes
- `alerts`: Límite de alertas
- `trading_pairs`: Pares de trading
- `portfolios`: Portafolios
- `indicators`: Indicadores

### Ejemplo de Verificación

```python
# Verificar si usuario puede usar RL
allowed, message = subscription_service.check_user_permissions(
    user_id="user123",
    feature="reinforcement_learning"
)

if allowed:
    # Usar RL
    pass
else:
    # Mostrar mensaje de upgrade
    print(message)
```

## 📊 Métricas de Uso

### Métricas Trackeadas
- `api_requests_today`: Requests API del día
- `predictions_made_today`: Predicciones realizadas
- `backtests_run_today`: Backtests ejecutados
- `alerts_created`: Alertas creadas
- `rl_episodes_trained`: Episodios de RL entrenados
- `custom_models_created`: Modelos personalizados creados
- `trades_executed`: Trades ejecutados

### Reset Automático
- Las métricas se resetean diariamente
- Se pueden configurar resets semanales o mensuales
- Sistema de alertas cuando se alcanza 80% del límite

## 🧪 Testing

### Ejecutar Tests
```bash
cd backend/src
python -m pytest tests/test_subscription_system.py -v
```

### Tests Incluidos
- Creación de planes por defecto
- Gestión de suscripciones de usuarios
- Verificación de permisos
- Límites de uso
- Upgrades y cancelaciones
- Métricas de uso
- Persistencia de datos

## 🚀 Demo

### Ejecutar Demo
```bash
cd backend/src
python demo_subscription_system.py
```

### Funcionalidades del Demo
1. Mostrar planes disponibles
2. Crear usuarios de ejemplo
3. Verificar permisos por plan
4. Demostrar límites de uso
5. Simular upgrades de suscripción
6. Mostrar métricas de uso
7. Estadísticas del sistema
8. Cancelación de suscripciones
9. Características por plan

## 🔧 Configuración

### Variables de Entorno
```bash
SUBSCRIPTION_DATA_DIR=data/subscriptions
SUBSCRIPTION_LOGS_DIR=logs/subscriptions
```

### Archivos de Datos
- `plans.json`: Configuración de planes
- `subscriptions.json`: Suscripciones de usuarios
- `usage.json`: Métricas de uso

## 📈 Integración con Frontend

### Headers Requeridos
```javascript
// En requests del frontend
headers: {
    'X-User-ID': 'user123',
    'Content-Type': 'application/json'
}
```

### Verificación de Permisos
```javascript
// Verificar si feature está disponible
const checkPermission = async (feature) => {
    const response = await fetch(`/api/subscriptions/users/${userId}/permissions/check?feature=${feature}`);
    const data = await response.json();
    return data.allowed;
};
```

### Manejo de Errores
```javascript
// Manejar errores de permisos
if (response.status === 403) {
    const error = await response.json();
    if (error.upgrade_required) {
        showUpgradeModal();
    }
}
```

## 🔄 Flujo de Trabajo

### 1. Crear Suscripción
```python
# Backend
subscription = service.create_user_subscription(
    user_id="user123",
    plan_type=PlanType.BASIC,
    trial_days=7
)
```

### 2. Verificar Permisos
```python
# Middleware automático
allowed, message = service.check_user_permissions(
    user_id="user123",
    feature="reinforcement_learning"
)
```

### 3. Actualizar Métricas
```python
# Automático en middleware
service.update_usage_metrics(user_id, "predictions")
```

### 4. Upgrade de Plan
```python
# Frontend solicita upgrade
new_subscription = service.upgrade_user_subscription(
    user_id="user123",
    new_plan_type=PlanType.PRO
)
```

## 🎯 Próximos Pasos

### Fase 2: Frontend Integration
1. Componentes de UI por plan
2. Sistema de upgrade/cancelación
3. Dashboard de métricas de uso
4. Modales de upgrade

### Fase 3: Payment Integration
1. Integración con Stripe/PayPal
2. Gestión de pagos
3. Facturación automática
4. Gestión de trials

### Fase 4: Analytics
1. Métricas de conversión
2. Análisis de uso
3. A/B testing
4. Optimización de precios

## 📞 Soporte

### Niveles de Soporte por Plan
- **Freemium**: Community forum (72h)
- **Básico**: Email support (48h)
- **Pro**: Email + Chat (24h)
- **Elite**: Phone + Dedicated manager (4h)

### Contacto
- Email: support@aitrading.com
- Community: forum.aitrading.com
- Documentation: docs.aitrading.com 