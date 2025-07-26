# 🚀 AITRADERX - Documentación del Sistema

## 📋 **Resumen Ejecutivo**

AITRADERX es una plataforma de trading con inteligencia artificial que ofrece diferentes niveles de suscripción para acceder a herramientas avanzadas de análisis y trading automatizado. El sistema está diseñado con una arquitectura moderna que incluye frontend React/TypeScript, backend FastAPI/Python, y base de datos MySQL.

---

## 🏗️ **Arquitectura del Sistema**

### **Frontend (React + TypeScript)**
- **Framework**: React 18 con TypeScript
- **Styling**: Tailwind CSS con diseño moderno y responsive
- **Estado**: Context API para autenticación y suscripciones
- **Iconos**: Lucide React para interfaz consistente
- **Puerto**: 3000

### **Backend (FastAPI + Python)**
- **Framework**: FastAPI con Python 3.13
- **Base de datos**: MySQL con SQLAlchemy
- **Autenticación**: JWT tokens
- **APIs**: RESTful con WebSocket para datos en tiempo real
- **Puerto**: 8000

### **Base de Datos (MySQL + XAMPP)**
- **Motor**: MySQL 8.0
- **Gestión**: XAMPP para desarrollo local
- **Migraciones**: Alembic para control de versiones
- **Puerto**: 3306

---

## 💳 **Sistema de Suscripciones**

### **Plan Freemium (Gratuito)**
**Precio**: $0/mes
**Características**:
- ✅ Dashboard básico
- ✅ Información de mercado limitada
- ✅ Acceso a documentación

**Restricciones**:
- ❌ No acceso a herramientas de trading
- ❌ No análisis técnico avanzado
- ❌ No alertas personalizadas
- ❌ No IA predictiva

### **Plan Basic ($29/mes)**
**Precio**: $29/mes
**Características**:
- ✅ Dashboard completo
- ✅ Trading manual con 5 pares
- ✅ Portfolio básico
- ✅ Análisis técnico básico
- ✅ Sistema de alertas
- ✅ Soporte por email

**Funcionalidades adicionales**:
- 📊 Indicadores técnicos básicos
- 🔔 Alertas de precio y volumen
- 💼 Gestión de portfolio
- 📈 Gráficos en tiempo real

### **Plan Pro ($99/mes)**
**Precio**: $99/mes
**Características**:
- ✅ Todo del plan Basic
- ✅ Monitor de IA en tiempo real
- ✅ Reinforcement Learning
- ✅ Reportes avanzados
- ✅ Integración MT4
- ✅ Soporte prioritario

**Funcionalidades adicionales**:
- 🤖 Monitor de modelos IA
- ⚡ Agente de Reinforcement Learning
- 📋 Reportes detallados
- 🔗 Integración con MetaTrader 4
- 📊 Análisis fundamental avanzado

### **Plan Elite ($299/mes)**
**Precio**: $299/mes
**Características**:
- ✅ Todo del plan Pro
- ✅ Acceso completo a API
- ✅ Modelos personalizados
- ✅ Soporte 24/7
- ✅ Comunidad exclusiva
- ✅ Estrategias premium

**Funcionalidades adicionales**:
- 🔌 API completa para desarrollo
- 🧠 Modelos IA personalizados
- 👥 Comunidad de traders elite
- 🎯 Estrategias exclusivas
- 📞 Soporte telefónico 24/7

---

## 🤖 **Sistemas de Inteligencia Artificial Integrados**

### **1. IA Tradicional (Machine Learning)**
**Tecnologías**:
- **Algoritmos**: Random Forest, LSTM, XGBoost
- **Framework**: Scikit-learn, TensorFlow
- **Funcionalidad**: Predicción de precios y señales de trading

**Características**:
- 📊 Análisis de patrones históricos
- 🎯 Predicción de movimientos de precio
- 📈 Indicadores de confianza
- 🔄 Entrenamiento automático

### **2. Reinforcement Learning (RL)**
**Tecnologías**:
- **Framework**: Gymnasium, Stable-Baselines3
- **Algoritmos**: DQN (Deep Q-Network) + PPO (Proximal Policy Optimization)
- **Funcionalidad**: Optimización de estrategias de trading

**Características**:
- ⚡ Aprendizaje continuo
- 🎮 Simulación de trading
- 📊 Optimización de parámetros
- 🔄 Adaptación a cambios de mercado

#### **DQN (Deep Q-Network)**:
- **Tipo**: Decisiones discretas (comprar/vender/esperar)
- **Uso**: Estrategias simples y directas
- **Ventaja**: Rápido y eficiente

#### **PPO (Proximal Policy Optimization)**:
- **Tipo**: Decisiones continuas y complejas
- **Uso**: Estrategias avanzadas con múltiples parámetros
- **Ventaja**: Más estable y preciso

### **3. Auto-Training System**
**Tecnologías**:
- **Framework**: Custom Python
- **Funcionalidad**: Entrenamiento automático de modelos

**Características**:
- 🤖 Entrenamiento automático
- 📊 Monitoreo de rendimiento
- 🔄 Actualización de modelos
- 📈 Optimización continua

---

## 🛠️ **Funcionalidades Implementadas**

### **Sistema de Autenticación**
- ✅ Login/logout con JWT
- ✅ Control de sesiones
- ✅ Verificación de suscripciones
- ✅ Middleware de autorización

### **Dashboard Principal**
- ✅ Métricas en tiempo real
- ✅ Gráficos interactivos
- ✅ Información de portfolio
- ✅ Estado de sistemas IA

### **Control de Acceso**
- ✅ Verificación por plan de suscripción
- ✅ Modal de upgrade automático
- ✅ Redirección a secciones permitidas
- ✅ Información de plan actual

### **Interfaz de Usuario**
- ✅ Diseño responsive
- ✅ Navegación intuitiva
- ✅ Menú de usuario con información
- ✅ Loading screens con branding

### **Sistema de Alertas**
- ✅ Alertas de precio
- ✅ Notificaciones de volumen
- ✅ Señales de trading
- ✅ Configuración personalizada

---

## 📊 **Componentes del Frontend**

### **Páginas Principales**
1. **Login** (`/components/Auth/Login.tsx`)
   - Formulario de autenticación
   - Validación de credenciales
   - Manejo de errores

2. **Dashboard** (`/components/Dashboard/Dashboard.tsx`)
   - Métricas principales
   - Gráficos de rendimiento
   - Estado de sistemas

3. **Trading** (`/components/Trading/TradingView.tsx`)
   - Interfaz de trading
   - Gráficos en tiempo real
   - Órdenes de compra/venta

4. **Portfolio** (`/components/Portfolio/`)
   - Gestión de posiciones
   - Análisis de rendimiento
   - Historial de operaciones

5. **Análisis** (`/components/Analysis/`)
   - Análisis técnico
   - Análisis fundamental
   - Indicadores avanzados

6. **Monitor IA** (`/components/AIMonitor/`)
   - Estado de modelos IA
   - Métricas de rendimiento
   - Configuración de parámetros

7. **Reinforcement Learning** (`/components/RL/RLDashboard.tsx`)
   - Estado del agente RL
   - Métricas de aprendizaje
   - Configuración de estrategias

### **Componentes Comunes**
- **Layout** (`/components/Common/Layout.tsx`)
- **LoadingSpinner** (`/components/Common/LoadingSpinner.tsx`)
- **UpgradeModal** (`/components/Common/UpgradeModal.tsx`)
- **MobileNav** (`/components/Common/MobileNav.tsx`)

---

## 🔧 **APIs del Backend**

### **Endpoints de Autenticación**
- `POST /api/auth/login` - Login de usuario
- `POST /api/auth/logout` - Logout de usuario
- `GET /api/auth/me` - Información del usuario actual

### **Endpoints de Suscripciones**
- `GET /api/subscriptions/me` - Suscripción actual
- `GET /api/subscriptions/plans` - Planes disponibles
- `POST /api/subscriptions/upgrade` - Upgrade de plan

### **Endpoints de Trading**
- `GET /api/assets` - Activos recomendados
- `GET /api/price-data/{symbol}` - Datos de precio
- `POST /api/trading/order` - Colocar orden

### **Endpoints de IA**
- `GET /api/model/status` - Estado de modelos IA
- `GET /api/rl/status` - Estado de RL
- `POST /api/predict-price` - Predicción de precios

### **Endpoints de Análisis**
- `GET /api/technical-analysis/{symbol}` - Análisis técnico
- `GET /api/fundamental-analysis/{symbol}` - Análisis fundamental
- `GET /api/alerts` - Alertas activas

### **WebSocket**
- `WS /ws` - Datos en tiempo real

---

## 🗄️ **Estructura de Base de Datos**

### **Tablas Principales**
1. **users** - Información de usuarios
2. **subscriptions** - Suscripciones y planes
3. **trading_portfolios** - Portfolios de trading
4. **ai_models** - Modelos de IA
5. **alerts** - Sistema de alertas
6. **audit_logs** - Logs de auditoría

### **Relaciones**
- Usuario → Suscripción (1:1)
- Usuario → Portfolio (1:1)
- Usuario → Alertas (1:N)
- Usuario → Logs (1:N)

---

## 🚀 **Estado Actual del Desarrollo**

### **✅ Completado**
- Sistema de autenticación
- Control de acceso por suscripción
- Dashboard principal
- Interfaz de usuario responsive
- Sistema de loading con branding
- Modal de upgrade
- Base de datos configurada
- APIs básicas implementadas

### **🔄 En Desarrollo**
- Integración completa con MySQL
- Sistema de pagos
- Funcionalidades avanzadas de trading
- Reportes detallados
- Comunidad de usuarios

### **📋 Pendiente**
- Sistema de notificaciones push
- App móvil
- Integración con exchanges
- Backtesting avanzado
- Machine Learning en producción

---

## 🛡️ **Seguridad Implementada**

### **Autenticación**
- JWT tokens seguros
- Expiración automática
- Refresh tokens
- Middleware de autorización

### **Autorización**
- Verificación de suscripciones
- Control de acceso por plan
- Validación de permisos
- Logs de auditoría

### **Protección de Datos**
- Encriptación de contraseñas
- Sanitización de inputs
- Validación de datos
- Rate limiting

---

## 📈 **Métricas y Monitoreo**

### **Frontend**
- Tiempo de carga de páginas
- Errores de JavaScript
- Rendimiento de componentes
- Uso de memoria

### **Backend**
- Tiempo de respuesta de APIs
- Uso de CPU y memoria
- Errores de base de datos
- Latencia de WebSocket

### **IA**
- Precisión de predicciones
- Tiempo de entrenamiento
- Uso de recursos
- Métricas de rendimiento

---

## 🎯 **Próximos Pasos**

### **Corto Plazo (1-2 semanas)**
1. Completar integración con MySQL
2. Implementar sistema de pagos
3. Finalizar funcionalidades de trading
4. Testing completo del sistema

### **Mediano Plazo (1-2 meses)**
1. Despliegue en producción
2. Optimización de rendimiento
3. Implementación de ML en producción
4. Sistema de notificaciones

### **Largo Plazo (3-6 meses)**
1. App móvil nativa
2. Integración con más exchanges
3. IA más avanzada
4. Expansión internacional

---

## 📞 **Soporte y Contacto**

### **Desarrollo**
- **Backend**: FastAPI + Python
- **Frontend**: React + TypeScript
- **Base de datos**: MySQL + XAMPP
- **IA**: Scikit-learn + TensorFlow + Gymnasium

### **Credenciales de Desarrollo**
- **Usuario**: admin
- **Contraseña**: admin123
- **Plan**: Elite (acceso completo)

### **Puertos**
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:8000
- **Base de datos**: localhost:3306

---

## 🎉 **Conclusión**

AITRADERX es una plataforma robusta y escalable que combina las mejores tecnologías de desarrollo web con sistemas avanzados de inteligencia artificial para ofrecer una experiencia de trading única y personalizada según el nivel de suscripción del usuario.

El sistema está diseñado para crecer y adaptarse a las necesidades de los usuarios, con una arquitectura modular que permite agregar nuevas funcionalidades y mejorar los sistemas de IA de manera continua.

---

*Documentación actualizada: Julio 2025*
*Versión del sistema: 1.0.0* 