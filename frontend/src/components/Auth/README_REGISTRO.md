# Sistema de Registro AITRADERX

## Descripción General

El sistema de registro permite a los usuarios crear cuentas en AITRADERX con dos opciones principales:

1. **Registro Gratuito (Freemium)**: Acceso básico sin costo
2. **Registro con Plan Premium**: Acceso completo con pago inmediato

## Características Implementadas

### ✅ Funcionalidades Completadas

#### Frontend (React + TypeScript)
- **Formulario de Registro Completo**
  - Campos: Nombre, Apellido, Usuario, Email, Teléfono, Contraseña
  - Validaciones en tiempo real
  - Confirmación de contraseña
  - Indicadores de fortaleza de contraseña

- **Selección de Planes**
  - Visualización de todos los planes disponibles
  - Comparación de características
  - Selección interactiva con iconos
  - Resumen del plan seleccionado

- **Sistema de Pago**
  - Múltiples métodos de pago (Stripe, PayPal, Crypto)
  - Interfaz intuitiva para selección
  - Procesamiento simulado de pagos
  - Validación de métodos de pago

- **Experiencia de Usuario**
  - Diseño responsive y moderno
  - Transiciones suaves
  - Estados de carga
  - Mensajes de error y éxito
  - Auto-login después del registro

#### Backend (FastAPI + Python)
- **Endpoints de Autenticación**
  - `/api/auth/register` - Registro de usuarios
  - `/api/auth/login` - Inicio de sesión
  - `/api/auth/logout` - Cierre de sesión
  - `/api/auth/me` - Información del usuario actual

- **Validaciones de Seguridad**
  - Validación de contraseñas (longitud, complejidad)
  - Validación de emails (formato correcto)
  - Validación de nombres de usuario
  - Prevención de usuarios duplicados

- **Sistema de Planes**
  - Integración con sistema de suscripciones
  - Creación automática de suscripciones
  - Gestión de planes freemium y premium

- **JWT Authentication**
  - Tokens seguros con expiración
  - Configuración flexible
  - Manejo de sesiones

## Estructura de Archivos

```
frontend/src/components/Auth/
├── Login.tsx              # Componente de login
├── Register.tsx           # Componente de registro
└── README_REGISTRO.md    # Esta documentación

backend/src/
├── api/
│   ├── auth_routes.py    # Endpoints de autenticación
│   └── subscription_routes.py
├── config/
│   └── auth_config.py    # Configuración de autenticación
└── services/
    └── subscription_service.py
```

## Planes Disponibles

### 🆓 Freemium (Gratis)
- **Precio**: $0/mes
- **Características**:
  - Señales básicas de trading
  - 1 indicador técnico (RSI)
  - Predicciones limitadas (3 días)
  - Backtesting básico (30 días)
  - 1 par de trading (EUR/USD)
  - 3 alertas básicas

### 💳 Básico ($29/mes)
- **Características**:
  - AI Tradicional completa
  - 3 indicadores técnicos
  - Predicciones mejoradas (7 días)
  - Backtesting avanzado (90 días)
  - 5 pares de trading
  - 10 alertas avanzadas

### 👑 Pro ($99/mes)
- **Características**:
  - AI Tradicional + LSTM
  - Reinforcement Learning (DQN)
  - Todos los indicadores técnicos
  - Predicciones avanzadas (14 días)
  - Backtesting profesional
  - Todos los pares de trading

### 🏆 Elite ($299/mes)
- **Características**:
  - AI Tradicional Elite
  - Reinforcement Learning completo
  - Ensemble AI avanzado
  - Predicciones elite (30 días)
  - Backtesting institucional
  - Auto-Trading con AI
  - Custom Models personalizados

## Métodos de Pago

### 💳 Tarjeta de Crédito/Débito
- **Proveedor**: Stripe
- **Tarjetas**: Visa, Mastercard, American Express
- **Seguridad**: PCI DSS compliant

### 🔵 PayPal
- **Proveedor**: PayPal
- **Ventajas**: Pago seguro y rápido
- **Monedas**: Múltiples monedas soportadas

### ₿ Criptomonedas
- **Monedas**: Bitcoin, Ethereum, USDT
- **Proveedor**: Integración directa
- **Ventajas**: Pagos anónimos y descentralizados

## Flujo de Registro

### 1. Registro Gratuito
```
Usuario llena formulario → Validación → Creación cuenta freemium → Auto-login → Dashboard
```

### 2. Registro con Pago
```
Usuario llena formulario → Selecciona plan → Elige método de pago → 
Procesamiento de pago → Creación cuenta premium → Auto-login → Dashboard
```

## Validaciones Implementadas

### Frontend
- ✅ Campos obligatorios
- ✅ Formato de email
- ✅ Longitud de contraseña (mínimo 8 caracteres)
- ✅ Confirmación de contraseña
- ✅ Selección de plan para pagos
- ✅ Selección de método de pago

### Backend
- ✅ Validación de contraseña (complejidad)
- ✅ Validación de email (formato)
- ✅ Validación de username (longitud y caracteres)
- ✅ Verificación de usuario duplicado
- ✅ Validación de método de pago
- ✅ Creación de suscripción automática

## Seguridad

### 🔐 Medidas Implementadas
- **Contraseñas**: Hash SHA-256
- **JWT**: Tokens con expiración
- **Validaciones**: Múltiples capas
- **Rate Limiting**: Configurado
- **CORS**: Configurado para desarrollo

### 🚀 Próximas Mejoras
- [ ] Verificación de email
- [ ] Autenticación de dos factores (2FA)
- [ ] Integración real con Stripe/PayPal
- [ ] Rate limiting por IP
- [ ] Logs de auditoría
- [ ] Encriptación de datos sensibles

## Configuración

### Variables de Entorno (Backend)
```bash
SECRET_KEY=your-secret-key
ACCESS_TOKEN_EXPIRE_MINUTES=30
PASSWORD_MIN_LENGTH=8
EMAIL_VERIFICATION_REQUIRED=false
```

### Configuración de Desarrollo
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn src.main:app --reload

# Frontend
cd frontend
npm install
npm start
```

## Uso

### Acceso al Registro
1. Ir a `http://localhost:3000/register`
2. Llenar el formulario
3. Elegir plan (gratis o premium)
4. Si es premium, seleccionar método de pago
5. Completar registro

### Credenciales de Desarrollo
- **Usuario**: admin
- **Contraseña**: admin123
- **Plan**: Elite (acceso completo)

## Próximos Pasos

### 🔄 Mejoras Pendientes
1. **Integración de Pagos Reales**
   - Stripe API
   - PayPal API
   - Procesadores de crypto

2. **Base de Datos**
   - Migración a MySQL/PostgreSQL
   - Modelos de usuario
   - Historial de pagos

3. **Funcionalidades Avanzadas**
   - Verificación de email
   - Recuperación de contraseña
   - Perfil de usuario
   - Historial de suscripciones

4. **Testing**
   - Unit tests
   - Integration tests
   - E2E tests

## Contribución

Para contribuir al sistema de registro:

1. Fork el repositorio
2. Crear feature branch
3. Implementar cambios
4. Agregar tests
5. Crear pull request

## Soporte

Para soporte técnico:
- 📧 Email: support@aitraderx.com
- 📖 Documentación: `/docs`
- 🐛 Issues: GitHub Issues 