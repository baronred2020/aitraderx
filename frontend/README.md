# AITRADERX - Plataforma de Trading con IA

Una plataforma de trading moderna y profesional impulsada por inteligencia artificial, diseñada para ofrecer la mejor experiencia de usuario en el mercado de divisas.

## 🚀 Características Principales

### Diseño Moderno y Profesional
- **Tema Oscuro**: Interfaz elegante con tema oscuro inspirado en las mejores plataformas de trading
- **Responsive Design**: Optimizado para desktop, tablet y móvil
- **Animaciones Suaves**: Transiciones y efectos visuales profesionales
- **Glass Morphism**: Efectos de cristal y transparencias modernos

### Componentes de Trading Profesionales
- **Gráficos Avanzados**: Candlestick, líneas y áreas con indicadores técnicos
- **Panel de Órdenes**: Interfaz intuitiva para colocar órdenes de mercado, límite y stop
- **Posiciones en Vivo**: Seguimiento en tiempo real de posiciones abiertas
- **Historial de Órdenes**: Registro completo de todas las operaciones

### Sistema de IA Integrado
- **Monitor de IA**: Seguimiento en tiempo real del rendimiento de los modelos
- **Reinforcement Learning**: Dashboard especializado para agentes de RL
- **Análisis Avanzado**: Herramientas de análisis técnico y fundamental
- **Alertas Inteligentes**: Sistema de notificaciones personalizadas

## 🎨 Diseño y UX

### Paleta de Colores
```css
--primary-bg: #0a0e1a      /* Fondo principal oscuro */
--secondary-bg: #1a1f2e    /* Fondo secundario */
--tertiary-bg: #252b3d     /* Fondo terciario */
--accent-text: #38b2ac     /* Color de acento */
--success-color: #48bb78   /* Verde para ganancias */
--danger-color: #f56565    /* Rojo para pérdidas */
--warning-color: #ed8936   /* Naranja para advertencias */
```

### Tipografía
- **Fuente Principal**: Inter (sistema de fuentes sans-serif)
- **Jerarquía Clara**: Títulos, subtítulos y texto con pesos apropiados
- **Legibilidad Optimizada**: Contraste y espaciado profesional

### Componentes Reutilizables
- **Trading Cards**: Tarjetas con efectos hover y animaciones
- **Status Indicators**: Indicadores de estado con colores semánticos
- **Notification System**: Sistema completo de notificaciones
- **Modal System**: Ventanas modales con backdrop blur

## 📱 Responsive Design

### Breakpoints
- **Mobile**: < 768px
- **Tablet**: 768px - 1024px
- **Desktop**: > 1024px

### Características Móviles
- **Sidebar Colapsible**: Navegación optimizada para móvil
- **Touch Targets**: Botones de tamaño apropiado (44px mínimo)
- **Gestos**: Soporte para gestos táctiles
- **Performance**: Optimizado para dispositivos móviles

## 🔧 Tecnologías Utilizadas

### Frontend
- **React 18**: Framework principal
- **TypeScript**: Tipado estático
- **Tailwind CSS**: Framework de utilidades CSS
- **Lucide React**: Iconografía moderna
- **Recharts**: Gráficos profesionales
- **React Router**: Navegación SPA

### Características Técnicas
- **Componentes Funcionales**: Arquitectura moderna con hooks
- **Estado Global**: Gestión de estado con React Context
- **Animaciones CSS**: Transiciones suaves y efectos
- **Optimización**: Lazy loading y code splitting

## 🏗️ Arquitectura de Componentes

### Layout Principal
```
Layout/
├── Header (con stats en tiempo real)
├── Sidebar (navegación principal)
└── Main Content (área de contenido)
```

### Componentes Especializados
```
components/
├── Common/
│   ├── Layout.tsx
│   ├── Notification.tsx
│   └── LoadingSpinner.tsx
├── Dashboard/
│   └── Dashboard.tsx
├── Trading/
│   ├── TradingView.tsx
│   └── TradingChart.tsx
├── Portfolio/
├── Analysis/
├── AIMonitor/
├── RL/
└── Alerts/
```

## 🎯 Funcionalidades por Sección

### Dashboard
- **Métricas en Tiempo Real**: Balance, P&L, precisión IA
- **Gráficos de Rendimiento**: Evolución del portfolio
- **Estado del Sistema**: Monitoreo de servicios IA
- **Pares Activos**: Lista de instrumentos disponibles

### Trading
- **Gráficos Profesionales**: Candlestick, líneas, áreas
- **Panel de Órdenes**: Compra/venta con tipos múltiples
- **Posiciones Abiertas**: Seguimiento en vivo
- **Historial de Órdenes**: Registro completo

### Portfolio
- **Análisis de Posiciones**: P&L por instrumento
- **Distribución de Activos**: Gráficos de asignación
- **Rendimiento Histórico**: Evolución temporal
- **Métricas de Riesgo**: Sharpe ratio, drawdown

### IA Monitor
- **Estado de Modelos**: Precisión y rendimiento
- **Entrenamiento en Vivo**: Progreso de RL
- **Detección de Drift**: Monitoreo de cambios
- **Alertas de Sistema**: Notificaciones automáticas

## 🚀 Instalación y Desarrollo

### Prerrequisitos
- Node.js 16+
- npm o yarn

### Instalación
```bash
# Clonar el repositorio
git clone <repository-url>
cd aitraderx/frontend

# Instalar dependencias
npm install

# Iniciar servidor de desarrollo
npm start
```

### Scripts Disponibles
```bash
npm start          # Servidor de desarrollo
npm run build      # Build de producción
npm run test       # Ejecutar tests
npm run lint       # Linting del código
npm run format     # Formateo automático
```

### Estructura de Archivos
```
frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/     # Componentes React
│   ├── contexts/       # Contextos de estado
│   ├── hooks/          # Custom hooks
│   ├── services/       # Servicios API
│   ├── styles/         # Estilos adicionales
│   ├── types/          # Definiciones TypeScript
│   ├── utils/          # Utilidades
│   ├── App.tsx         # Componente principal
│   └── index.tsx       # Punto de entrada
├── package.json
├── tailwind.config.js
└── tsconfig.json
```

## 🎨 Personalización

### Variables CSS
El tema se puede personalizar modificando las variables CSS en `src/index.css`:

```css
:root {
  --primary-bg: #0a0e1a;
  --secondary-bg: #1a1f2e;
  --accent-text: #38b2ac;
  /* ... más variables */
}
```

### Componentes
Cada componente está diseñado para ser reutilizable y personalizable:

```tsx
// Ejemplo de uso de TradingCard
<div className="trading-card p-6">
  <h3 className="text-lg font-semibold text-white">Título</h3>
  <p className="text-gray-400">Contenido</p>
</div>
```

## 📊 Performance

### Optimizaciones Implementadas
- **Code Splitting**: Carga diferida de componentes
- **Memoización**: React.memo para componentes pesados
- **Lazy Loading**: Carga bajo demanda
- **Optimización de Gráficos**: Recharts optimizado
- **CSS Optimizado**: Tailwind purged

### Métricas Objetivo
- **First Contentful Paint**: < 1.5s
- **Largest Contentful Paint**: < 2.5s
- **Cumulative Layout Shift**: < 0.1
- **First Input Delay**: < 100ms

## 🔒 Seguridad

### Mejores Prácticas
- **Input Sanitization**: Validación de entradas
- **XSS Prevention**: Escape de contenido dinámico
- **CSRF Protection**: Tokens de seguridad
- **HTTPS Only**: Conexiones seguras

## 🌐 Compatibilidad

### Navegadores Soportados
- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

### Dispositivos
- **Desktop**: 1920x1080 y superiores
- **Tablet**: 768x1024
- **Mobile**: 375x667 y superiores

## 🤝 Contribución

### Guías de Desarrollo
1. **Fork** el repositorio
2. **Crea** una rama para tu feature
3. **Commit** tus cambios
4. **Push** a la rama
5. **Abre** un Pull Request

### Estándares de Código
- **TypeScript**: Tipado estricto
- **ESLint**: Reglas de linting
- **Prettier**: Formateo automático
- **Conventional Commits**: Mensajes de commit

## 📈 Roadmap

### Próximas Características
- [ ] **WebSocket Integration**: Datos en tiempo real
- [ ] **Advanced Charts**: Más tipos de gráficos
- [ ] **Mobile App**: Aplicación nativa
- [ ] **Social Trading**: Funcionalidades sociales
- [ ] **AI Insights**: Análisis predictivo avanzado

### Mejoras de UX
- [ ] **Keyboard Shortcuts**: Atajos de teclado
- [ ] **Custom Themes**: Temas personalizables
- [ ] **Accessibility**: Mejoras de accesibilidad
- [ ] **Performance**: Optimizaciones adicionales

## 📞 Soporte

### Contacto
- **Email**: support@aitraderx.com
- **Documentación**: docs.aitraderx.com
- **Issues**: GitHub Issues

### Recursos
- **API Documentation**: api.aitraderx.com
- **Design System**: design.aitraderx.com
- **Component Library**: components.aitraderx.com

---

**AITRADERX** - Plataforma de Trading con IA del Futuro 🚀 