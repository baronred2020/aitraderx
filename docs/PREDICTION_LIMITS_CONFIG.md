# Configuración de Límites de Predicciones

## 📊 Límites por Plan de Suscripción

### **🆓 STARTER (Gratuito)**
- **Predicciones por día**: 5
- **Pares de trading**: 1 (EURUSD)
- **Timeframes**: 1
- **Backtests por mes**: 5
- **Soporte**: Comunitario

### **📈 TRADER ($29/mes)**
- **Predicciones por día**: 20
- **Pares de trading**: 5
- **Timeframes**: 2
- **Backtests por mes**: 20
- **Soporte**: Email

### **🚀 EXPERT ($99/mes)**
- **Predicciones por día**: 50
- **Pares de trading**: 50
- **Timeframes**: 5
- **Backtests por mes**: 100
- **Soporte**: Email

### **👑 PREMIUM ($299/mes)**
- **Predicciones por día**: 100
- **Pares de trading**: 1000
- **Timeframes**: 10
- **Backtests por mes**: 500
- **Soporte**: Teléfono

### **🏢 INSTITUTIONAL (Personalizado)**
- **Predicciones por día**: **ILIMITADO** (-1)
- **Pares de trading**: 5000
- **Timeframes**: 15
- **Backtests por mes**: 2000
- **Soporte**: Dedicado

### **⚡ ADMIN**
- **Predicciones por día**: **ILIMITADO** (-1)
- **Acceso completo**: Todas las funcionalidades
- **Sin restricciones**: Límites deshabilitados

## 🔧 Implementación Técnica

### **Frontend (React)**
```typescript
// En BrainTrader.tsx
const getPlanLimitations = () => {
  switch (subscription.planType) {
    case 'starter':
      return { maxPredictionsPerDay: 5, ... };
    case 'trader':
      return { maxPredictionsPerDay: 20, ... };
    case 'expert':
      return { maxPredictionsPerDay: 50, ... };
    case 'premium':
      return { maxPredictionsPerDay: 100, ... };
    case 'institutional':
      return { maxPredictionsPerDay: -1, ... }; // Ilimitado
  }
};
```

### **Backend (Python)**
```python
# En prediction_service.py
def _get_user_plan_limits(self, user_id: int, plan_type: str) -> Dict:
    plan_limits = {
        'starter': 5,
        'trader': 20,
        'expert': 50,
        'premium': 100,
        'institutional': -1,  # Sin límite
        'admin': -1  # Sin límite
    }
    return {
        'max_predictions_per_day': plan_limits.get(plan_type, 5),
        'has_unlimited': plan_limits.get(plan_type, 5) == -1
    }
```

### **Base de Datos**
```sql
-- Tabla user_prediction_limits
CREATE TABLE user_prediction_limits (
    user_id INT PRIMARY KEY,
    plan_type VARCHAR(20) NOT NULL,
    max_predictions_per_day INT NOT NULL,
    predictions_used_today INT DEFAULT 0,
    last_reset_date DATE DEFAULT CURRENT_DATE
);

-- Valores por defecto
INSERT INTO user_prediction_limits VALUES 
(1, 'starter', 5),
(2, 'trader', 20),
(3, 'expert', 50),
(4, 'premium', 100),
(5, 'institutional', -1); -- Ilimitado
```

## 🎯 Lógica de Verificación

### **Límites Ilimitados**
- **Valor**: -1
- **Verificación**: `has_unlimited = True`
- **Comportamiento**: Siempre puede generar predicciones

### **Límites Finitos**
- **Verificación**: `remaining_predictions > 0`
- **Reset**: Diario automático
- **Contador**: Se incrementa con cada predicción

### **Middleware de Verificación**
```python
# En subscription_middleware.py
if feature == "predictions":
    usage = self._get_today_usage(user_id)
    if usage.predictions_made_today >= plan.max_predictions_per_day:
        return False, f"Límite diario alcanzado ({plan.max_predictions_per_day})"
```

## 📈 Métricas y Monitoreo

### **Contadores Diarios**
- `predictions_used_today`: Predicciones usadas hoy
- `last_reset_date`: Fecha del último reset
- `max_predictions_per_day`: Límite del plan

### **Estadísticas por Usuario**
- Total de predicciones
- Tasa de éxito
- Mejor par de trading
- Predicciones del día actual

## 🔄 Reset Automático

El sistema incluye un **reset automático diario**:

```python
def can_generate_prediction(self):
    today = datetime.utcnow().date()
    if self.last_reset_date != today:
        self.predictions_used_today = 0
        self.last_reset_date = today
    
    return self.predictions_used_today < self.max_predictions_per_day
```

## 🚀 Próximos Pasos

1. **Implementar verificación real** de predicciones usadas
2. **Conectar con base de datos** para límites dinámicos
3. **Agregar métricas** de uso por usuario
4. **Implementar notificaciones** cuando se alcancen límites
5. **Crear dashboard** de administración de límites

---

**Última actualización**: Enero 2025
**Versión**: 1.0
**Estado**: Implementado