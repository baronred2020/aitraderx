# 🔧 Debug de Yahoo Finance - TradingChart

## 🚨 **Problemas Identificados y Soluciones**

### **1. Cache demasiado agresivo**
**Problema**: Los hooks tenían cache de 2-5 minutos, impidiendo ver cambios inmediatos.

**Solución implementada**:
- ✅ **Market Data**: Cache reducido de 2 minutos → 30 segundos
- ✅ **Candles**: Cache reducido de 5 minutos → 1 minuto
- ✅ **Backend**: Cache reducido de 60 segundos → 30 segundos (precios) y 5 minutos → 1 minuto (velas)

### **2. Falta de limpieza de cache**
**Problema**: Al cambiar símbolo, el cache anterior interfería.

**Solución implementada**:
- ✅ **Limpieza automática**: Los hooks ahora limpian cache anterior al cambiar símbolos
- ✅ **Logs de debug**: Se agregaron logs para rastrear limpieza de cache

### **3. Intervalos muy largos**
**Problema**: Actualizaciones cada 2-5 minutos eran demasiado lentas.

**Solución implementada**:
- ✅ **Market Data**: Actualización cada 30 segundos (antes: 2 minutos)
- ✅ **Candles**: Actualización cada 1 minuto (antes: 5 minutos)

### **4. Falta de logs de debug**
**Problema**: No había forma de ver qué estaba pasando.

**Solución implementada**:
- ✅ **Frontend logs**: Logs en hooks para rastrear fetch y cache
- ✅ **Backend logs**: Logs en endpoints y funciones de fetch
- ✅ **Componente de prueba**: `YahooTestChart` para testing

---

## 🧪 **Cómo Probar**

### **1. Usar el componente de prueba**
```typescript
import YahooTestChart from './components/Trading/YahooTestChart';

// En tu componente principal
<YahooTestChart />
```

### **2. Verificar logs en consola**
Abre las herramientas de desarrollador (F12) y busca estos logs:

**Frontend**:
```
[Yahoo Market Data] Fetching fresh data for AAPL
[Yahoo Market Data] Received data: {AAPL: {...}}
[Yahoo Candles] Fetching fresh data for AAPL
[Yahoo Candles] Received data for AAPL: {...}
```

**Backend** (terminal donde corre el servidor):
```
[Yahoo Endpoint] Request for symbols: ['AAPL']
[Yahoo Backend] Fetching price for symbol: AAPL
[Yahoo Backend] Mapped to Yahoo symbol: AAPL
[Yahoo Backend] Successfully fetched price for AAPL: {...}
```

### **3. Probar cambio de símbolos**
1. Abre el componente de prueba
2. Cambia el símbolo en el selector
3. Verifica que los datos se actualicen
4. Revisa los logs para confirmar que se está haciendo fetch nuevo

### **4. Probar diferentes timeframes**
1. Cambia el timeframe en el selector
2. Verifica que las velas se actualicen
3. Confirma que el número de velas cambie según el timeframe

---

## 🔍 **Debugging Step by Step**

### **Paso 1: Verificar que el backend está corriendo**
```bash
# En el directorio backend
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

### **Paso 2: Probar endpoints directamente**
```bash
# Probar endpoint de precios
curl "http://localhost:8000/api/market-data-yahoo?symbols=AAPL"

# Probar endpoint de velas
curl "http://localhost:8000/api/candles-yahoo?symbol=AAPL&interval=15&count=50"
```

### **Paso 3: Verificar logs del backend**
Los logs deberían mostrar:
```
[Yahoo Endpoint] Request for symbols: ['AAPL']
[Yahoo Backend] Fetching price for symbol: AAPL
[Yahoo Backend] Mapped to Yahoo symbol: AAPL
[Yahoo Backend] Successfully fetched price for AAPL: {...}
```

### **Paso 4: Verificar logs del frontend**
En la consola del navegador deberías ver:
```
[Yahoo Market Data] Fetching fresh data for AAPL
[Yahoo Market Data] Received data: {AAPL: {...}}
```

### **Paso 5: Probar cambio de símbolos**
1. Cambia el símbolo en el componente de prueba
2. Verifica que aparezcan estos logs:
```
[Yahoo Market Data] Clearing old cache for symbol change
[Yahoo Market Data] Fetching fresh data for MSFT
```

---

## 🐛 **Problemas Comunes y Soluciones**

### **Problema 1: "No hay datos disponibles"**
**Causa**: Error en la API de Yahoo Finance o símbolo no soportado.

**Solución**:
1. Verificar que el símbolo esté en `YAHOO_SYMBOL_MAP`
2. Revisar logs del backend para errores
3. Probar con símbolos conocidos como 'AAPL', 'MSFT'

### **Problema 2: "Cargando..." indefinidamente**
**Causa**: Error en la red o backend no disponible.

**Solución**:
1. Verificar que el backend esté corriendo en puerto 8000
2. Revisar la consola del navegador para errores de red
3. Verificar que no haya CORS issues

### **Problema 3: Datos no se actualizan al cambiar símbolo**
**Causa**: Cache no se está limpiando correctamente.

**Solución**:
1. Verificar logs de limpieza de cache
2. Forzar refresh manual (F5)
3. Verificar que los hooks se estén re-ejecutando

### **Problema 4: Errores de CORS**
**Causa**: El backend no permite requests del frontend.

**Solución**:
1. Verificar que CORS esté configurado en el backend
2. Asegurar que el frontend esté en el puerto correcto
3. Verificar que las URLs sean correctas

---

## 📊 **Métricas de Rendimiento Esperadas**

### **Tiempos de respuesta**:
- **Market Data**: 1-3 segundos
- **Candles**: 2-5 segundos
- **Cache hit**: < 100ms

### **Frecuencia de actualización**:
- **Market Data**: Cada 30 segundos
- **Candles**: Cada 1 minuto
- **Cache TTL**: 30 segundos (precios) / 1 minuto (velas)

### **Logs esperados por minuto**:
- **Market Data**: 2 logs por símbolo
- **Candles**: 1 log por símbolo
- **Cache hits**: 1-2 logs por símbolo

---

## 🎯 **Próximos Pasos**

### **Si todo funciona**:
1. ✅ Integrar `YahooTradingChart` en el dashboard principal
2. ✅ Reemplazar `TradingChart` con la versión de Yahoo Finance
3. ✅ Implementar selector de fuente de datos
4. ✅ Optimizar UI/UX

### **Si hay problemas**:
1. 🔍 Revisar logs detalladamente
2. 🔍 Probar con diferentes símbolos
3. 🔍 Verificar conectividad de red
4. 🔍 Comprobar que Yahoo Finance esté disponible

---

## 📞 **Soporte**

Si encuentras problemas:

1. **Revisa los logs** en consola del navegador y terminal del backend
2. **Prueba el componente de test** para aislar el problema
3. **Verifica la conectividad** con Yahoo Finance
4. **Comprueba que el backend esté corriendo** en el puerto correcto

Los logs agregados te darán información detallada sobre qué está pasando en cada paso del proceso. 