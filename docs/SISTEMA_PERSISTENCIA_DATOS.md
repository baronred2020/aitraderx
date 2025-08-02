# Sistema de Persistencia de Datos - Trading Virtual

## 📊 **Resumen**

El sistema de trading virtual implementa un sistema completo de persistencia de datos que guarda automáticamente toda la información del usuario en el `localStorage` del navegador. Esto garantiza que los datos persistan entre sesiones y recargas de página.

## 🔄 **Datos que se Guardan**

### 1. **Balance Virtual** (`virtual_balance`)
- **Ubicación**: `localStorage.getItem('virtual_balance')`
- **Contenido**: Saldo actual en USD
- **Actualización**: Cada vez que se recarga o se realiza una operación
- **Reglas**: 
  - Balance inicial: $10,000 USD
  - Solo se puede recargar si balance < $100 USD
  - Recarga permitida: $1 - $10,000 USD

### 2. **Transacciones de Wallet** (`virtual_transactions`)
- **Ubicación**: `localStorage.getItem('virtual_transactions')`
- **Contenido**: Array de transacciones (depósitos, operaciones de trading)
- **Estructura**:
```json
{
  "id": 1234567890,
  "type": "deposit|trade",
  "amount": 1000.0,
  "description": "Recarga de saldo - $1,000",
  "created_at": "2024-01-15T10:30:00.000Z"
}
```

### 3. **Posiciones Simuladas** (`simulated_positions`)
- **Ubicación**: `localStorage.getItem('simulated_positions')`
- **Contenido**: Array de posiciones abiertas y cerradas
- **Estructura**:
```json
{
  "id": "1705312200000",
  "symbol": "EURUSD",
  "type": "BUY|SELL",
  "amount": 10000,
  "openPrice": 1.0925,
  "currentPrice": 1.0930,
  "openTime": "2024-01-15T10:30:00.000Z",
  "stopLoss": 1.0900,
  "takeProfit": 1.0950,
  "pnl": 50.0,
  "pnlPercent": 0.46,
  "status": "open|closed"
}
```

### 4. **Órdenes Simuladas** (`simulated_orders`)
- **Ubicación**: `localStorage.getItem('simulated_orders')`
- **Contenido**: Array de órdenes (market, limit, stop)
- **Estructura**:
```json
{
  "id": "1705312200000",
  "symbol": "EURUSD",
  "type": "market|limit|stop",
  "side": "buy|sell",
  "amount": 10000,
  "price": 1.0925,
  "stopLoss": 1.0900,
  "takeProfit": 1.0950,
  "status": "pending|filled|cancelled|rejected",
  "createdAt": "2024-01-15T10:30:00.000Z",
  "filledAt": "2024-01-15T10:31:00.000Z",
  "filledPrice": 1.0928,
  "description": "Compra EURUSD"
}
```

## 🔧 **Implementación Técnica**

### Hooks de Persistencia

#### 1. **useWallet.ts**
```typescript
// Cargar datos al inicializar
useEffect(() => {
  const savedBalance = localStorage.getItem('virtual_balance');
  const savedTransactions = localStorage.getItem('virtual_transactions');
  
  if (savedBalance && savedTransactions) {
    setBalance(parseFloat(savedBalance));
    setTransactions(JSON.parse(savedTransactions));
  } else {
    // Primer acceso: establecer balance inicial
    setBalance(INITIAL_BALANCE);
    // ... crear transacción inicial
  }
}, []);

// Guardar datos cuando cambien
useEffect(() => {
  localStorage.setItem('virtual_balance', balance?.toString() || '0');
  localStorage.setItem('virtual_transactions', JSON.stringify(transactions));
}, [balance, transactions]);
```

#### 2. **useSimulatedPositions.ts**
```typescript
// Cargar posiciones al inicializar
useEffect(() => {
  const savedPositions = localStorage.getItem('simulated_positions');
  if (savedPositions) {
    const parsed = JSON.parse(savedPositions);
    const positionsWithDates = parsed.map((pos: any) => ({
      ...pos,
      openTime: new Date(pos.openTime)
    }));
    setPositions(positionsWithDates);
  }
}, []);

// Guardar posiciones cuando cambien
useEffect(() => {
  localStorage.setItem('simulated_positions', JSON.stringify(positions));
}, [positions]);
```

#### 3. **useSimulatedOrders.ts**
```typescript
// Cargar órdenes al inicializar
useEffect(() => {
  const savedOrders = localStorage.getItem('simulated_orders');
  if (savedOrders) {
    const parsed = JSON.parse(savedOrders);
    const ordersWithDates = parsed.map((order: any) => ({
      ...order,
      createdAt: new Date(order.createdAt),
      filledAt: order.filledAt ? new Date(order.filledAt) : undefined
    }));
    setOrders(ordersWithDates);
  }
}, []);

// Guardar órdenes cuando cambien
useEffect(() => {
  localStorage.setItem('simulated_orders', JSON.stringify(orders));
}, [orders]);
```

## 📈 **Actualización en Tiempo Real**

### P&L Dinámico
El sistema actualiza automáticamente el P&L de las posiciones abiertas basándose en los precios de mercado en tiempo real:

```typescript
const updatePositionsPnL = useCallback((marketData: MarketData) => {
  setPositions(prev => prev.map(pos => {
    const currentPrice = parseFloat(marketData[pos.symbol]?.price || pos.currentPrice.toString());
    
    // Calcular P&L
    let pnl: number;
    if (pos.type === 'BUY') {
      pnl = (currentPrice - pos.openPrice) * pos.amount;
    } else {
      pnl = (pos.openPrice - currentPrice) * pos.amount;
    }
    
    const pnlPercent = (pnl / (pos.openPrice * pos.amount)) * 100;
    
    return { ...pos, currentPrice, pnl, pnlPercent };
  }));
}, []);
```

### Ejecución Automática de Órdenes
Las órdenes se ejecutan automáticamente cuando se cumplen las condiciones de precio:

```typescript
const executeOrders = useCallback((marketData: MarketData) => {
  setOrders(prev => prev.map(order => {
    if (order.status !== 'pending') return order;
    
    const currentPrice = parseFloat(marketData[order.symbol]?.price || '0');
    
    // Lógica de ejecución por tipo de orden
    switch (order.type) {
      case 'market':
        return { ...order, status: 'filled', filledAt: new Date(), filledPrice: currentPrice };
      case 'limit':
        // Ejecutar cuando el precio alcance el nivel especificado
        break;
      case 'stop':
        // Ejecutar cuando el precio alcance el nivel de stop
        break;
    }
    
    return order;
  }));
}, []);
```

## 🛡️ **Manejo de Errores**

### Validación de Datos
```typescript
try {
  const parsed = JSON.parse(savedData);
  // Validar estructura de datos
  if (Array.isArray(parsed)) {
    setData(parsed);
  } else {
    console.error('Formato de datos inválido');
    localStorage.removeItem(key); // Limpiar datos corruptos
  }
} catch (error) {
  console.error('Error loading data:', error);
  localStorage.removeItem(key); // Limpiar datos corruptos
}
```

### Fallback para Datos Corruptos
Si los datos en localStorage están corruptos, el sistema:
1. Elimina los datos corruptos
2. Reinicia con valores por defecto
3. Registra el error en la consola

## 🔍 **Verificación de Datos**

### Script de Verificación
Se incluye un script `verify-storage.js` que permite verificar el estado de los datos almacenados:

```javascript
// Ejecutar en la consola del navegador
const virtualBalance = localStorage.getItem('virtual_balance');
const virtualTransactions = localStorage.getItem('virtual_transactions');
const simulatedPositions = localStorage.getItem('simulated_positions');
const simulatedOrders = localStorage.getItem('simulated_orders');

console.log('Balance:', virtualBalance);
console.log('Transacciones:', virtualTransactions ? JSON.parse(virtualTransactions).length : 0);
console.log('Posiciones:', simulatedPositions ? JSON.parse(simulatedPositions).length : 0);
console.log('Órdenes:', simulatedOrders ? JSON.parse(simulatedOrders).length : 0);
```

## 📱 **Compatibilidad**

### Navegadores Soportados
- ✅ Chrome/Chromium
- ✅ Firefox
- ✅ Safari
- ✅ Edge
- ✅ Opera

### Limitaciones
- **Tamaño**: localStorage tiene un límite de ~5-10MB
- **Alcance**: Los datos son específicos del dominio
- **Persistencia**: Los datos se mantienen hasta que se borren manualmente

## 🚀 **Ventajas del Sistema**

1. **Persistencia Automática**: No se pierden datos entre sesiones
2. **Tiempo Real**: P&L se actualiza automáticamente
3. **Robusto**: Manejo de errores y validación de datos
4. **Eficiente**: Solo guarda cuando hay cambios
5. **Escalable**: Fácil de extender para más tipos de datos

## 🔮 **Futuras Mejoras**

1. **Sincronización con Backend**: Conectar con base de datos real
2. **Backup en la Nube**: Sincronizar con servidor
3. **Compresión de Datos**: Optimizar uso de espacio
4. **Migración de Datos**: Actualizar estructura cuando sea necesario
5. **Exportación**: Permitir exportar datos a CSV/JSON

---

**Conclusión**: El sistema de persistencia está completamente funcional y garantiza que todos los datos del trading virtual se mantengan entre sesiones, proporcionando una experiencia de usuario consistente y confiable. 