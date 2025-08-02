# 💼 IMPLEMENTACIÓN PORTFOLIO TRADING REAL
## Sistema de Gestión de Portfolios para Trading Real

---

## 📋 RESUMEN EJECUTIVO

### Objetivo
Implementar un sistema completo de gestión de portfolios para trading real, integrando múltiples brokers (MetaTrader 4/5, Binance, etc.) con monitoreo en tiempo real, análisis de riesgo y gestión automática de posiciones.

### Disponibilidad
- ✅ **Plan Premium** ($299/mes)
- ✅ **Plan Institutional** ($999/mes)
- ❌ **Plan Expert** ($99/mes) - No disponible
- ❌ **Plan Trader** ($29/mes) - No disponible
- ❌ **Plan Starter** (Gratis) - No disponible

---

## 🏗️ ARQUITECTURA DEL SISTEMA

### **ESTRUCTURA GENERAL:**
```
🧠 PORTFOLIO TRADING REAL
├── 🔗 Broker Connectors (MT4/MT5, Binance, APIs)
├── 📊 Real-time Data Feed
├── 💼 Position Management
├── 📈 Performance Analytics
├── 🛡️ Risk Management
└── 🔔 Real-time Alerts
```

### **FLUJO DE DATOS:**
```
1. Broker APIs → 2. Data Processing → 3. Portfolio Engine → 4. UI Components
```

---

## 🔗 INTEGRACIÓN CON BROKERS

### **1. MetaTrader 4/5 Integration**

#### **Configuración MT4/MT5:**
```python
# backend/src/services/brokers/mt4_connector.py
import MetaTrader5 as mt5
from typing import List, Dict, Optional

class MT4Connector:
    def __init__(self, account_id: str, password: str, server: str):
        self.account_id = account_id
        self.password = password
        self.server = server
        self.connected = False
    
    async def connect(self) -> bool:
        """Conectar a MetaTrader"""
        try:
            if not mt5.initialize():
                return False
            
            # Login a la cuenta
            authorized = mt5.login(
                login=int(self.account_id),
                password=self.password,
                server=self.server
            )
            
            self.connected = authorized
            return authorized
            
        except Exception as e:
            print(f"Error conectando a MT4/MT5: {e}")
            return False
    
    async def get_positions(self) -> List[Dict]:
        """Obtener posiciones abiertas"""
        if not self.connected:
            return []
        
        positions = mt5.positions_get()
        return [
            {
                'id': pos.ticket,
                'symbol': pos.symbol,
                'type': 'BUY' if pos.type == 0 else 'SELL',
                'volume': pos.volume,
                'open_price': pos.price_open,
                'current_price': pos.price_current,
                'profit': pos.profit,
                'swap': pos.swap,
                'open_time': pos.time,
                'broker': 'MT4/MT5'
            }
            for pos in positions
        ]
    
    async def get_balance(self) -> Dict:
        """Obtener balance de la cuenta"""
        if not self.connected:
            return {'balance': 0, 'equity': 0, 'margin': 0}
        
        account_info = mt5.account_info()
        return {
            'balance': account_info.balance,
            'equity': account_info.equity,
            'margin': account_info.margin,
            'free_margin': account_info.margin_free
        }
    
    async def close_position(self, ticket: int) -> bool:
        """Cerrar posición específica"""
        if not self.connected:
            return False
        
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        pos = position[0]
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": pos.volume,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": mt5.symbol_info_tick(pos.symbol).bid,
            "deviation": 20,
            "magic": 234000,
            "comment": "python script close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
```

### **2. Binance API Integration**

#### **Configuración Binance:**
```python
# backend/src/services/brokers/binance_connector.py
from binance.client import Client
from binance.exceptions import BinanceAPIException
from typing import List, Dict

class BinanceConnector:
    def __init__(self, api_key: str, api_secret: str):
        self.client = Client(api_key, api_secret)
        self.connected = True
    
    async def get_positions(self) -> List[Dict]:
        """Obtener posiciones de Binance"""
        try:
            # Obtener posiciones de futuros
            futures_positions = self.client.futures_position_information()
            
            positions = []
            for pos in futures_positions:
                if float(pos['positionAmt']) != 0:  # Solo posiciones abiertas
                    positions.append({
                        'id': pos['symbol'],
                        'symbol': pos['symbol'],
                        'type': 'BUY' if float(pos['positionAmt']) > 0 else 'SELL',
                        'volume': abs(float(pos['positionAmt'])),
                        'open_price': float(pos['entryPrice']),
                        'current_price': float(pos['markPrice']),
                        'profit': float(pos['unRealizedProfit']),
                        'broker': 'Binance'
                    })
            
            return positions
            
        except BinanceAPIException as e:
            print(f"Error Binance API: {e}")
            return []
    
    async def get_balance(self) -> Dict:
        """Obtener balance de Binance"""
        try:
            # Balance de spot
            spot_account = self.client.get_account()
            spot_balance = sum([
                float(asset['free']) + float(asset['locked'])
                for asset in spot_account['balances']
                if float(asset['free']) > 0 or float(asset['locked']) > 0
            ])
            
            # Balance de futuros
            futures_account = self.client.futures_account_balance()
            futures_balance = sum([
                float(balance['balance'])
                for balance in futures_account
                if float(balance['balance']) > 0
            ])
            
            return {
                'spot_balance': spot_balance,
                'futures_balance': futures_balance,
                'total_balance': spot_balance + futures_balance
            }
            
        except BinanceAPIException as e:
            print(f"Error obteniendo balance Binance: {e}")
            return {'spot_balance': 0, 'futures_balance': 0, 'total_balance': 0}
```

---

## 📊 MODELOS DE DATOS

### **1. Portfolio Models**

```python
# backend/src/models/portfolio_models.py
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime
from enum import Enum

class BrokerType(str, Enum):
    MT4 = "MT4"
    MT5 = "MT5"
    BINANCE = "Binance"
    OTHER = "Other"

class PositionType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

class RealPosition(BaseModel):
    id: str
    symbol: str
    type: PositionType
    volume: float
    open_price: float
    current_price: float
    profit_loss: float
    profit_loss_percent: float
    open_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    broker: BrokerType
    account_id: str
    swap: float = 0.0
    commission: float = 0.0

class PortfolioBalance(BaseModel):
    total_balance: float
    total_equity: float
    total_margin: float
    free_margin: float
    daily_pnl: float
    daily_pnl_percent: float
    by_broker: Dict[str, float]

class RiskMetrics(BaseModel):
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float

class PortfolioPerformance(BaseModel):
    total_return: float
    monthly_return: float
    weekly_return: float
    daily_return: float
    risk_metrics: RiskMetrics
    positions: List[RealPosition]
    balance: PortfolioBalance
```

---

## 🔧 APIS DEL BACKEND

### **1. Portfolio APIs**

```python
# backend/src/routers/portfolio.py
from fastapi import APIRouter, Depends, HTTPException
from typing import List
from ..models.portfolio_models import *
from ..services.portfolio_service import PortfolioService
from ..middleware.subscription_middleware import require_premium_plan

router = APIRouter(prefix="/api/portfolio", tags=["Portfolio"])

@router.get("/positions", response_model=List[RealPosition])
@require_premium_plan
async def get_real_positions(user_id: str = Depends(get_current_user)):
    """Obtiene posiciones reales de todos los brokers conectados"""
    try:
        portfolio_service = PortfolioService(user_id)
        positions = await portfolio_service.get_all_positions()
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/balance", response_model=PortfolioBalance)
@require_premium_plan
async def get_portfolio_balance(user_id: str = Depends(get_current_user)):
    """Obtiene balance total del portfolio real"""
    try:
        portfolio_service = PortfolioService(user_id)
        balance = await portfolio_service.get_total_balance()
        return balance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/performance", response_model=PortfolioPerformance)
@require_premium_plan
async def get_portfolio_performance(user_id: str = Depends(get_current_user)):
    """Obtiene métricas de rendimiento del portfolio"""
    try:
        portfolio_service = PortfolioService(user_id)
        performance = await portfolio_service.get_performance_metrics()
        return performance
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/positions/{position_id}/close")
@require_premium_plan
async def close_position(
    position_id: str, 
    user_id: str = Depends(get_current_user)
):
    """Cierra una posición específica"""
    try:
        portfolio_service = PortfolioService(user_id)
        success = await portfolio_service.close_position(position_id)
        
        if success:
            return {"message": "Position closed successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to close position")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/positions/{position_id}")
@require_premium_plan
async def modify_position(
    position_id: str,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    user_id: str = Depends(get_current_user)
):
    """Modifica stop loss y take profit de una posición"""
    try:
        portfolio_service = PortfolioService(user_id)
        success = await portfolio_service.modify_position(
            position_id, stop_loss, take_profit
        )
        
        if success:
            return {"message": "Position modified successfully"}
        else:
            raise HTTPException(status_code=400, detail="Failed to modify position")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/brokers/status")
@require_premium_plan
async def get_broker_status(user_id: str = Depends(get_current_user)):
    """Obtiene estado de conexión de todos los brokers"""
    try:
        portfolio_service = PortfolioService(user_id)
        status = await portfolio_service.get_broker_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### **2. Portfolio Service**

```python
# backend/src/services/portfolio_service.py
from typing import List, Dict, Optional
from ..models.portfolio_models import *
from ..services.brokers.mt4_connector import MT4Connector
from ..services.brokers.binance_connector import BinanceConnector
import asyncio

class PortfolioService:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.brokers = {}
        self._load_user_brokers()
    
    def _load_user_brokers(self):
        """Carga configuración de brokers del usuario"""
        # TODO: Cargar desde base de datos
        user_brokers = self._get_user_broker_config()
        
        for broker_config in user_brokers:
            if broker_config['type'] == 'MT4':
                self.brokers[broker_config['id']] = MT4Connector(
                    broker_config['account_id'],
                    broker_config['password'],
                    broker_config['server']
                )
            elif broker_config['type'] == 'Binance':
                self.brokers[broker_config['id']] = BinanceConnector(
                    broker_config['api_key'],
                    broker_config['api_secret']
                )
    
    async def get_all_positions(self) -> List[RealPosition]:
        """Obtiene posiciones de todos los brokers"""
        all_positions = []
        
        for broker_id, broker in self.brokers.items():
            try:
                positions = await broker.get_positions()
                for pos in positions:
                    pos['broker_id'] = broker_id
                    all_positions.append(RealPosition(**pos))
            except Exception as e:
                print(f"Error obteniendo posiciones de {broker_id}: {e}")
        
        return all_positions
    
    async def get_total_balance(self) -> PortfolioBalance:
        """Calcula balance total de todos los brokers"""
        total_balance = 0
        total_equity = 0
        total_margin = 0
        free_margin = 0
        by_broker = {}
        
        for broker_id, broker in self.brokers.items():
            try:
                balance = await broker.get_balance()
                by_broker[broker_id] = balance.get('balance', 0)
                total_balance += balance.get('balance', 0)
                total_equity += balance.get('equity', 0)
                total_margin += balance.get('margin', 0)
                free_margin += balance.get('free_margin', 0)
            except Exception as e:
                print(f"Error obteniendo balance de {broker_id}: {e}")
        
        # Calcular P&L diario
        daily_pnl = await self._calculate_daily_pnl()
        daily_pnl_percent = (daily_pnl / total_balance * 100) if total_balance > 0 else 0
        
        return PortfolioBalance(
            total_balance=total_balance,
            total_equity=total_equity,
            total_margin=total_margin,
            free_margin=free_margin,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            by_broker=by_broker
        )
    
    async def get_performance_metrics(self) -> PortfolioPerformance:
        """Calcula métricas de rendimiento del portfolio"""
        positions = await self.get_all_positions()
        balance = await self.get_total_balance()
        
        # Calcular métricas de riesgo
        risk_metrics = await self._calculate_risk_metrics(positions)
        
        # Calcular retornos
        returns = await self._calculate_returns()
        
        return PortfolioPerformance(
            total_return=returns['total'],
            monthly_return=returns['monthly'],
            weekly_return=returns['weekly'],
            daily_return=returns['daily'],
            risk_metrics=risk_metrics,
            positions=positions,
            balance=balance
        )
    
    async def close_position(self, position_id: str) -> bool:
        """Cierra una posición específica"""
        # Encontrar el broker correspondiente
        for broker_id, broker in self.brokers.items():
            try:
                if hasattr(broker, 'close_position'):
                    success = await broker.close_position(position_id)
                    if success:
                        return True
            except Exception as e:
                print(f"Error cerrando posición en {broker_id}: {e}")
        
        return False
```

---

## 🎨 COMPONENTES DEL FRONTEND

### **1. Portfolio Container**

```typescript
// frontend/src/components/Portfolio/Portfolio.tsx
import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useFeatureAccess } from '../../hooks/useFeatureAccess';
import PortfolioHeader from './PortfolioHeader';
import ActivePositions from './ActivePositions';
import PerformanceCharts from './PerformanceCharts';
import RiskMetrics from './RiskMetrics';
import BrokerConnections from './BrokerConnections';
import { RealPosition, PortfolioBalance, PortfolioPerformance } from '../../types/portfolio';

const Portfolio: React.FC = () => {
  const { user } = useAuth();
  const { hasAccess } = useFeatureAccess();
  const [positions, setPositions] = useState<RealPosition[]>([]);
  const [balance, setBalance] = useState<PortfolioBalance | null>(null);
  const [performance, setPerformance] = useState<PortfolioPerformance | null>(null);
  const [loading, setLoading] = useState(true);

  // Verificar acceso
  if (!hasAccess('portfolio_real_trading')) {
    return (
      <div className="p-6">
        <div className="trading-card p-6 text-center">
          <h2 className="text-2xl font-bold text-white mb-4">Portfolio Trading Real</h2>
          <p className="text-gray-400 mb-4">
            Esta funcionalidad está disponible solo para suscripciones Premium e Institutional
          </p>
          <button className="btn-primary">Upgrade Plan</button>
        </div>
      </div>
    );
  }

  useEffect(() => {
    loadPortfolioData();
    const interval = setInterval(loadPortfolioData, 30000); // Actualizar cada 30s
    return () => clearInterval(interval);
  }, []);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      
      // Cargar datos en paralelo
      const [positionsRes, balanceRes, performanceRes] = await Promise.all([
        fetch('/api/portfolio/positions'),
        fetch('/api/portfolio/balance'),
        fetch('/api/portfolio/performance')
      ]);

      if (positionsRes.ok) {
        const positionsData = await positionsRes.json();
        setPositions(positionsData);
      }

      if (balanceRes.ok) {
        const balanceData = await balanceRes.json();
        setBalance(balanceData);
      }

      if (performanceRes.ok) {
        const performanceData = await performanceRes.json();
        setPerformance(performanceData);
      }

    } catch (error) {
      console.error('Error loading portfolio data:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleClosePosition = async (positionId: string) => {
    try {
      const response = await fetch(`/api/portfolio/positions/${positionId}/close`, {
        method: 'POST'
      });

      if (response.ok) {
        // Recargar datos
        loadPortfolioData();
      }
    } catch (error) {
      console.error('Error closing position:', error);
    }
  };

  if (loading) {
    return (
      <div className="p-6">
        <div className="trading-card p-6 text-center">
          <div className="loading-spinner"></div>
          <p className="text-gray-400 mt-4">Cargando datos del portfolio...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header con balance total */}
      {balance && <PortfolioHeader balance={balance} />}
      
      {/* Posiciones activas */}
      <ActivePositions 
        positions={positions} 
        onClosePosition={handleClosePosition}
      />
      
      {/* Gráficos de rendimiento */}
      {performance && <PerformanceCharts performance={performance} />}
      
      {/* Métricas de riesgo */}
      {performance && <RiskMetrics metrics={performance.risk_metrics} />}
      
      {/* Configuración de brokers */}
      <BrokerConnections />
    </div>
  );
};

export default Portfolio;
```

### **2. Portfolio Header**

```typescript
// frontend/src/components/Portfolio/PortfolioHeader.tsx
import React from 'react';
import { PortfolioBalance } from '../../types/portfolio';

interface PortfolioHeaderProps {
  balance: PortfolioBalance;
}

const PortfolioHeader: React.FC<PortfolioHeaderProps> = ({ balance }) => {
  return (
    <div className="trading-card p-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {/* Balance Total */}
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-400 mb-2">Balance Total</h3>
          <div className="text-3xl font-bold text-white">
            ${balance.total_balance.toLocaleString()}
          </div>
          <div className="text-sm text-gray-500">Equity: ${balance.total_equity.toLocaleString()}</div>
        </div>

        {/* P&L Diario */}
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-400 mb-2">P&L Diario</h3>
          <div className={`text-3xl font-bold ${balance.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {balance.daily_pnl >= 0 ? '+' : ''}${balance.daily_pnl.toFixed(2)}
          </div>
          <div className={`text-sm ${balance.daily_pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {balance.daily_pnl_percent >= 0 ? '+' : ''}{balance.daily_pnl_percent.toFixed(2)}%
          </div>
        </div>

        {/* Margen Libre */}
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-400 mb-2">Margen Libre</h3>
          <div className="text-3xl font-bold text-blue-400">
            ${balance.free_margin.toLocaleString()}
          </div>
          <div className="text-sm text-gray-500">
            Utilizado: ${(balance.total_margin - balance.free_margin).toLocaleString()}
          </div>
        </div>

        {/* Distribución por Broker */}
        <div className="text-center">
          <h3 className="text-lg font-semibold text-gray-400 mb-2">Por Broker</h3>
          <div className="space-y-1">
            {Object.entries(balance.by_broker).map(([broker, amount]) => (
              <div key={broker} className="flex justify-between text-sm">
                <span className="text-gray-400">{broker}</span>
                <span className="text-white">${amount.toLocaleString()}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PortfolioHeader;
```

### **3. Active Positions**

```typescript
// frontend/src/components/Portfolio/ActivePositions.tsx
import React from 'react';
import { RealPosition } from '../../types/portfolio';

interface ActivePositionsProps {
  positions: RealPosition[];
  onClosePosition: (positionId: string) => void;
}

const ActivePositions: React.FC<ActivePositionsProps> = ({ positions, onClosePosition }) => {
  if (positions.length === 0) {
    return (
      <div className="trading-card p-6 text-center">
        <h3 className="text-xl font-semibold text-white mb-2">Posiciones Activas</h3>
        <p className="text-gray-400">No hay posiciones abiertas</p>
      </div>
    );
  }

  return (
    <div className="trading-card p-6">
      <h3 className="text-xl font-semibold text-white mb-4">Posiciones Activas ({positions.length})</h3>
      
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left p-2 text-gray-400">Símbolo</th>
              <th className="text-left p-2 text-gray-400">Tipo</th>
              <th className="text-left p-2 text-gray-400">Volumen</th>
              <th className="text-left p-2 text-gray-400">Precio Apertura</th>
              <th className="text-left p-2 text-gray-400">Precio Actual</th>
              <th className="text-left p-2 text-gray-400">P&L</th>
              <th className="text-left p-2 text-gray-400">P&L %</th>
              <th className="text-left p-2 text-gray-400">Broker</th>
              <th className="text-left p-2 text-gray-400">Acciones</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => (
              <tr key={position.id} className="border-b border-gray-800">
                <td className="p-2 text-white font-semibold">{position.symbol}</td>
                <td className="p-2">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    position.type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {position.type}
                  </span>
                </td>
                <td className="p-2 text-white">{position.volume}</td>
                <td className="p-2 text-white">${position.open_price.toFixed(5)}</td>
                <td className="p-2 text-white">${position.current_price.toFixed(5)}</td>
                <td className={`p-2 font-semibold ${
                  position.profit_loss >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {position.profit_loss >= 0 ? '+' : ''}${position.profit_loss.toFixed(2)}
                </td>
                <td className={`p-2 font-semibold ${
                  position.profit_loss_percent >= 0 ? 'text-green-400' : 'text-red-400'
                }`}>
                  {position.profit_loss_percent >= 0 ? '+' : ''}{position.profit_loss_percent.toFixed(2)}%
                </td>
                <td className="p-2 text-gray-400">{position.broker}</td>
                <td className="p-2">
                  <button
                    onClick={() => onClosePosition(position.id)}
                    className="px-3 py-1 bg-red-500 hover:bg-red-600 text-white rounded text-sm"
                  >
                    Cerrar
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default ActivePositions;
```

---

## 🛡️ SEGURIDAD Y AUTENTICACIÓN

### **1. Middleware de Verificación**

```python
# backend/src/middleware/subscription_middleware.py
from functools import wraps
from fastapi import HTTPException, Depends
from ..services.subscription_service import SubscriptionService

def require_premium_plan(func):
    """Decorator para verificar acceso a funcionalidades premium"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        user_id = kwargs.get('user_id')
        if not user_id:
            raise HTTPException(status_code=401, detail="Usuario no autenticado")
        
        subscription_service = SubscriptionService()
        user_plan = await subscription_service.get_user_plan(user_id)
        
        # Solo Premium e Institutional tienen acceso
        if user_plan not in ['premium', 'institutional']:
            raise HTTPException(
                status_code=403, 
                detail="Esta funcionalidad requiere plan Premium o Institutional"
            )
        
        return await func(*args, **kwargs)
    return wrapper
```

### **2. Encriptación de Credenciales**

```python
# backend/src/utils/encryption.py
from cryptography.fernet import Fernet
import base64
import os

class CredentialEncryption:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY', Fernet.generate_key())
        self.cipher_suite = Fernet(self.key)
    
    def encrypt(self, data: str) -> str:
        """Encripta credenciales de brokers"""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt(self, encrypted_data: str) -> str:
        """Desencripta credenciales de brokers"""
        return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
```

---

## 📊 BASE DE DATOS

### **1. Tablas de Portfolio**

```sql
-- Tabla de configuración de brokers por usuario
CREATE TABLE user_broker_configs (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    broker_type ENUM('MT4', 'MT5', 'Binance', 'Other') NOT NULL,
    broker_name VARCHAR(100) NOT NULL,
    account_id VARCHAR(100),
    encrypted_password TEXT,
    encrypted_api_key TEXT,
    encrypted_api_secret TEXT,
    server VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Tabla de posiciones históricas
CREATE TABLE portfolio_positions (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    broker_position_id VARCHAR(100) NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    position_type ENUM('BUY', 'SELL') NOT NULL,
    volume DECIMAL(15,5) NOT NULL,
    open_price DECIMAL(15,5) NOT NULL,
    close_price DECIMAL(15,5),
    profit_loss DECIMAL(15,2),
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP,
    broker VARCHAR(50) NOT NULL,
    status ENUM('OPEN', 'CLOSED') DEFAULT 'OPEN',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- Tabla de métricas de rendimiento
CREATE TABLE portfolio_performance (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    date DATE NOT NULL,
    total_balance DECIMAL(15,2) NOT NULL,
    total_equity DECIMAL(15,2) NOT NULL,
    daily_pnl DECIMAL(15,2) NOT NULL,
    sharpe_ratio DECIMAL(10,4),
    max_drawdown DECIMAL(10,4),
    win_rate DECIMAL(5,2),
    profit_factor DECIMAL(10,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id),
    UNIQUE KEY unique_user_date (user_id, date)
);
```

---

## 🚀 PLAN DE IMPLEMENTACIÓN

### **Fase 1: Infraestructura Base (Semana 1-2)**
- [ ] Crear modelos de datos
- [ ] Implementar encriptación de credenciales
- [ ] Configurar base de datos
- [ ] Crear middleware de verificación

### **Fase 2: Integración de Brokers (Semana 3-4)**
- [ ] Implementar MT4/MT5 connector
- [ ] Implementar Binance connector
- [ ] Crear sistema de gestión de conexiones
- [ ] Implementar APIs del backend

### **Fase 3: Frontend (Semana 5-6)**
- [ ] Crear componentes del portfolio
- [ ] Implementar actualización en tiempo real
- [ ] Crear gráficos de rendimiento
- [ ] Implementar gestión de posiciones

### **Fase 4: Testing y Optimización (Semana 7-8)**
- [ ] Testing de integración con brokers
- [ ] Optimización de rendimiento
- [ ] Testing de seguridad
- [ ] Documentación final

---

## 📈 MÉTRICAS DE ÉXITO

### **Técnicas:**
- **Tiempo de respuesta**: < 2 segundos para APIs
- **Uptime**: > 99.5%
- **Precisión de datos**: 100% sincronización con brokers
- **Seguridad**: 0 vulnerabilidades críticas

### **Negocio:**
- **Adopción**: 80% de usuarios Premium/Institutional
- **Retención**: 90% después de 3 meses
- **Satisfacción**: > 4.5/5 en encuestas de usuarios

---

## 🔒 CONSIDERACIONES DE SEGURIDAD

1. **Encriptación**: Todas las credenciales encriptadas
2. **Auditoría**: Logs completos de todas las operaciones
3. **Validación**: Verificación de permisos en cada operación
4. **Rate Limiting**: Protección contra abuso de APIs
5. **Backup**: Respaldo automático de configuraciones

---

## 📞 SOPORTE

### **Niveles de Soporte:**
- **Premium**: Soporte por email (24h)
- **Institutional**: Soporte telefónico 24/7 + gestor dedicado

### **Documentación:**
- Guías de configuración por broker
- Videos tutoriales
- FAQ completo
- Soporte técnico especializado

---

*Este documento define la implementación completa del sistema de Portfolio para trading real, disponible exclusivamente para usuarios Premium e Institutional.* 