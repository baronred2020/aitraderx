# 🤖 Modelos de IA y Estrategias de Trading - AI TraderX

## 📋 Índice
1. [Modelos de IA Disponibles](#modelos-de-ia-disponibles)
2. [Estrategias por Tipo de Análisis](#estrategias-por-tipo-de-análisis)
3. [Combinación de Modelos (Ensemble)](#combinación-de-modelos-ensemble)
4. [Implementación Práctica](#implementación-práctica)
5. [Configuración por Plan de Suscripción](#configuración-por-plan-de-suscripción)

---

## 🧠 Modelos de IA Disponibles

### 1. **IA Tradicional (Machine Learning)**

#### **A. Random Forest Classifier**
- **Propósito**: Clasificación de señales de trading (BUY/SELL/HOLD)
- **Características**:
  - 200 estimadores
  - Features: RSI, MACD, Momentum, Volatilidad, Volumen
  - Precisión objetivo: >60%
  - Entrenamiento automático

#### **B. Gradient Boosting Regressor**
- **Propósito**: Predicción de precios futuros
- **Características**:
  - 150 estimadores
  - Predicción de precios a 5-14 días
  - Métricas: MSE, MAE
  - Optimización de hiperparámetros

#### **C. LSTM (Long Short-Term Memory)**
- **Propósito**: Predicción de series temporales de precios
- **Arquitectura**:
  - 3 capas LSTM (50, 50, 25 unidades)
  - Dropout 0.2 entre capas
  - Lookback window: 60 períodos
  - Features: Precio, Volumen, RSI, MACD, Momentum

### 2. **Reinforcement Learning (RL)**

#### **A. DQN (Deep Q-Network)**
- **Propósito**: Decisiones discretas de trading
- **Acciones**: 7 niveles (HOLD, BUY 33%, BUY 66%, BUY 100%, SELL 33%, SELL 66%, SELL 100%)
- **Estado**: 70 dimensiones (precios + indicadores + portafolio)
- **Recompensa**: Basada en P&L, drawdown, Sharpe ratio

#### **B. PPO (Proximal Policy Optimization)**
- **Propósito**: Optimización de políticas continuas
- **Ventajas**: Más estable que DQN, mejor para mercados volátiles
- **Características**:
  - Policy network + Value network
  - GAE (Generalized Advantage Estimation)
  - Entrenamiento con múltiples episodios

### 3. **Ensemble AI**
- **Propósito**: Combinación de IA tradicional + RL
- **Pesos**: 40% Tradicional + 60% RL
- **Lógica**: Decisión final basada en consenso de modelos

---

## 📊 Estrategias por Tipo de Análisis

### 1. **Análisis Técnico**

#### **A. Indicadores de Momentum**
```python
# RSI Strategy
def rsi_strategy(data, rsi_period=14, oversold=30, overbought=70):
    if data['RSI'].iloc[-1] < oversold:
        return 'BUY'
    elif data['RSI'].iloc[-1] > overbought:
        return 'SELL'
    return 'HOLD'
```

#### **B. Indicadores de Tendencia**
```python
# MACD Strategy
def macd_strategy(data):
    if (data['MACD'].iloc[-1] > data['MACD_signal'].iloc[-1] and 
        data['MACD_histogram'].iloc[-1] > 0):
        return 'BUY'
    elif (data['MACD'].iloc[-1] < data['MACD_signal'].iloc[-1] and 
          data['MACD_histogram'].iloc[-1] < 0):
        return 'SELL'
    return 'HOLD'
```

#### **C. Indicadores de Volatilidad**
```python
# Bollinger Bands Strategy
def bollinger_strategy(data):
    current_price = data['Close'].iloc[-1]
    bb_upper = data['BB_upper'].iloc[-1]
    bb_lower = data['BB_lower'].iloc[-1]
    
    if current_price < bb_lower:
        return 'BUY'  # Precio bajo la banda inferior
    elif current_price > bb_upper:
        return 'SELL'  # Precio sobre la banda superior
    return 'HOLD'
```

### 2. **Soportes y Resistencias**

#### **A. Detección Automática**
```python
def support_resistance_strategy(data, window=20):
    # Soporte: mínimo local
    support = data['Low'].rolling(window).min()
    # Resistencia: máximo local
    resistance = data['High'].rolling(window).max()
    
    current_price = data['Close'].iloc[-1]
    current_support = support.iloc[-1]
    current_resistance = resistance.iloc[-1]
    
    # Calcular distancia a soporte/resistencia
    distance_to_support = (current_price - current_support) / current_price
    distance_to_resistance = (current_resistance - current_price) / current_price
    
    if distance_to_support < 0.02:  # Cerca del soporte
        return 'BUY'
    elif distance_to_resistance < 0.02:  # Cerca de la resistencia
        return 'SELL'
    return 'HOLD'
```

#### **B. Breakout Detection**
```python
def breakout_strategy(data, threshold=0.02):
    current_price = data['Close'].iloc[-1]
    resistance = data['Resistance'].iloc[-1]
    support = data['Support'].iloc[-1]
    
    # Breakout alcista
    if current_price > resistance * (1 + threshold):
        return 'BUY'
    # Breakout bajista
    elif current_price < support * (1 - threshold):
        return 'SELL'
    return 'HOLD'
```

### 3. **Análisis de Tendencias**

#### **A. Moving Averages**
```python
def trend_strategy(data):
    sma_20 = data['SMA_20'].iloc[-1]
    sma_50 = data['SMA_50'].iloc[-1]
    current_price = data['Close'].iloc[-1]
    
    # Tendencia alcista
    if current_price > sma_20 > sma_50:
        return 'BUY'
    # Tendencia bajista
    elif current_price < sma_20 < sma_50:
        return 'SELL'
    return 'HOLD'
```

#### **B. Price Action**
```python
def price_action_strategy(data, lookback=5):
    # Patrones de velas
    recent_data = data.tail(lookback)
    
    # Doji pattern
    doji_threshold = 0.001
    for i in range(len(recent_data)):
        open_price = recent_data['Open'].iloc[i]
        close_price = recent_data['Close'].iloc[i]
        if abs(close_price - open_price) / open_price < doji_threshold:
            return 'HOLD'  # Indecisión del mercado
    
    # Hammer pattern
    for i in range(len(recent_data)):
        high = recent_data['High'].iloc[i]
        low = recent_data['Low'].iloc[i]
        close = recent_data['Close'].iloc[i]
        open_price = recent_data['Open'].iloc[i]
        
        body = abs(close - open_price)
        lower_shadow = min(open_price, close) - low
        upper_shadow = high - max(open_price, close)
        
        if lower_shadow > 2 * body and upper_shadow < body:
            return 'BUY'  # Hammer pattern
    
    return 'HOLD'
```

### 4. **Análisis Fundamental**

#### **A. Eventos Económicos**
```python
def fundamental_strategy(market_data, economic_events):
    current_signal = 'HOLD'
    
    for event in economic_events:
        if event['impact'] == 'HIGH':
            if event['type'] == 'NFP' and event['actual'] > event['expected']:
                current_signal = 'BUY'  # USD fuerte
            elif event['type'] == 'INTEREST_RATE' and event['actual'] > event['expected']:
                current_signal = 'BUY'  # Tasa de interés sube
    
    return current_signal
```

#### **B. Noticias y Sentimiento**
```python
def news_sentiment_strategy(news_data, sentiment_score):
    if sentiment_score > 0.7:
        return 'BUY'  # Sentimiento muy positivo
    elif sentiment_score < -0.7:
        return 'SELL'  # Sentimiento muy negativo
    return 'HOLD'
```

### 5. **Análisis de Volumen**

#### **A. Volume Profile**
```python
def volume_strategy(data):
    current_volume = data['Volume'].iloc[-1]
    avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
    volume_ratio = current_volume / avg_volume
    
    if volume_ratio > 1.5:  # Volumen alto
        if data['Close'].iloc[-1] > data['Open'].iloc[-1]:
            return 'BUY'  # Volumen alto en subida
        else:
            return 'SELL'  # Volumen alto en bajada
    return 'HOLD'
```

#### **B. Volume Weighted Average Price (VWAP)**
```python
def vwap_strategy(data):
    # Calcular VWAP
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    
    current_price = data['Close'].iloc[-1]
    current_vwap = vwap.iloc[-1]
    
    if current_price > current_vwap * 1.01:
        return 'BUY'  # Precio sobre VWAP
    elif current_price < current_vwap * 0.99:
        return 'SELL'  # Precio bajo VWAP
    return 'HOLD'
```

---

## 🔄 Combinación de Modelos (Ensemble)

### 1. **Ensemble Tradicional + RL**

```python
def ensemble_strategy(traditional_ai, rl_agent, market_data):
    # Predicción IA tradicional
    traditional_pred = traditional_ai.predict_signal(market_data)
    
    # Predicción RL
    rl_state = prepare_rl_state(market_data)
    rl_action = rl_agent.predict_action(rl_state)
    
    # Combinar predicciones
    if traditional_pred['signal'] == 'BUY' and rl_action.action == ActionType.BUY:
        return {
            'action': 'BUY',
            'confidence': (traditional_pred['confidence'] * 0.4 + 
                         rl_action.confidence * 0.6),
            'reasoning': f"Traditional: {traditional_pred['signal']} + RL: {rl_action.reasoning}"
        }
    elif traditional_pred['signal'] == 'SELL' and rl_action.action == ActionType.SELL:
        return {
            'action': 'SELL',
            'confidence': (traditional_pred['confidence'] * 0.4 + 
                         rl_action.confidence * 0.6),
            'reasoning': f"Traditional: {traditional_pred['signal']} + RL: {rl_action.reasoning}"
        }
    else:
        # Señales conflictivas - decisión conservadora
        return {
            'action': 'HOLD',
            'confidence': max(traditional_pred['confidence'], rl_action.confidence) * 0.7,
            'reasoning': f"Conflicting signals - Traditional: {traditional_pred['signal']}, RL: {rl_action.action.name}"
        }
```

### 2. **Multi-Timeframe Analysis**

```python
def multi_timeframe_strategy(data_1h, data_4h, data_1d):
    # Análisis en múltiples timeframes
    signals = {}
    
    # 1H timeframe
    signals['1h'] = traditional_ai.predict_signal(data_1h)
    
    # 4H timeframe
    signals['4h'] = traditional_ai.predict_signal(data_4h)
    
    # 1D timeframe
    signals['1d'] = traditional_ai.predict_signal(data_1d)
    
    # Combinar señales
    buy_signals = sum(1 for s in signals.values() if s['signal'] == 'BUY')
    sell_signals = sum(1 for s in signals.values() if s['signal'] == 'SELL')
    
    if buy_signals >= 2:
        return 'BUY'
    elif sell_signals >= 2:
        return 'SELL'
    return 'HOLD'
```

### 3. **Risk-Adjusted Ensemble**

```python
def risk_adjusted_ensemble(market_data, volatility_threshold=0.02):
    # Calcular volatilidad actual
    current_volatility = market_data['Close'].pct_change().rolling(20).std().iloc[-1]
    
    if current_volatility > volatility_threshold:
        # Mercado volátil - usar más RL
        rl_weight = 0.8
        traditional_weight = 0.2
    else:
        # Mercado estable - usar más IA tradicional
        rl_weight = 0.3
        traditional_weight = 0.7
    
    # Obtener predicciones
    traditional_pred = traditional_ai.predict_signal(market_data)
    rl_action = rl_agent.predict_action(prepare_rl_state(market_data))
    
    # Combinar con pesos ajustados
    final_confidence = (traditional_pred['confidence'] * traditional_weight + 
                       rl_action.confidence * rl_weight)
    
    return {
        'action': determine_final_action(traditional_pred, rl_action),
        'confidence': final_confidence,
        'volatility': current_volatility,
        'weights': {'traditional': traditional_weight, 'rl': rl_weight}
    }
```

---

## ⚙️ Implementación Práctica

### 1. **Configuración de Modelos**

```python
# Inicializar modelos
traditional_ai = AdvancedTradingAI()
rl_system_dqn = RLTradingSystem(data_source, agent_type='DQN')
rl_system_ppo = RLTradingSystem(data_source, agent_type='PPO')

# Entrenar modelos
traditional_ai.train_signal_classifier(market_data)
rl_system_dqn.train_agent(episodes=1000)
rl_system_ppo.train_agent(episodes=1000)
```

### 2. **Estrategia Completa**

```python
def complete_trading_strategy(symbol, timeframe='1h'):
    # 1. Obtener datos de mercado
    market_data = get_market_data(symbol, timeframe, period='6mo')
    
    # 2. Análisis técnico
    technical_signals = technical_analyzer.generate_signals(market_data)
    
    # 3. Análisis fundamental (si está disponible)
    fundamental_signals = fundamental_analyzer.analyze(symbol)
    
    # 4. Predicción IA tradicional
    ai_prediction = traditional_ai.predict_signal(market_data)
    
    # 5. Predicción RL
    rl_prediction = rl_system_dqn.predict_action(prepare_rl_state(market_data))
    
    # 6. Ensemble final
    final_decision = ensemble_strategy(
        traditional_pred=ai_prediction,
        rl_action=rl_prediction,
        technical_signals=technical_signals,
        fundamental_signals=fundamental_signals
    )
    
    # 7. Risk management
    position_size = calculate_position_size(final_decision['confidence'])
    stop_loss = calculate_stop_loss(market_data, final_decision['action'])
    take_profit = calculate_take_profit(market_data, final_decision['action'])
    
    return {
        'symbol': symbol,
        'action': final_decision['action'],
        'confidence': final_decision['confidence'],
        'position_size': position_size,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'reasoning': final_decision['reasoning'],
        'timestamp': datetime.now()
    }
```

### 3. **Backtesting**

```python
def backtest_strategy(strategy_func, historical_data):
    backtest_engine = BacktestEngine(initial_capital=100000)
    
    results = backtest_engine.run_backtest(
        data=historical_data,
        strategy_func=strategy_func
    )
    
    # Métricas de rendimiento
    metrics = backtest_engine.get_performance_metrics()
    
    return {
        'total_return': metrics['total_return'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'win_rate': metrics['win_rate'],
        'profit_factor': metrics['profit_factor']
    }
```

---

## 💰 Configuración por Plan de Suscripción

### **Plan FREEMIUM ($0/mes)**
- ✅ Random Forest básico (1 indicador: RSI)
- ✅ Predicciones limitadas (3 días)
- ✅ Backtesting básico (30 días)
- ✅ 1 par de trading

### **Plan BÁSICO ($29/mes)**
- ✅ Random Forest completo (3 indicadores: RSI, MACD, Bollinger)
- ✅ Predicciones mejoradas (7 días)
- ✅ Backtesting avanzado (90 días)
- ✅ 5 pares de trading

### **Plan PRO ($99/mes)**
- ✅ Random Forest Premium + LSTM
- ✅ Reinforcement Learning (DQN)
- ✅ Ensemble AI (Tradicional + RL)
- ✅ Todos los indicadores técnicos
- ✅ Predicciones avanzadas (14 días)
- ✅ Backtesting profesional

### **Plan ELITE ($299/mes)**
- ✅ Random Forest Elite + máxima precisión
- ✅ Reinforcement Learning completo (DQN + PPO)
- ✅ Ensemble AI avanzado optimizado
- ✅ Predicciones elite (30 días)
- ✅ Backtesting institucional
- ✅ Custom Models personalizados

---

## 🎯 Recomendaciones de Uso

### **Para Mercados Laterales (Sideways)**
- **Modelo Principal**: Random Forest + RSI
- **Estrategia**: Range trading con soportes y resistencias
- **Timeframe**: 1H - 4H

### **Para Mercados Tendenciales**
- **Modelo Principal**: LSTM + Moving Averages
- **Estrategia**: Trend following
- **Timeframe**: 4H - 1D

### **Para Mercados Volátiles**
- **Modelo Principal**: RL (PPO) + Bollinger Bands
- **Estrategia**: Mean reversion
- **Timeframe**: 15M - 1H

### **Para Mercados de Alta Frecuencia**
- **Modelo Principal**: Ensemble (Tradicional + RL)
- **Estrategia**: Multi-timeframe analysis
- **Timeframe**: 1M - 15M

---

## 📈 Monitoreo y Optimización

### **1. Métricas de Rendimiento**
- **Accuracy**: Precisión de las predicciones
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Maximum Drawdown**: Máxima pérdida consecutiva
- **Win Rate**: Porcentaje de operaciones ganadoras
- **Profit Factor**: Ratio ganancias/pérdidas

### **2. Auto-Entrenamiento**
- **Frecuencia**: Cada 6 horas (configurable)
- **Trigger**: Drift detection > 0.1
- **Validación**: Cross-validation en datos históricos
- **Deployment**: Solo si mejora rendimiento

### **3. Alertas y Notificaciones**
- **Señales de Trading**: Tiempo real
- **Cambios de Modelo**: Cuando se actualiza
- **Anomalías**: Detección de comportamientos inusuales
- **Performance**: Reportes diarios/semanales

---

## 🔧 Configuración Técnica

### **Variables de Entorno**
```bash
# AI/ML Configuration
MODEL_PATH=models/
AUTO_TRAINING_ENABLED=true
AUTO_TRAINING_INTERVAL_HOURS=6
DRIFT_DETECTION_THRESHOLD=0.1

# Reinforcement Learning
RL_AGENT_TYPE=DQN
RL_TRAINING_EPISODES=1000
RL_EVALUATION_EPISODES=10
RL_MODEL_SAVE_INTERVAL=100
```

### **Estructura de Archivos**
```
models/
├── signal_classifier.pkl
├── lstm_model.h5
├── rl_trading_agent_dqn.pth
├── rl_trading_agent_ppo.pth
└── scaler.pkl
```

---

## 📚 Recursos Adicionales

### **Documentación Técnica**
- [API Reference](backend/src/README_SUBSCRIPTION_SYSTEM.md)
- [Database Schema](backend/src/models/database_models.py)
- [Configuration Guide](backend/env_template.txt)

### **Ejemplos de Código**
- [AI Models](backend/src/ai_models.py)
- [RL Trading Agent](backend/src/rl_trading_agent.py)
- [Auto Training System](backend/src/auto_training_system.py)

### **Herramientas de Desarrollo**
- [Technical Analysis](backend/src/main.py)
- [Backtesting Engine](backend/src/ai_models.py)
- [Performance Metrics](backend/src/rl_trading_agent.py)

---

*Este documento se actualiza automáticamente con cada nueva versión del sistema.* 