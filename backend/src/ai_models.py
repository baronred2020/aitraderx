# ai_models.py - Modelos avanzados de IA
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingAI:
    """Modelos avanzados de IA para trading"""
    
    def __init__(self):
        self.signal_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
        self.price_predictor = GradientBoostingRegressor(n_estimators=150, random_state=42)
        self.lstm_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
    
    def create_features(self, df):
        """Crea features avanzadas para el modelo"""
        # Features técnicas básicas
        df['rsi'] = self.calculate_rsi(df['Close'])
        df['macd'], df['macd_signal'] = self.calculate_macd(df['Close'])
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['Close'])
        
        # Features de momentum
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Features de volatilidad
        df['volatility_5'] = df['Close'].rolling(5).std()
        df['volatility_10'] = df['Close'].rolling(10).std()
        df['volatility_20'] = df['Close'].rolling(20).std()
        
        # Features de volumen
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['price_volume'] = df['Close'] * df['Volume']
        
        # Features de precio
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        
        # Features temporales
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        
        # Features de tendencia
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_50'] = df['Close'].rolling(50).mean()
        df['trend_5_20'] = df['sma_5'] / df['sma_20'] - 1
        df['trend_20_50'] = df['sma_20'] / df['sma_50'] - 1
        
        return df
    
    def calculate_rsi(self, prices, window=14):
        """Calcula RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calcula MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calcula Bandas de Bollinger"""
        sma = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return upper, lower
    
    def create_signals(self, df):
        """Crea señales de trading"""
        signals = []
        
        for i in range(len(df)):
            if i < 50:  # No suficientes datos
                signals.append('HOLD')
                continue
            
            # Condiciones para BUY
            buy_conditions = [
                df['rsi'].iloc[i] < 30,  # RSI sobrevendido
                df['Close'].iloc[i] < df['bb_lower'].iloc[i],  # Precio bajo BB inferior
                df['macd'].iloc[i] > df['macd_signal'].iloc[i],  # MACD alcista
                df['trend_5_20'].iloc[i] > 0.01,  # Tendencia alcista
                df['volume_ratio'].iloc[i] > 1.2  # Volumen alto
            ]
            
            # Condiciones para SELL
            sell_conditions = [
                df['rsi'].iloc[i] > 70,  # RSI sobrecomprado
                df['Close'].iloc[i] > df['bb_upper'].iloc[i],  # Precio sobre BB superior
                df['macd'].iloc[i] < df['macd_signal'].iloc[i],  # MACD bajista
                df['trend_5_20'].iloc[i] < -0.01,  # Tendencia bajista
                df['momentum_5'].iloc[i] < -0.02  # Momentum negativo
            ]
            
            buy_score = sum(buy_conditions)
            sell_score = sum(sell_conditions)
            
            if buy_score >= 3:
                signals.append('BUY')
            elif sell_score >= 3:
                signals.append('SELL')
            else:
                signals.append('HOLD')
        
        return signals
    
    def train_signal_classifier(self, df):
        """Entrena el clasificador de señales"""
        # Crear features
        df = self.create_features(df)
        
        # Crear señales target
        signals = self.create_signals(df)
        df['signal'] = signals
        
        # Seleccionar features para el modelo
        feature_columns = [
            'rsi', 'macd', 'momentum_5', 'momentum_10', 'momentum_20',
            'volatility_5', 'volatility_10', 'volume_ratio', 'high_low_ratio',
            'trend_5_20', 'trend_20_50', 'day_of_week', 'month'
        ]
        
        # Preparar datos
        X = df[feature_columns].fillna(0)
        y = df['signal']
        
        # Filtrar datos válidos
        valid_mask = ~X.isin([np.inf, -np.inf]).any(axis=1)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 100:
            return False
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.signal_classifier.fit(X_train_scaled, y_train)
        
        # Evaluar modelo
        y_pred = self.signal_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Precisión del clasificador de señales: {accuracy:.2f}")
        
        return accuracy > 0.6
    
    def create_lstm_model(self, input_shape):
        """Crea modelo LSTM para predicción de precios"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(25),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def prepare_lstm_data(self, df, lookback=60):
        """Prepara datos para LSTM"""
        df = self.create_features(df)
        
        feature_columns = [
            'Close', 'Volume', 'rsi', 'macd', 'momentum_5',
            'volatility_5', 'volume_ratio', 'trend_5_20'
        ]
        
        data = df[feature_columns].fillna(method='ffill').values
        
        # Normalizar datos
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(lookback, len(data_scaled)):
            X.append(data_scaled[i-lookback:i])
            y.append(data_scaled[i, 0])  # Precio de cierre
        
        return np.array(X), np.array(y), scaler
    
    def train_lstm_predictor(self, df):
        """Entrena el modelo LSTM"""
        try:
            X, y, scaler = self.prepare_lstm_data(df)
            
            if len(X) < 100:
                return False
            
            # Dividir datos
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            # Crear y entrenar modelo
            self.lstm_model = self.create_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            # Entrenar con early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                patience=10, restore_best_weights=True
            )
            
            history = self.lstm_model.fit(
                X_train, y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Evaluar modelo
            train_loss = self.lstm_model.evaluate(X_train, y_train, verbose=0)
            test_loss = self.lstm_model.evaluate(X_test, y_test, verbose=0)
            
            print(f"LSTM - Loss entrenamiento: {train_loss[0]:.4f}")
            print(f"LSTM - Loss prueba: {test_loss[0]:.4f}")
            
            return test_loss[0] < 0.1
            
        except Exception as e:
            print(f"Error entrenando LSTM: {e}")
            return False
    
    def predict_signal(self, df):
        """Predice señal de trading"""
        try:
            df = self.create_features(df)
            
            feature_columns = [
                'rsi', 'macd', 'momentum_5', 'momentum_10', 'momentum_20',
                'volatility_5', 'volatility_10', 'volume_ratio', 'high_low_ratio',
                'trend_5_20', 'trend_20_50', 'day_of_week', 'month'
            ]
            
            X = df[feature_columns].fillna(0).iloc[-1:].values
            X_scaled = self.scaler.transform(X)
            
            # Predicción
            signal = self.signal_classifier.predict(X_scaled)[0]
            probability = self.signal_classifier.predict_proba(X_scaled).max()
            
            return {
                'signal': signal,
                'confidence': int(probability * 100),
                'features': dict(zip(feature_columns, X[0]))
            }
            
        except Exception as e:
            print(f"Error prediciendo señal: {e}")
            return {'signal': 'HOLD', 'confidence': 50, 'features': {}}
    
    def predict_price_lstm(self, df, days_ahead=5):
        """Predice precio usando LSTM"""
        try:
            if self.lstm_model is None:
                return None
            
            X, _, scaler = self.prepare_lstm_data(df)
            
            if len(X) == 0:
                return None
            
            # Usar últimos datos para predicción
            last_sequence = X[-1:]
            
            # Predicir múltiples días
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                pred = self.lstm_model.predict(current_sequence, verbose=0)
                predictions.append(pred[0, 0])
                
                # Actualizar secuencia
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, 0] = pred[0, 0]
            
            # Desnormalizar predicciones
            current_price = df['Close'].iloc[-1]
            predicted_prices = []
            
            for pred in predictions:
                # Aproximación simple de desnormalización
                predicted_price = current_price * (1 + pred * 0.1)
                predicted_prices.append(predicted_price)
            
            return {
                'current_price': current_price,
                'predictions': predicted_prices,
                'target_price': predicted_prices[-1],
                'days_ahead': days_ahead
            }
            
        except Exception as e:
            print(f"Error prediciendo precio LSTM: {e}")
            return None
    
    def save_models(self, path="models/"):
        """Guarda los modelos entrenados"""
        try:
            joblib.dump(self.signal_classifier, f"{path}/signal_classifier.pkl")
            joblib.dump(self.scaler, f"{path}/scaler.pkl")
            
            if self.lstm_model:
                self.lstm_model.save(f"{path}/lstm_model.h5")
            
            print("Modelos guardados exitosamente")
            
        except Exception as e:
            print(f"Error guardando modelos: {e}")
    
    def load_models(self, path="models/"):
        """Carga los modelos entrenados"""
        try:
            self.signal_classifier = joblib.load(f"{path}/signal_classifier.pkl")
            self.scaler = joblib.load(f"{path}/scaler.pkl")
            
            try:
                self.lstm_model = tf.keras.models.load_model(f"{path}/lstm_model.h5")
            except:
                print("No se pudo cargar modelo LSTM")
            
            self.is_trained = True
            print("Modelos cargados exitosamente")
            
        except Exception as e:
            print(f"Error cargando modelos: {e}")

# backtesting.py - Sistema de backtesting
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class BacktestEngine:
    """Motor de backtesting para estrategias de trading"""
    
    def __init__(self, initial_capital=100000, commission=0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.reset()
    
    def reset(self):
        """Reinicia el estado del backtest"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.daily_returns = []
        self.max_drawdown = 0
        self.peak_value = self.initial_capital
    
    def execute_trade(self, symbol, action, price, quantity, date):
        """Ejecuta una operación de trading"""
        trade_value = price * quantity
        commission_cost = trade_value * self.commission
        
        if action == 'BUY':
            total_cost = trade_value + commission_cost
            
            if total_cost <= self.capital:
                self.capital -= total_cost
                
                if symbol in self.positions:
                    # Promedio de costo
                    current_qty = self.positions[symbol]['quantity']
                    current_avg = self.positions[symbol]['avg_price']
                    new_avg = ((current_avg * current_qty) + trade_value) / (current_qty + quantity)
                    
                    self.positions[symbol]['quantity'] += quantity
                    self.positions[symbol]['avg_price'] = new_avg
                else:
                    self.positions[symbol] = {
                        'quantity': quantity,
                        'avg_price': price
                    }
                
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': action,
                    'price': price,
                    'quantity': quantity,
                    'commission': commission_cost,
                    'capital_after': self.capital
                })
                
                return True
            else:
                return False  # Fondos insuficientes
        
        elif action == 'SELL':
            if symbol in self.positions and self.positions[symbol]['quantity'] >= quantity:
                revenue = trade_value - commission_cost
                self.capital += revenue
                
                self.positions[symbol]['quantity'] -= quantity
                
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
                
                # Calcular P&L
                avg_price = self.positions.get(symbol, {}).get('avg_price', 0)
                pnl = (price - avg_price) * quantity - commission_cost
                
                self.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'action': action,
                    'price': price,
                    'quantity': quantity,
                    'commission': commission_cost,
                    'pnl': pnl,
                    'capital_after': self.capital
                })
                
                return True
            else:
                return False  # Posición insuficiente
    
    def calculate_portfolio_value(self, current_prices):
        """Calcula el valor actual del portafolio"""
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['quantity'] * current_prices[symbol]
        
        total_value = self.capital + positions_value
        self.portfolio_values.append(total_value)
        
        # Calcular drawdown
        if total_value > self.peak_value:
            self.peak_value = total_value
        
        current_drawdown = (self.peak_value - total_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        return total_value
    
    def run_backtest(self, data, strategy_func, start_date=None, end_date=None):
        """Ejecuta el backtest con una estrategia dada"""
        self.reset()
        
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        for i, (date, row) in enumerate(data.iterrows()):
            if i == 0:
                continue  # Saltar primera fila
            
            # Obtener datos históricos hasta la fecha actual
            historical_data = data.loc[:date]
            
            # Ejecutar estrategia
            signals = strategy_func(historical_data)
            
            if signals:
                for signal in signals:
                    symbol = signal.get('symbol')
                    action = signal.get('action')
                    quantity = signal.get('quantity', 100)
                    price = row.get('Close', row.get(f'{symbol}_Close'))
                    
                    if symbol and action and price:
                        self.execute_trade(symbol, action, price, quantity, date)
            
            # Actualizar valor del portafolio
            current_prices = {col.replace('_Close', ''): row[col] 
                            for col in row.index if '_Close' in col}
            if not current_prices:
                current_prices = {'main': row.get('Close', 0)}
            
            portfolio_value = self.calculate_portfolio_value(current_prices)
            
            # Calcular retorno diario
            if len(self.portfolio_values) > 1:
                daily_return = (portfolio_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
                self.daily_returns.append(daily_return)
    
    def get_performance_metrics(self):
        """Calcula métricas de rendimiento"""
        if not self.portfolio_values:
            return {}
        
        final_value = self.portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calcular métricas anualizadas
        num_days = len(self.portfolio_values)
        annual_return = ((1 + total_return) ** (252 / num_days)) - 1
        
        # Volatilidad
        if self.daily_returns:
            volatility = np.std(self.daily_returns) * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Estadísticas de trades
        winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_factor = abs(sum(t['pnl'] for t in winning_trades) / 
                          sum(t['pnl'] for t in losing_trades)) if losing_trades else float('inf')
        
        return {
            'total_return': total_return * 100,
            'annual_return': annual_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': self.max_drawdown * 100,
            'win_rate': win_rate * 100,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_capital': final_value
        }
    
    def plot_results(self):
        """Genera gráficos de resultados"""
        if not self.portfolio_values:
            print("No hay datos para graficar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gráfico de evolución del portafolio
        axes[0, 0].plot(self.portfolio_values)
        axes[0, 0].axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Evolución del Portafolio')
        axes[0, 0].set_ylabel('Valor ($)')
        
        # Gráfico de drawdown
        drawdowns = []
        peak = self.initial_capital
        for value in self.portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            drawdowns.append(drawdown)
        
        axes[0, 1].fill_between(range(len(drawdowns)), drawdowns, alpha=0.7, color='red')
        axes[0, 1].set_title('Drawdown (%)')
        axes[0, 1].set_ylabel('Drawdown (%)')
        
        # Distribución de retornos diarios
        if self.daily_returns:
            axes[1, 0].hist(self.daily_returns, bins=50, alpha=0.7)
            axes[1, 0].set_title('Distribución de Retornos Diarios')
            axes[1, 0].set_xlabel('Retorno')
        
        # P&L por trade
        trade_pnls = [t.get('pnl', 0) for t in self.trades if 'pnl' in t]
        if trade_pnls:
            axes[1, 1].bar(range(len(trade_pnls)), trade_pnls, 
                          color=['green' if pnl > 0 else 'red' for pnl in trade_pnls])
            axes[1, 1].set_title('P&L por Trade')
            axes[1, 1].set_xlabel('Trade #')
            axes[1, 1].set_ylabel('P&L ($)')
        
        plt.tight_layout()
        plt.show()

# strategy_examples.py - Ejemplos de estrategias
def simple_moving_average_strategy(data, short_window=20, long_window=50):
    """Estrategia simple de medias móviles"""
    if len(data) < long_window:
        return []
    
    data['SMA_short'] = data['Close'].rolling(short_window).mean()
    data['SMA_long'] = data['Close'].rolling(long_window).mean()
    
    signals = []
    
    # Generar señales en cruces
    if len(data) >= 2:
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Golden Cross (compra)
        if (current['SMA_short'] > current['SMA_long'] and 
            previous['SMA_short'] <= previous['SMA_long']):
            signals.append({
                'symbol': 'main',
                'action': 'BUY',
                'quantity': 100
            })
        
        # Death Cross (venta)
        elif (current['SMA_short'] < current['SMA_long'] and 
              previous['SMA_short'] >= previous['SMA_long']):
            signals.append({
                'symbol': 'main',
                'action': 'SELL',
                'quantity': 100
            })
    
    return signals

def rsi_strategy(data, rsi_period=14, oversold=30, overbought=70):
    """Estrategia basada en RSI"""
    if len(data) < rsi_period + 1:
        return []
    
    # Calcular RSI
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    signals = []
    current_rsi = data['RSI'].iloc[-1]
    
    if current_rsi < oversold:
        signals.append({
            'symbol': 'main',
            'action': 'BUY',
            'quantity': 100
        })
    elif current_rsi > overbought:
        signals.append({
            'symbol': 'main',
            'action': 'SELL',
            'quantity': 100
        })
    
    return signals

def ai_strategy(data, ai_model):
    """Estrategia basada en modelo de IA"""
    if len(data) < 50:
        return []
    
    try:
        prediction = ai_model.predict_signal(data)
        signal = prediction.get('signal', 'HOLD')
        confidence = prediction.get('confidence', 50)
        
        signals = []
        
        if signal == 'BUY' and confidence > 70:
            signals.append({
                'symbol': 'main',
                'action': 'BUY',
                'quantity': 100
            })
        elif signal == 'SELL' and confidence > 70:
            signals.append({
                'symbol': 'main',
                'action': 'SELL',
                'quantity': 100
            })
        
        return signals
        
    except Exception as e:
        print(f"Error en estrategia IA: {e}")
        return []

# deployment.py - Scripts de despliegue
import os
import subprocess
import json

def setup_environment():
    """Configura el ambiente de producción"""
    
    # Crear directorios necesarios
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    # Variables de ambiente
    env_vars = {
        'ENVIRONMENT': 'production',
        'DATABASE_URL': 'postgresql://trader:password@localhost:5432/trading_db',
        'REDIS_URL': 'redis://localhost:6379',
        'API_KEY_ALPHA_VANTAGE': 'your_api_key_here',
        'API_KEY_FINNHUB': 'd1nvqv1r01qtrauu8fggd1nvqv1r01qtrauu8fh0',
        'SECRET_KEY': 'your_secret_key_here'
    }
    
    # Escribir archivo .env
    with open('.env', 'w') as f:
        for key, value in env_vars.items():
            f.write(f'{key}={value}\n')
    
    print("✅ Ambiente configurado")

def deploy_with_docker():
    """Despliega la aplicación usando Docker"""
    
    try:
        # Construir imagen
        print("🐳 Construyendo imagen Docker...")
        subprocess.run(['docker-compose', 'build'], check=True)
        
        # Iniciar servicios
        print("🚀 Iniciando servicios...")
        subprocess.run(['docker-compose', 'up', '-d'], check=True)
        
        print("✅ Aplicación desplegada exitosamente")
        print("📊 API disponible en: http://localhost:8000")
        print("📋 Documentación en: http://localhost:8000/docs")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en despliegue: {e}")

def health_check():
    """Verifica el estado de los servicios"""
    import requests
    
    services = {
        'API': 'http://localhost:8000/health',
        'Database': 'postgresql://trader:password@localhost:5432/trading_db',
        'Redis': 'redis://localhost:6379'
    }
    
    for service, url in services.items():
        try:
            if service == 'API':
                response = requests.get(url, timeout=5)
                status = "✅ OK" if response.status_code == 200 else "❌ ERROR"
            else:
                status = "⏳ CHECKING"
            
            print(f"{service}: {status}")
            
        except Exception as e:
            print(f"{service}: ❌ ERROR - {e}")

if __name__ == "__main__":
    print("🔧 Configurando sistema de trading...")
    setup_environment()
    deploy_with_docker()
    
    import time
    print("⏳ Esperando que los servicios inicien...")
    time.sleep(10)
    
    health_check()
    print("\n🎉 Sistema listo para usar!")