# ====================================================================
# CPU-ONLY TRADING BRAIN - VERSIÓN MEJORADA
# Soluciones a los problemas identificados
# ====================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import warnings
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# ====================================================================
# ARQUITECTURA MEJORADA CON TÉCNICAS ANTI-OVERFITTING
# ====================================================================

class ImprovedTransformer(nn.Module):
    """Transformer mejorado con regularización"""
    
    def __init__(self, input_dim, d_model=128, num_heads=4, num_layers=3):
        super().__init__()
        
        # Proyección de entrada más robusta
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(0.2)
        )
        
        # Attention con más heads
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, batch_first=True, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Normas separadas para atención y FFN
        self.attn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(num_layers)
        ])
        
        # FFN más robustos
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(0.1)
            ) for _ in range(num_layers)
        ])
        
        # Classifier mejorado
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 3)
        )
        
        # Confidence con más capas
        self.confidence = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Transformer layers con residual connections mejorados
        for i, (attn, attn_norm, ffn, ffn_norm) in enumerate(
            zip(self.attention_layers, self.attn_norms, self.ffns, self.ffn_norms)
        ):
            # Self-attention con residual
            residual = x
            attn_out, _ = attn(x, x, x)
            x = attn_norm(residual + attn_out)
            
            # FFN con residual
            residual = x
            ffn_out = ffn(x)
            x = ffn_norm(residual + ffn_out)
        
        # Global average pooling
        pooled = x.mean(dim=1)
        
        return {
            'direction': self.classifier(pooled),
            'confidence': self.confidence(pooled)
        }

class ImprovedLSTM(nn.Module):
    """LSTM mejorado con attention y regularización"""
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        
        # LSTM con más capacidad
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism mejorado
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier robusto
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)
        )
        
        # Confidence mejorado
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Weighted average
        context = (lstm_out * attn_weights).sum(dim=1)
        
        return {
            'direction': self.classifier(context),
            'confidence': self.confidence(context)
        }

class ImprovedLinearEnsemble(nn.Module):
    """Ensemble linear mejorado con regularización"""
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        # Múltiples ramas con diferentes arquitecturas
        self.branches = nn.ModuleList()
        
        for i, hidden_dim in enumerate(hidden_dims):
            if i == 0:
                # Rama más compleja
                branch = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden_dim // 2, 3)
                )
            else:
                # Ramas más simples
                branch = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 2, 3)
                )
            self.branches.append(branch)
        
        # Fusion layer mejorado
        self.fusion = nn.Sequential(
            nn.Linear(len(hidden_dims) * 3, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
        # Confidence mejorado
        self.confidence = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Flatten sequence
        x_flat = x.view(x.size(0), -1)
        
        # Multiple branches
        branch_outputs = []
        for branch in self.branches:
            branch_outputs.append(branch(x_flat))
        
        # Concatenate and fuse
        combined = torch.cat(branch_outputs, dim=1)
        direction = self.fusion(combined)
        confidence = self.confidence(x_flat)
        
        return {
            'direction': direction,
            'confidence': confidence
        }

# ====================================================================
# SISTEMA DE DATOS REALES
# ====================================================================

class RealDataCollector:
    """Recolector de datos reales de mercado"""
    
    def __init__(self, symbol='EURUSD=X', period='2y', interval='1h'):
        self.symbol = symbol
        self.period = period
        self.interval = interval
        self.scaler = StandardScaler()
        
    def get_market_data(self):
        """Obtiene datos reales de Yahoo Finance"""
        
        print(f"📊 Descargando datos reales de {self.symbol}...")
        
        try:
            # Descargar datos
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(period=self.period, interval=self.interval)
            
            if data.empty:
                raise ValueError(f"No se pudieron obtener datos para {self.symbol}")
            
            print(f"✅ Datos obtenidos: {len(data)} registros")
            return data
            
        except Exception as e:
            print(f"❌ Error obteniendo datos: {e}")
            print("🔄 Usando datos sintéticos mejorados...")
            return self._generate_improved_synthetic_data()
    
    def _generate_improved_synthetic_data(self):
        """Genera datos sintéticos con patrones más realistas"""
        
        print("🔄 Generando datos sintéticos mejorados...")
        
        # Simular datos de mercado más realistas
        n_samples = 5000
        dates = pd.date_range(start='2022-01-01', periods=n_samples, freq='1H')
        
        # Precios base con tendencia
        base_price = 1.1000
        trend = np.cumsum(np.random.randn(n_samples) * 0.0001)
        prices = base_price + trend
        
        # Volatilidad variable
        volatility = 0.0005 + 0.0003 * np.sin(np.arange(n_samples) * 0.01)
        returns = np.random.randn(n_samples) * volatility
        
        # Aplicar retornos
        prices = prices * (1 + returns)
        
        # Crear DataFrame
        data = pd.DataFrame({
            'Open': prices * (1 + np.random.randn(n_samples) * 0.0001),
            'High': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.0005),
            'Low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.0005),
            'Close': prices,
            'Volume': np.random.randint(1000, 10000, n_samples)
        }, index=dates)
        
        return data
    
    def prepare_features(self, data, sequence_length=30):
        """Prepara features técnicos reales"""
        
        print("🔧 Preparando features técnicos...")
        
        # Features técnicos básicos
        df = data.copy()
        
        # Returns
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['ma_5'] = df['Close'].rolling(5).mean()
        df['ma_20'] = df['Close'].rolling(20).mean()
        df['ma_50'] = df['Close'].rolling(50).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        
        # Volume features
        df['volume_ma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_ma']
        
        # Price position features
        df['price_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Momentum features
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Clean NaN values
        df = df.dropna()
        
        # Select features
        feature_columns = [
            'returns', 'log_returns', 'rsi', 'macd', 'macd_signal',
            'volatility', 'volume_ratio', 'price_position',
            'momentum_5', 'momentum_10', 'momentum_20'
        ]
        
        features = df[feature_columns].values
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        for i in range(sequence_length, len(features_scaled) - 1):
            seq = features_scaled[i-sequence_length:i]
            
            # Target: next period return direction
            future_return = df['returns'].iloc[i]
            
            # Define targets based on return magnitude
            if future_return > 0.0005:  # Strong up
                target = 0
            elif future_return < -0.0005:  # Strong down
                target = 2
            else:  # Sideways
                target = 1
            
            sequences.append(seq)
            targets.append(target)
        
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        print(f"✅ Features preparados:")
        print(f"   • Secuencias: {len(sequences)}")
        print(f"   • Features: {features_scaled.shape[1]}")
        print(f"   • Distribución targets: {np.bincount(targets)}")
        
        return sequences, targets

# ====================================================================
# SISTEMA DE ENTRENAMIENTO MEJORADO
# ====================================================================

class ImprovedTrainer:
    """Entrenador mejorado con técnicas anti-overfitting"""
    
    def __init__(self, model, train_loader, val_loader, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizaciones CPU
        torch.set_num_threads(mp.cpu_count())
        torch.set_flush_denormal(True)
        
    def train_epoch(self, optimizer, criterion, scheduler=None):
        """Entrena una época con mejoras"""
        
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            data, targets = data.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            
            loss = criterion(outputs['direction'], targets)
            loss.backward()
            
            # Gradient clipping más agresivo
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
            
            # Accuracy calculation
            _, predicted = torch.max(outputs['direction'].data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Memory cleanup
            if batch_idx % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self, criterion):
        """Validación mejorada"""
        
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs['direction'], targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs['direction'].data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        return val_loss / len(self.val_loader), correct / total

# ====================================================================
# SISTEMA ENSEMBLE MEJORADO
# ====================================================================

class ImprovedEnsembleSystem:
    """Sistema ensemble mejorado"""
    
    def __init__(self, input_dim, sequence_length):
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.models = {}
        self.weights = {}
        
        # Crear modelos mejorados
        self._build_improved_ensemble()
    
    def _build_improved_ensemble(self):
        """Construye ensemble de modelos mejorados"""
        
        # Modelo 1: Transformer mejorado
        self.models['improved_transformer'] = ImprovedTransformer(
            input_dim=self.input_dim,
            d_model=128,
            num_heads=4,
            num_layers=3
        )
        
        # Modelo 2: LSTM mejorado
        self.models['improved_lstm'] = ImprovedLSTM(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=3
        )
        
        # Modelo 3: Linear ensemble mejorado
        flattened_dim = self.input_dim * min(self.sequence_length, 30)
        self.models['improved_linear'] = ImprovedLinearEnsemble(
            input_dim=flattened_dim,
            hidden_dims=[256, 128, 64]
        )
        
        # Pesos iniciales uniformes
        num_models = len(self.models)
        for model_name in self.models:
            self.weights[model_name] = 1.0 / num_models
    
    def train_ensemble(self, train_data, val_data, epochs=50):
        """Entrena todos los modelos del ensemble con mejoras"""
        
        print(f"🚀 Entrenando ensemble mejorado de {len(self.models)} modelos...")
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n🔄 Entrenando {model_name}...")
            
            # Preparar datos según el modelo
            if 'linear' in model_name:
                X_train, y_train = self._prepare_linear_data(train_data)
                X_val, y_val = self._prepare_linear_data(val_data)
            else:
                X_train, y_train = self._prepare_sequence_data(train_data)
                X_val, y_val = self._prepare_sequence_data(val_data)
            
            # DataLoaders con batch size más pequeño
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
            
            # Trainer mejorado
            trainer = ImprovedTrainer(model, train_loader, val_loader)
            
            # Optimizer con weight decay
            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
            criterion = nn.CrossEntropyLoss()
            
            # Scheduler con warmup
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_loader)
            )
            
            # Training loop mejorado
            best_val_acc = 0
            patience_counter = 0
            train_history = []
            val_history = []
            
            for epoch in range(epochs):
                train_loss, train_acc = trainer.train_epoch(optimizer, criterion, scheduler)
                val_loss, val_acc = trainer.validate(criterion)
                
                train_history.append(train_acc)
                val_history.append(val_acc)
                
                if epoch % 5 == 0:
                    print(f"  Época {epoch:2d} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")
                
                # Early stopping con más paciencia
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 15:  # Más paciencia
                        print(f"  ⏰ Early stopping en época {epoch}")
                        break
            
            results[model_name] = {
                'best_val_acc': best_val_acc,
                'final_train_acc': train_acc,
                'final_val_acc': val_acc,
                'train_history': train_history,
                'val_history': val_history
            }
            
            print(f"  ✅ {model_name} completado - Val Acc: {best_val_acc:.3f}")
        
        # Actualizar pesos del ensemble
        self._update_ensemble_weights(results)
        
        return results
    
    def _prepare_sequence_data(self, data):
        """Prepara datos para modelos secuenciales"""
        X = torch.FloatTensor(data['sequences'])
        y = torch.LongTensor(data['targets'])
        return X, y
    
    def _prepare_linear_data(self, data):
        """Prepara datos para modelos lineales"""
        sequences = data['sequences']
        # Flatten sequences but limit length
        max_len = min(self.sequence_length, 30)
        if sequences.shape[1] > max_len:
            sequences = sequences[:, -max_len:, :]
        
        X = torch.FloatTensor(sequences.reshape(sequences.shape[0], -1))
        y = torch.LongTensor(data['targets'])
        return X, y
    
    def _update_ensemble_weights(self, results):
        """Actualiza pesos del ensemble basado en performance"""
        
        # Calculate weights based on validation accuracy
        total_performance = sum(r['best_val_acc'] for r in results.values())
        
        if total_performance > 0:
            for model_name, result in results.items():
                self.weights[model_name] = result['best_val_acc'] / total_performance
        
        print(f"\n⚖️ Pesos del ensemble actualizados:")
        for model_name, weight in self.weights.items():
            print(f"  {model_name}: {weight:.3f}")
    
    def predict(self, x):
        """Predicción ensemble mejorada"""
        
        ensemble_predictions = []
        ensemble_confidences = []
        
        for model_name, model in self.models.items():
            model.eval()
            
            with torch.no_grad():
                # Preparar input según el modelo
                if 'linear' in model_name:
                    max_len = min(self.sequence_length, 30)
                    if x.shape[1] > max_len:
                        x_input = x[:, -max_len:, :].reshape(x.shape[0], -1)
                    else:
                        x_input = x.reshape(x.shape[0], -1)
                else:
                    x_input = x
                
                output = model(x_input)
                
                # Weighted prediction
                weight = self.weights[model_name]
                pred_probs = torch.softmax(output['direction'], dim=-1)
                
                ensemble_predictions.append(pred_probs * weight)
                ensemble_confidences.append(output['confidence'] * weight)
        
        # Combine predictions
        final_prediction = torch.sum(torch.stack(ensemble_predictions), dim=0)
        final_confidence = torch.sum(torch.stack(ensemble_confidences), dim=0)
        
        return {
            'direction': final_prediction,
            'confidence': final_confidence
        }

# ====================================================================
# BRAIN PRINCIPAL MEJORADO
# ====================================================================

class ImprovedTradingBrain:
    """Trading Brain mejorado con datos reales"""
    
    def __init__(self, symbol='EURUSD=X', style='DAY_TRADING'):
        self.symbol = symbol
        self.style = style
        
        # Configuración mejorada
        self.config = self._get_improved_config()
        
        # Componentes
        self.data_collector = RealDataCollector(symbol)
        self.ensemble_system = None
        self.performance_tracker = {
            'training_history': [],
            'ensemble_performance': {},
            'final_metrics': {}
        }
        
        print(f"🖥️ Improved Trading Brain inicializado")
        print(f"   • Par: {symbol}")
        print(f"   • Estilo: {style}")
        print(f"   • CPU Cores: {mp.cpu_count()}")
        print(f"   • Configuración: Mejorada con datos reales")
    
    def _get_improved_config(self):
        """Configuración mejorada"""
        
        base_configs = {
            'SCALPING': {
                'sequence_length': 20,
                'batch_size': 16,
                'epochs': 40,
                'learning_rate': 1e-3
            },
            'DAY_TRADING': {
                'sequence_length': 30,
                'batch_size': 16,
                'epochs': 50,
                'learning_rate': 5e-4
            },
            'SWING_TRADING': {
                'sequence_length': 45,
                'batch_size': 16,
                'epochs': 60,
                'learning_rate': 3e-4
            },
            'POSITION_TRADING': {
                'sequence_length': 60,
                'batch_size': 12,
                'epochs': 70,
                'learning_rate': 2e-4
            }
        }
        
        return base_configs.get(self.style, base_configs['DAY_TRADING'])
    
    def prepare_data(self):
        """Preparación de datos reales"""
        
        print("📊 Preparando datos reales...")
        
        # Obtener datos reales
        market_data = self.data_collector.get_market_data()
        
        # Preparar features
        sequences, targets = self.data_collector.prepare_features(
            market_data, 
            sequence_length=self.config['sequence_length']
        )
        
        # Train/val/test split
        train_idx = int(len(sequences) * 0.7)
        val_idx = int(len(sequences) * 0.85)
        
        self.train_data = {
            'sequences': sequences[:train_idx],
            'targets': targets[:train_idx]
        }
        
        self.val_data = {
            'sequences': sequences[train_idx:val_idx],
            'targets': targets[train_idx:val_idx]
        }
        
        self.test_data = {
            'sequences': sequences[val_idx:],
            'targets': targets[val_idx:]
        }
        
        print(f"✅ Datos preparados:")
        print(f"   • Train: {len(self.train_data['sequences'])} secuencias")
        print(f"   • Val: {len(self.val_data['sequences'])} secuencias")
        print(f"   • Test: {len(self.test_data['sequences'])} secuencias")
        print(f"   • Features: {sequences.shape[2]}")
        print(f"   • Sequence length: {self.config['sequence_length']}")
    
    def build_and_train_system(self):
        """Construye y entrena el sistema completo"""
        
        print("\n🏗️ Construyendo sistema ensemble mejorado...")
        
        # Crear ensemble system
        n_features = self.train_data['sequences'].shape[2]
        self.ensemble_system = ImprovedEnsembleSystem(
            input_dim=n_features,
            sequence_length=self.config['sequence_length']
        )
        
        # Entrenar ensemble
        results = self.ensemble_system.train_ensemble(
            self.train_data,
            self.val_data,
            epochs=self.config['epochs']
        )
        
        self.performance_tracker['ensemble_performance'] = results
        
        return results
    
    def evaluate_system(self):
        """Evaluación completa del sistema"""
        
        print("\n📊 Evaluando sistema...")
        
        # Usar datos de test
        test_sequences = torch.FloatTensor(self.test_data['sequences'])
        test_targets = self.test_data['targets']
        
        # Predicciones del ensemble
        with torch.no_grad():
            ensemble_output = self.ensemble_system.predict(test_sequences)
            
            # Get predictions
            _, predicted = torch.max(ensemble_output['direction'], 1)
            confidence = ensemble_output['confidence'].numpy().flatten()
            
            # Calculate metrics
            accuracy = (predicted.numpy() == test_targets).mean()
            
            # Confidence-based metrics
            high_conf_mask = confidence > 0.7
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = (predicted.numpy()[high_conf_mask] == test_targets[high_conf_mask]).mean()
            else:
                high_conf_accuracy = 0
            
            # Per-class accuracy
            class_accuracies = []
            for i in range(3):
                class_mask = test_targets == i
                if class_mask.sum() > 0:
                    class_acc = (predicted.numpy()[class_mask] == test_targets[class_mask]).mean()
                    class_accuracies.append(class_acc)
                else:
                    class_accuracies.append(0)
        
        metrics = {
            'overall_accuracy': accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'avg_confidence': confidence.mean(),
            'high_conf_predictions': high_conf_mask.sum(),
            'total_predictions': len(confidence),
            'class_accuracies': class_accuracies
        }
        
        self.performance_tracker['final_metrics'] = metrics
        
        print(f"📈 Resultados finales:")
        print(f"   • Precisión general: {accuracy:.3f}")
        print(f"   • Precisión alta confianza: {high_conf_accuracy:.3f}")
        print(f"   • Confianza promedio: {confidence.mean():.3f}")
        print(f"   • Predicciones alta confianza: {high_conf_mask.sum()}/{len(confidence)}")
        print(f"   • Precisión por clase: {class_accuracies}")
        
        return metrics
    
    def run_complete_training(self):
        """Ejecuta entrenamiento completo mejorado"""
        
        print("🚀 INICIANDO ENTRENAMIENTO MEJORADO")
        print("=" * 50)
        
        start_time = pd.Timestamp.now()
        
        try:
            # Paso 1: Preparar datos reales
            self.prepare_data()
            
            # Paso 2: Construir y entrenar
            ensemble_results = self.build_and_train_system()
            
            # Paso 3: Evaluar
            final_metrics = self.evaluate_system()
            
            end_time = pd.Timestamp.now()
            training_time = end_time - start_time
            
            print(f"\n🎉 ENTRENAMIENTO COMPLETADO!")
            print(f"⏱️ Tiempo total: {training_time}")
            print(f"📊 Precisión final: {final_metrics['overall_accuracy']:.3f}")
            
            return {
                'ensemble_results': ensemble_results,
                'final_metrics': final_metrics,
                'training_time': training_time,
                'success': True
            }
            
        except Exception as e:
            print(f"❌ Error durante entrenamiento: {e}")
            return {
                'error': str(e),
                'success': False
            }

# ====================================================================
# FUNCIONES DE UTILIDAD
# ====================================================================

def improved_demo(symbol='EURUSD=X', style='DAY_TRADING'):
    """Demo mejorado con datos reales"""
    
    print("⚡ DEMO MEJORADO CON DATOS REALES")
    print("-" * 40)
    
    brain = ImprovedTradingBrain(symbol, style)
    results = brain.run_complete_training()
    
    if results['success']:
        print(f"\n✅ Demo completado exitosamente!")
        print(f"📊 Tiempo: {results['training_time']}")
        print(f"🎯 Precisión: {results['final_metrics']['overall_accuracy']:.3f}")
    else:
        print(f"\n❌ Demo falló: {results['error']}")
    
    return results

def main():
    """Función principal para sistema mejorado"""
    
    print("🖥️ SISTEMA DE TRADING MEJORADO")
    print("=" * 40)
    print("🎯 Datos reales + Arquitectura mejorada")
    print("⚡ Ensemble con regularización")
    print("🧠 Técnicas anti-overfitting")
    print("=" * 40)
    
    # Configuración
    symbol = 'EURUSD=X'  # Cambiar aquí
    style = 'DAY_TRADING'  # Cambiar aquí
    
    # Crear y entrenar sistema
    brain = ImprovedTradingBrain(symbol, style)
    results = brain.run_complete_training()
    
    if results['success']:
        print(f"\n🏆 SISTEMA ENTRENADO EXITOSAMENTE!")
        print(f"🎯 Listo para trading en {symbol}")
    else:
        print(f"\n❌ Error en entrenamiento: {results['error']}")
    
    return results

if __name__ == "__main__":
    # Ejecutar sistema principal
    results = main()
    
    print("\n✨ IMPROVED TRADING BRAIN COMPLETADO") 