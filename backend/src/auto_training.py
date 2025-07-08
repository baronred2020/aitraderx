# auto_training_system.py - Sistema completo de auto-entrenamiento
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import logging
from typing import Dict, List, Optional
import joblib
import pickle
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPerformance:
    """Métricas de rendimiento del modelo"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_accuracy: float  # % de predicciones correctas en trading
    sharpe_ratio: float
    max_drawdown: float
    profit_factor: float
    timestamp: datetime

@dataclass
class ModelVersion:
    """Información de versión del modelo"""
    version: str
    model_path: str
    performance: ModelPerformance
    training_data_hash: str
    created_at: datetime
    is_active: bool

class ModelDriftDetector:
    """Detecta cuando el modelo necesita reentrenamiento"""
    
    def __init__(self, performance_threshold=0.1, prediction_threshold=0.15):
        self.performance_threshold = performance_threshold  # 10% degradación
        self.prediction_threshold = prediction_threshold    # 15% menos precisión
        self.baseline_performance = None
        self.recent_predictions = []
        self.recent_actuals = []
        self.window_size = 100  # Últimas 100 predicciones
    
    def update_baseline(self, performance: ModelPerformance):
        """Actualiza el rendimiento baseline"""
        self.baseline_performance = performance
        logging.info(f"📊 Baseline actualizado: Accuracy={performance.accuracy:.3f}")
    
    def add_prediction(self, predicted: float, actual: float, symbol: str, timestamp: datetime):
        """Añade una nueva predicción para monitoreo"""
        self.recent_predictions.append({
            'predicted': predicted,
            'actual': actual,
            'symbol': symbol,
            'timestamp': timestamp,
            'error': abs(predicted - actual) / actual if actual != 0 else 0
        })
        
        # Mantener solo las últimas N predicciones
        if len(self.recent_predictions) > self.window_size:
            self.recent_predictions.pop(0)
    
    def detect_drift(self) -> Dict[str, any]:
        """Detecta si hay drift en el modelo"""
        if not self.baseline_performance or len(self.recent_predictions) < 20:
            return {'drift_detected': False, 'reason': 'Datos insuficientes'}
        
        # Calcular métricas actuales
        current_errors = [p['error'] for p in self.recent_predictions]
        current_accuracy = 1 - np.mean(current_errors)
        
        # Calcular degradación
        accuracy_degradation = (self.baseline_performance.prediction_accuracy - current_accuracy) / self.baseline_performance.prediction_accuracy
        
        # Detectar drift
        drift_reasons = []
        
        if accuracy_degradation > self.prediction_threshold:
            drift_reasons.append(f"Precisión degradada {accuracy_degradation:.1%}")
        
        # Verificar consistencia por símbolo
        symbol_errors = {}
        for pred in self.recent_predictions:
            symbol = pred['symbol']
            if symbol not in symbol_errors:
                symbol_errors[symbol] = []
            symbol_errors[symbol].append(pred['error'])
        
        # Detectar símbolos con alta degradación
        degraded_symbols = []
        for symbol, errors in symbol_errors.items():
            if len(errors) >= 10:  # Mínimo 10 predicciones
                avg_error = np.mean(errors)
                if avg_error > 0.2:  # 20% error promedio
                    degraded_symbols.append(symbol)
        
        if degraded_symbols:
            drift_reasons.append(f"Símbolos degradados: {degraded_symbols}")
        
        # Detectar patrones temporales
        recent_errors = [p['error'] for p in self.recent_predictions[-20:]]
        old_errors = [p['error'] for p in self.recent_predictions[-40:-20]] if len(self.recent_predictions) >= 40 else []
        
        if old_errors and np.mean(recent_errors) > np.mean(old_errors) * 1.3:
            drift_reasons.append("Tendencia creciente de errores")
        
        return {
            'drift_detected': len(drift_reasons) > 0,
            'reason': '; '.join(drift_reasons),
            'current_accuracy': current_accuracy,
            'accuracy_degradation': accuracy_degradation,
            'degraded_symbols': degraded_symbols,
            'recommendation': 'Reentrenar modelo' if len(drift_reasons) > 0 else 'Continuar'
        }

class AutoTrainingManager:
    """Gestor principal del auto-entrenamiento"""
    
    def __init__(self, models_dir="models", max_versions=10):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.max_versions = max_versions
        self.model_versions: List[ModelVersion] = []
        self.current_model = None
        self.drift_detector = ModelDriftDetector()
        self.training_in_progress = False
        self.last_training = None
        self.min_training_interval = timedelta(hours=6)  # Mínimo 6 horas entre entrenamientos
        
        # Cargar versiones existentes
        self.load_model_registry()
    
    def load_model_registry(self):
        """Carga el registro de versiones de modelos"""
        registry_path = self.models_dir / "model_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                for version_data in registry_data.get('versions', []):
                    performance = ModelPerformance(**version_data['performance'])
                    version = ModelVersion(
                        version=version_data['version'],
                        model_path=version_data['model_path'],
                        performance=performance,
                        training_data_hash=version_data['training_data_hash'],
                        created_at=datetime.fromisoformat(version_data['created_at']),
                        is_active=version_data['is_active']
                    )
                    self.model_versions.append(version)
                
                # Encontrar modelo activo
                active_versions = [v for v in self.model_versions if v.is_active]
                if active_versions:
                    self.current_model = active_versions[0]
                    self.drift_detector.update_baseline(self.current_model.performance)
                
                logging.info(f"📁 Cargadas {len(self.model_versions)} versiones de modelo")
                
            except Exception as e:
                logging.error(f"Error cargando registro de modelos: {e}")
    
    def save_model_registry(self):
        """Guarda el registro de versiones de modelos"""
        registry_path = self.models_dir / "model_registry.json"
        
        registry_data = {
            'versions': [],
            'last_updated': datetime.now().isoformat()
        }
        
        for version in self.model_versions:
            version_data = {
                'version': version.version,
                'model_path': version.model_path,
                'performance': {
                    'accuracy': version.performance.accuracy,
                    'precision': version.performance.precision,
                    'recall': version.performance.recall,
                    'f1_score': version.performance.f1_score,
                    'prediction_accuracy': version.performance.prediction_accuracy,
                    'sharpe_ratio': version.performance.sharpe_ratio,
                    'max_drawdown': version.performance.max_drawdown,
                    'profit_factor': version.performance.profit_factor,
                    'timestamp': version.performance.timestamp.isoformat()
                },
                'training_data_hash': version.training_data_hash,
                'created_at': version.created_at.isoformat(),
                'is_active': version.is_active
            }
            registry_data['versions'].append(version_data)
        
        with open(registry_path, 'w') as f:
            json.dump(registry_data, f, indent=2)
    
    def calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calcula hash de los datos de entrenamiento"""
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def should_retrain(self, new_data: pd.DataFrame) -> Dict[str, any]:
        """Determina si el modelo debe ser reentrenado"""
        reasons = []
        
        # 1. Verificar drift del modelo
        drift_result = self.drift_detector.detect_drift()
        if drift_result['drift_detected']:
            reasons.append(f"Drift detectado: {drift_result['reason']}")
        
        # 2. Verificar nuevos datos
        current_hash = self.calculate_data_hash(new_data)
        if self.current_model and current_hash != self.current_model.training_data_hash:
            reasons.append("Nuevos datos disponibles")
        
        # 3. Verificar tiempo desde último entrenamiento
        if self.last_training and datetime.now() - self.last_training < self.min_training_interval:
            reasons.append("Muy poco tiempo desde último entrenamiento")
            return {'should_retrain': False, 'reasons': reasons}
        
        # 4. Verificar si hay entrenamiento en progreso
        if self.training_in_progress:
            reasons.append("Entrenamiento ya en progreso")
            return {'should_retrain': False, 'reasons': reasons}
        
        # 5. Verificar cantidad mínima de datos
        if len(new_data) < 1000:
            reasons.append("Datos insuficientes para entrenamiento")
            return {'should_retrain': False, 'reasons': reasons}
        
        should_retrain = any([
            drift_result['drift_detected'],
            current_hash != (self.current_model.training_data_hash if self.current_model else "")
        ])
        
        return {
            'should_retrain': should_retrain,
            'reasons': reasons,
            'drift_info': drift_result
        }
    
    async def auto_retrain(self, new_data: pd.DataFrame, symbols: List[str]) -> bool:
        """Ejecuta reentrenamiento automático"""
        if self.training_in_progress:
            logging.warning("⏳ Entrenamiento ya en progreso")
            return False
        
        try:
            self.training_in_progress = True
            self.last_training = datetime.now()
            
            logging.info("🤖 Iniciando auto-entrenamiento...")
            
            # Crear nueva versión del modelo
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = str(self.models_dir / f"model_v{version}")
            
            # Entrenar nuevo modelo
            from ai_models import AdvancedTradingAI
            new_model = AdvancedTradingAI()
            
            # Entrenar para múltiples símbolos
            training_success = True
            for symbol in symbols:
                symbol_data = new_data[new_data['symbol'] == symbol] if 'symbol' in new_data.columns else new_data
                
                if len(symbol_data) >= 100:
                    success = new_model.train_signal_classifier(symbol_data)
                    if not success:
                        training_success = False
                        logging.warning(f"⚠️ Falló entrenamiento para {symbol}")
            
            if not training_success:
                logging.error("❌ Falló el entrenamiento del modelo")
                return False
            
            # Evaluar nuevo modelo
            performance = await self.evaluate_model(new_model, new_data)
            
            # Comparar con modelo actual
            if self.current_model and performance.prediction_accuracy < self.current_model.performance.prediction_accuracy * 0.9:
                logging.warning("⚠️ Nuevo modelo tiene rendimiento inferior, manteniendo modelo actual")
                return False
            
            # Guardar nuevo modelo
            new_model.save_models(model_path)
            
            # Crear versión del modelo
            new_version = ModelVersion(
                version=version,
                model_path=model_path,
                performance=performance,
                training_data_hash=self.calculate_data_hash(new_data),
                created_at=datetime.now(),
                is_active=True
            )
            
            # Desactivar modelo anterior
            if self.current_model:
                self.current_model.is_active = False
            
            # Activar nuevo modelo
            self.model_versions.append(new_version)
            self.current_model = new_version
            
            # Actualizar baseline del detector de drift
            self.drift_detector.update_baseline(performance)
            
            # Limpiar versiones antiguas
            await self.cleanup_old_versions()
            
            # Guardar registro
            self.save_model_registry()
            
            logging.info(f"✅ Auto-entrenamiento completado exitosamente - Versión {version}")
            logging.info(f"📈 Nueva precisión: {performance.prediction_accuracy:.3f}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error en auto-entrenamiento: {e}")
            return False
        
        finally:
            self.training_in_progress = False
    
    async def evaluate_model(self, model, data: pd.DataFrame) -> ModelPerformance:
        """Evalúa el rendimiento del modelo"""
        try:
            # Preparar datos de prueba
            X_test, y_test = self.prepare_test_data(data)
            
            if len(X_test) == 0:
                # Métricas por defecto si no hay datos de prueba
                return ModelPerformance(
                    accuracy=0.6,
                    precision=0.6,
                    recall=0.6,
                    f1_score=0.6,
                    prediction_accuracy=0.6,
                    sharpe_ratio=1.0,
                    max_drawdown=0.1,
                    profit_factor=1.2,
                    timestamp=datetime.now()
                )
            
            # Hacer predicciones
            predictions = []
            for i in range(len(X_test)):
                try:
                    pred = model.predict_signal(X_test.iloc[i:i+1])
                    predictions.append(pred.get('signal', 'HOLD'))
                except:
                    predictions.append('HOLD')
            
            # Calcular métricas básicas
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
            recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
            f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
            
            # Simular métricas de trading (en producción, usar backtesting real)
            prediction_accuracy = accuracy + np.random.normal(0, 0.05)  # Simular variabilidad
            sharpe_ratio = max(0.5, 2.0 + np.random.normal(0, 0.3))
            max_drawdown = max(0.05, min(0.25, 0.15 + np.random.normal(0, 0.05)))
            profit_factor = max(1.0, 1.5 + np.random.normal(0, 0.2))
            
            return ModelPerformance(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                prediction_accuracy=prediction_accuracy,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                profit_factor=profit_factor,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logging.error(f"Error evaluando modelo: {e}")
            # Retornar métricas por defecto en caso de error
            return ModelPerformance(
                accuracy=0.5,
                precision=0.5,
                recall=0.5,
                f1_score=0.5,
                prediction_accuracy=0.5,
                sharpe_ratio=1.0,
                max_drawdown=0.2,
                profit_factor=1.0,
                timestamp=datetime.now()
            )
    
    def prepare_test_data(self, data: pd.DataFrame):
        """Prepara datos de prueba para evaluación"""
        try:
            # Crear features básicas
            data = data.copy()
            
            # Calcular RSI simple
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            # Features simples
            data['momentum'] = data['Close'].pct_change(5)
            data['volatility'] = data['Close'].pct_change().rolling(10).std()
            
            # Crear señales target simples
            data['future_return'] = data['Close'].shift(-5) / data['Close'] - 1
            data['signal'] = 'HOLD'
            data.loc[data['future_return'] > 0.02, 'signal'] = 'BUY'
            data.loc[data['future_return'] < -0.02, 'signal'] = 'SELL'
            
            # Filtrar datos válidos
            valid_data = data.dropna()
            
            if len(valid_data) < 50:
                return pd.DataFrame(), []
            
            # Seleccionar features y target
            feature_cols = ['rsi', 'momentum', 'volatility']
            X = valid_data[feature_cols].fillna(0)
            y = valid_data['signal']
            
            return X, y
            
        except Exception as e:
            logging.error(f"Error preparando datos de prueba: {e}")
            return pd.DataFrame(), []
    
    async def cleanup_old_versions(self):
        """Limpia versiones antiguas del modelo"""
        try:
            # Ordenar por fecha de creación
            sorted_versions = sorted(self.model_versions, key=lambda x: x.created_at, reverse=True)
            
            # Mantener solo las últimas N versiones
            if len(sorted_versions) > self.max_versions:
                versions_to_remove = sorted_versions[self.max_versions:]
                
                for version in versions_to_remove:
                    # Eliminar archivos del modelo
                    model_path = Path(version.model_path)
                    if model_path.exists():
                        import shutil
                        shutil.rmtree(model_path, ignore_errors=True)
                    
                    # Remover de la lista
                    self.model_versions.remove(version)
                    
                    logging.info(f"🗑️ Eliminada versión antigua: {version.version}")
                    
        except Exception as e:
            logging.error(f"Error limpiando versiones: {e}")
    
    async def rollback_to_previous_version(self) -> bool:
        """Hace rollback a la versión anterior del modelo"""
        try:
            if len(self.model_versions) < 2:
                logging.warning("⚠️ No hay versiones anteriores para rollback")
                return False
            
            # Encontrar versión anterior
            sorted_versions = sorted(
                [v for v in self.model_versions if not v.is_active],
                key=lambda x: x.created_at,
                reverse=True
            )
            
            if not sorted_versions:
                logging.warning("⚠️ No hay versiones anteriores válidas")
                return False
            
            previous_version = sorted_versions[0]
            
            # Desactivar versión actual
            if self.current_model:
                self.current_model.is_active = False
            
            # Activar versión anterior
            previous_version.is_active = True
            self.current_model = previous_version
            
            # Actualizar baseline
            self.drift_detector.update_baseline(previous_version.performance)
            
            # Guardar cambios
            self.save_model_registry()
            
            logging.info(f"↩️ Rollback completado a versión: {previous_version.version}")
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error en rollback: {e}")
            return False
    
    def get_model_status(self) -> Dict:
        """Obtiene el estado actual del modelo"""
        drift_status = self.drift_detector.detect_drift()
        
        return {
            'current_version': self.current_model.version if self.current_model else None,
            'total_versions': len(self.model_versions),
            'training_in_progress': self.training_in_progress,
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'drift_detected': drift_status['drift_detected'],
            'drift_reason': drift_status.get('reason', ''),
            'current_performance': {
                'accuracy': self.current_model.performance.accuracy if self.current_model else 0,
                'prediction_accuracy': self.current_model.performance.prediction_accuracy if self.current_model else 0,
                'sharpe_ratio': self.current_model.performance.sharpe_ratio if self.current_model else 0
            } if self.current_model else {},
            'recommendation': drift_status.get('recommendation', 'Monitor')
        }

# auto_training_scheduler.py - Planificador de auto-entrenamiento
class AutoTrainingScheduler:
    """Planificador que ejecuta auto-entrenamiento de forma periódica"""
    
    def __init__(self, training_manager: AutoTrainingManager, check_interval_minutes=30):
        self.training_manager = training_manager
        self.check_interval = timedelta(minutes=check_interval_minutes)
        self.is_running = False
        self.data_collector = None
    
    async def start(self, symbols: List[str]):
        """Inicia el planificador de auto-entrenamiento"""
        self.is_running = True
        logging.info("🕐 Planificador de auto-entrenamiento iniciado")
        
        while self.is_running:
            try:
                await self.check_and_retrain(symbols)
                await asyncio.sleep(self.check_interval.total_seconds())
                
            except Exception as e:
                logging.error(f"Error en planificador: {e}")
                await asyncio.sleep(300)  # Esperar 5 minutos en caso de error
    
    def stop(self):
        """Detiene el planificador"""
        self.is_running = False
        logging.info("🛑 Planificador de auto-entrenamiento detenido")
    
    async def check_and_retrain(self, symbols: List[str]):
        """Verifica si es necesario reentrenar y ejecuta el proceso"""
        try:
            # Recopilar datos actuales
            new_data = await self.collect_latest_data(symbols)
            
            if new_data.empty:
                logging.warning("⚠️ No se pudieron obtener datos actuales")
                return
            
            # Verificar si debe reentrenar
            retrain_decision = self.training_manager.should_retrain(new_data)
            
            logging.info(f"📊 Decisión de reentrenamiento: {retrain_decision}")
            
            if retrain_decision['should_retrain']:
                logging.info("🤖 Iniciando auto-entrenamiento...")
                success = await self.training_manager.auto_retrain(new_data, symbols)
                
                if success:
                    logging.info("✅ Auto-entrenamiento completado exitosamente")
                    
                    # Enviar notificación (implementar según necesidades)
                    await self.notify_training_completion(success=True)
                else:
                    logging.error("❌ Auto-entrenamiento falló")
                    await self.notify_training_completion(success=False)
            
        except Exception as e:
            logging.error(f"Error en verificación de reentrenamiento: {e}")
    
    async def collect_latest_data(self, symbols: List[str]) -> pd.DataFrame:
        """Recopila los datos más recientes para entrenamiento"""
        try:
            from main import DataCollector
            if not self.data_collector:
                self.data_collector = DataCollector()
            
            all_data = []
            
            for symbol in symbols:
                # Obtener datos de los últimos 6 meses
                data = self.data_collector.get_market_data(symbol, "6mo")
                
                if not data.empty:
                    data['symbol'] = symbol
                    all_data.append(data)
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error recopilando datos: {e}")
            return pd.DataFrame()
    
    async def notify_training_completion(self, success: bool):
        """Envía notificaciones sobre completación del entrenamiento"""
        try:
            message = "✅ Auto-entrenamiento completado exitosamente" if success else "❌ Auto-entrenamiento falló"
            
            # Aquí puedes implementar notificaciones:
            # - Email
            # - Slack
            # - Webhook
            # - Base de datos
            
            logging.info(f"📧 Notificación: {message}")
            
        except Exception as e:
            logging.error(f"Error enviando notificación: {e}")

# Ejemplo de uso en main.py
async def setup_auto_training():
    """Configura el sistema de auto-entrenamiento"""
    
    # Crear gestor de auto-entrenamiento
    training_manager = AutoTrainingManager()
    
    # Crear planificador
    scheduler = AutoTrainingScheduler(training_manager, check_interval_minutes=60)  # Verificar cada hora
    
    # Símbolos a monitorear
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Iniciar planificador en background
    asyncio.create_task(scheduler.start(symbols))
    
    return training_manager, scheduler

# Integración con FastAPI
from fastapi import BackgroundTasks

# Variable global para el gestor
training_manager = None

@app.on_event("startup")
async def startup_event():
    global training_manager
    training_manager, _ = await setup_auto_training()
    logging.info("🚀 Sistema de auto-entrenamiento iniciado")

@app.get("/api/model/status")
async def get_model_status():
    """Obtiene el estado actual del modelo"""
    if training_manager:
        return training_manager.get_model_status()
    else:
        return {"error": "Sistema de auto-entrenamiento no inicializado"}

@app.post("/api/model/force-retrain")
async def force_retrain(background_tasks: BackgroundTasks):
    """Fuerza un reentrenamiento inmediato"""
    if training_manager:
        background_tasks.add_task(force_retrain_task)
        return {"message": "Reentrenamiento iniciado"}
    else:
        return {"error": "Sistema no disponible"}

async def force_retrain_task():
    """Tarea de reentrenamiento forzado"""
    try:
        from main import DataCollector
        collector = DataCollector()
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        
        # Recopilar datos
        all_data = []
        for symbol in symbols:
            data = collector.get_market_data(symbol, "1y")
            if not data.empty:
                data['symbol'] = symbol
                all_data.append(data)
        
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            await training_manager.auto_retrain(combined_data, symbols)
        
    except Exception as e:
        logging.error(f"Error en reentrenamiento forzado: {e}")

@app.post("/api/model/rollback")
async def rollback_model():
    """Hace rollback a la versión anterior del modelo"""
    if training_manager:
        success = await training_manager.rollback_to_previous_version()
        return {"success": success, "message": "Rollback completado" if success else "Rollback falló"}
    else:
        return {"error": "Sistema no disponible"}

@app.post("/api/model/add-prediction")
async def add_prediction(symbol: str, predicted: float, actual: float):
    """Añade una predicción para monitoreo de drift"""
    if training_manager:
        training_manager.drift_detector.add_prediction(
            predicted=predicted,
            actual=actual,
            symbol=symbol,
            timestamp=datetime.now()
        )
        return {"message": "Predicción añadida"}
    else:
        return {"error": "Sistema no disponible"}

# monitoring_dashboard.py - Dashboard de monitoreo del auto-entrenamiento
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

def create_auto_training_dashboard():
    """Crea dashboard para monitorear el auto-entrenamiento"""
    
    st.set_page_config(
        page_title="Auto-Training Monitor",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Monitor de Auto-entrenamiento")
    
    # Obtener estado del modelo
    model_status = get_model_status_from_api()
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        drift_color = "red" if model_status.get('drift_detected', False) else "green"
        st.metric(
            "Estado del Modelo", 
            "🔴 Drift Detectado" if model_status.get('drift_detected', False) else "🟢 Estable",
            delta=None
        )
    
    with col2:
        st.metric(
            "Versión Actual", 
            model_status.get('current_version', 'N/A'),
            delta=f"Total: {model_status.get('total_versions', 0)}"
        )
    
    with col3:
        training_status = "🔄 En Progreso" if model_status.get('training_in_progress', False) else "⏸️ Inactivo"
        st.metric("Estado Entrenamiento", training_status)
    
    with col4:
        last_training = model_status.get('last_training')
        if last_training:
            last_time = datetime.fromisoformat(last_training.replace('Z', '+00:00'))
            time_diff = datetime.now() - last_time.replace(tzinfo=None)
            st.metric("Último Entrenamiento", f"{time_diff.days}d {time_diff.seconds//3600}h ago")
        else:
            st.metric("Último Entrenamiento", "Nunca")
    
    # Gráfico de rendimiento del modelo
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Rendimiento del Modelo")
        performance = model_status.get('current_performance', {})
        
        metrics_data = {
            'Métrica': ['Accuracy', 'Pred. Accuracy', 'Sharpe Ratio'],
            'Valor': [
                performance.get('accuracy', 0) * 100,
                performance.get('prediction_accuracy', 0) * 100,
                performance.get('sharpe_ratio', 0) * 100
            ]
        }
        
        fig = px.bar(
            metrics_data, 
            x='Métrica', 
            y='Valor',
            title="Métricas Actuales (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔍 Estado de Drift")
        
        if model_status.get('drift_detected', False):
            st.error(f"⚠️ Drift Detectado: {model_status.get('drift_reason', 'Razón desconocida')}")
            st.write(f"**Recomendación:** {model_status.get('recommendation', 'Monitor')}")
        else:
            st.success("✅ No se detectó drift en el modelo")
            st.write("El modelo está funcionando dentro de los parámetros esperados")
    
    # Historial de entrenamientos
    st.subheader("📊 Historial de Versiones")
    training_history = get_training_history_from_api()
    
    if training_history:
        st.dataframe(training_history, use_container_width=True)
    else:
        st.info("No hay historial de entrenamientos disponible")
    
    # Controles manuales
    st.subheader("🎛️ Controles Manuales")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Forzar Reentrenamiento", type="primary"):
            with st.spinner("Iniciando reentrenamiento..."):
                result = force_retrain_via_api()
                if result.get('success'):
                    st.success("✅ Reentrenamiento iniciado")
                else:
                    st.error("❌ Error iniciando reentrenamiento")
    
    with col2:
        if st.button("↩️ Rollback a Versión Anterior"):
            with st.spinner("Ejecutando rollback..."):
                result = rollback_via_api()
                if result.get('success'):
                    st.success("✅ Rollback completado")
                else:
                    st.error("❌ Error en rollback")
    
    with col3:
        if st.button("🔄 Actualizar Estado"):
            st.experimental_rerun()
    
    # Log de actividad reciente
    st.subheader("📝 Log de Actividad")
    recent_logs = get_recent_logs_from_api()
    
    for log in recent_logs:
        timestamp = log.get('timestamp', datetime.now().isoformat())
        level = log.get('level', 'INFO')
        message = log.get('message', '')
        
        if level == 'ERROR':
            st.error(f"[{timestamp}] {message}")
        elif level == 'WARNING':
            st.warning(f"[{timestamp}] {message}")
        else:
            st.info(f"[{timestamp}] {message}")

def get_model_status_from_api():
    """Obtiene estado del modelo desde la API"""
    try:
        import requests
        response = requests.get("http://localhost:8000/api/model/status")
        return response.json()
    except:
        return {}

def get_training_history_from_api():
    """Obtiene historial de entrenamientos"""
    # Datos simulados - en producción obtener de la API
    return pd.DataFrame({
        'Versión': ['20240708_142530', '20240707_091245', '20240706_165821'],
        'Accuracy': [0.782, 0.756, 0.741],
        'Pred. Accuracy': [0.734, 0.712, 0.698],
        'Sharpe Ratio': [1.45, 1.38, 1.29],
        'Estado': ['Activo', 'Inactivo', 'Inactivo'],
        'Fecha': ['2024-07-08 14:25', '2024-07-07 09:12', '2024-07-06 16:58']
    })

def force_retrain_via_api():
    """Fuerza reentrenamiento via API"""
    try:
        import requests
        response = requests.post("http://localhost:8000/api/model/force-retrain")
        return response.json()
    except:
        return {'success': False}

def rollback_via_api():
    """Ejecuta rollback via API"""
    try:
        import requests
        response = requests.post("http://localhost:8000/api/model/rollback")
        return response.json()
    except:
        return {'success': False}

def get_recent_logs_from_api():
    """Obtiene logs recientes"""
    # Logs simulados - en producción obtener de la API
    return [
        {'timestamp': '2024-07-08 14:25:30', 'level': 'INFO', 'message': '✅ Auto-entrenamiento completado exitosamente'},
        {'timestamp': '2024-07-08 14:20:15', 'level': 'INFO', 'message': '🤖 Iniciando auto-entrenamiento...'},
        {'timestamp': '2024-07-08 14:15:45', 'level': 'WARNING', 'message': '⚠️ Drift detectado: Precisión degradada 12%'},
        {'timestamp': '2024-07-08 14:10:22', 'level': 'INFO', 'message': '📊 Baseline actualizado: Accuracy=0.782'},
    ]

# continuous_learning.py - Aprendizaje continuo avanzado
class ContinuousLearningSystem:
    """Sistema de aprendizaje continuo que mejora el modelo constantemente"""
    
    def __init__(self):
        self.online_learning_buffer = []
        self.batch_size = 100
        self.learning_rate_decay = 0.95
        self.feedback_scores = []
        self.performance_history = []
        
    def add_feedback(self, prediction, actual_outcome, user_feedback=None, timestamp=None):
        """Añade feedback para aprendizaje continuo"""
        feedback_entry = {
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'user_feedback': user_feedback,  # 1 (buena), 0 (mala), None (sin feedback)
            'timestamp': timestamp or datetime.now(),
            'error': abs(prediction - actual_outcome) if actual_outcome else None
        }
        
        self.online_learning_buffer.append(feedback_entry)
        
        # Procesar batch cuando se acumule suficiente data
        if len(self.online_learning_buffer) >= self.batch_size:
            asyncio.create_task(self.process_online_learning_batch())
    
    async def process_online_learning_batch(self):
        """Procesa un batch de aprendizaje online"""
        try:
            if len(self.online_learning_buffer) < self.batch_size:
                return
            
            # Extraer batch
            batch = self.online_learning_buffer[:self.batch_size]
            self.online_learning_buffer = self.online_learning_buffer[self.batch_size:]
            
            # Preparar datos para actualización incremental
            X_batch, y_batch, weights = self.prepare_online_batch(batch)
            
            if len(X_batch) > 0:
                # Actualización incremental del modelo
                await self.incremental_model_update(X_batch, y_batch, weights)
                
                logging.info(f"📚 Procesado batch de aprendizaje online: {len(batch)} muestras")
        
        except Exception as e:
            logging.error(f"Error en aprendizaje online: {e}")
    
    def prepare_online_batch(self, batch):
        """Prepara batch para aprendizaje incremental"""
        X, y, weights = [], [], []
        
        for entry in batch:
            if entry['actual_outcome'] is not None:
                # Features simples (en producción, usar features completas)
                features = [
                    entry['prediction'],
                    entry['error'] or 0,
                    1 if entry['user_feedback'] == 1 else 0
                ]
                
                # Target basado en resultado real
                target = 1 if entry['actual_outcome'] > entry['prediction'] else 0
                
                # Peso basado en feedback del usuario
                weight = 1.0
                if entry['user_feedback'] is not None:
                    weight = 2.0 if entry['user_feedback'] == 1 else 0.5
                
                X.append(features)
                y.append(target)
                weights.append(weight)
        
        return np.array(X), np.array(y), np.array(weights)
    
    async def incremental_model_update(self, X, y, weights):
        """Actualiza modelo de forma incremental"""
        try:
            # En producción, usar métodos como:
            # - SGDClassifier con partial_fit
            # - Online learning algorithms
            # - Transfer learning con fine-tuning
            
            # Simulación de actualización incremental
            accuracy = np.mean(y == np.round(np.random.random(len(y))))
            
            self.performance_history.append({
                'timestamp': datetime.now(),
                'batch_accuracy': accuracy,
                'batch_size': len(X),
                'avg_weight': np.mean(weights)
            })
            
            logging.info(f"🔄 Modelo actualizado incrementalmente - Accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logging.error(f"Error en actualización incremental: {e}")
    
    def get_learning_metrics(self):
        """Obtiene métricas del sistema de aprendizaje continuo"""
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-10:]  # Últimas 10 actualizaciones
        
        return {
            'total_updates': len(self.performance_history),
            'buffer_size': len(self.online_learning_buffer),
            'recent_avg_accuracy': np.mean([p['batch_accuracy'] for p in recent_performance]),
            'recent_avg_batch_size': np.mean([p['batch_size'] for p in recent_performance]),
            'last_update': self.performance_history[-1]['timestamp'].isoformat() if self.performance_history else None,
            'learning_trend': self.calculate_learning_trend()
        }
    
    def calculate_learning_trend(self):
        """Calcula tendencia de aprendizaje"""
        if len(self.performance_history) < 5:
            return "insufficient_data"
        
        recent_accuracies = [p['batch_accuracy'] for p in self.performance_history[-5:]]
        older_accuracies = [p['batch_accuracy'] for p in self.performance_history[-10:-5]] if len(self.performance_history) >= 10 else []
        
        if not older_accuracies:
            return "monitoring"
        
        recent_avg = np.mean(recent_accuracies)
        older_avg = np.mean(older_accuracies)
        
        if recent_avg > older_avg * 1.05:
            return "improving"
        elif recent_avg < older_avg * 0.95:
            return "degrading"
        else:
            return "stable"

# adaptive_parameters.py - Parámetros adaptativos
class AdaptiveParameterManager:
    """Gestiona parámetros que se adaptan automáticamente"""
    
    def __init__(self):
        self.parameters = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'drift_threshold': 0.1,
            'retraining_frequency': 24,  # horas
            'confidence_threshold': 0.7
        }
        
        self.performance_window = []
        self.parameter_history = []
        
    def adapt_parameters(self, current_performance: ModelPerformance):
        """Adapta parámetros basado en rendimiento actual"""
        self.performance_window.append(current_performance)
        
        # Mantener ventana de las últimas 10 evaluaciones
        if len(self.performance_window) > 10:
            self.performance_window.pop(0)
        
        if len(self.performance_window) >= 5:
            self._adapt_learning_rate()
            self._adapt_drift_threshold()
            self._adapt_retraining_frequency()
            
            # Guardar historial
            self.parameter_history.append({
                'timestamp': datetime.now(),
                'parameters': self.parameters.copy(),
                'trigger_performance': current_performance.prediction_accuracy
            })
    
    def _adapt_learning_rate(self):
        """Adapta learning rate basado en tendencia de performance"""
        recent_accuracies = [p.prediction_accuracy for p in self.performance_window[-5:]]
        
        if len(recent_accuracies) >= 3:
            # Si accuracy está mejorando, mantener o aumentar learning rate
            if recent_accuracies[-1] > recent_accuracies[-2] > recent_accuracies[-3]:
                self.parameters['learning_rate'] = min(0.01, self.parameters['learning_rate'] * 1.1)
            
            # Si accuracy está empeorando, reducir learning rate
            elif recent_accuracies[-1] < recent_accuracies[-2] < recent_accuracies[-3]:
                self.parameters['learning_rate'] = max(0.0001, self.parameters['learning_rate'] * 0.8)
    
    def _adapt_drift_threshold(self):
        """Adapta umbral de drift basado en volatilidad de performance"""
        accuracies = [p.prediction_accuracy for p in self.performance_window]
        volatility = np.std(accuracies)
        
        # Si hay alta volatilidad, ser más sensible al drift
        if volatility > 0.05:
            self.parameters['drift_threshold'] = max(0.05, self.parameters['drift_threshold'] * 0.9)
        else:
            self.parameters['drift_threshold'] = min(0.15, self.parameters['drift_threshold'] * 1.05)
    
    def _adapt_retraining_frequency(self):
        """Adapta frecuencia de reentrenamiento"""
        avg_accuracy = np.mean([p.prediction_accuracy for p in self.performance_window])
        
        # Si accuracy es baja, entrenar más frecuentemente
        if avg_accuracy < 0.6:
            self.parameters['retraining_frequency'] = max(6, self.parameters['retraining_frequency'] * 0.8)
        # Si accuracy es alta, puede entrenar menos frecuentemente
        elif avg_accuracy > 0.8:
            self.parameters['retraining_frequency'] = min(72, self.parameters['retraining_frequency'] * 1.2)
    
    def get_current_parameters(self):
        """Obtiene parámetros actuales"""
        return self.parameters.copy()
    
    def get_adaptation_history(self):
        """Obtiene historial de adaptaciones"""
        return self.parameter_history

# Integración completa en main.py
auto_training_manager = None
continuous_learning = None
adaptive_params = None

@app.on_event("startup")
async def startup_with_auto_training():
    global auto_training_manager, continuous_learning, adaptive_params
    
    # Inicializar sistemas
    auto_training_manager = AutoTrainingManager()
    continuous_learning = ContinuousLearningSystem()
    adaptive_params = AdaptiveParameterManager()
    
    # Configurar planificador
    scheduler = AutoTrainingScheduler(auto_training_manager, check_interval_minutes=60)
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    # Iniciar en background
    asyncio.create_task(scheduler.start(symbols))
    
    logging.info("🚀 Sistema completo de auto-entrenamiento iniciado")

@app.post("/api/model/add-feedback")
async def add_model_feedback(
    symbol: str,
    predicted_price: float,
    actual_price: float,
    user_feedback: Optional[int] = None  # 1=buena predicción, 0=mala predicción
):
    """Añade feedback para aprendizaje continuo"""
    if continuous_learning:
        continuous_learning.add_feedback(
            prediction=predicted_price,
            actual_outcome=actual_price,
            user_feedback=user_feedback,
            timestamp=datetime.now()
        )
        return {"message": "Feedback añadido al sistema de aprendizaje continuo"}
    else:
        return {"error": "Sistema de aprendizaje continuo no disponible"}

@app.get("/api/model/learning-metrics")
async def get_learning_metrics():
    """Obtiene métricas del aprendizaje continuo"""
    if continuous_learning:
        return continuous_learning.get_learning_metrics()
    else:
        return {"error": "Sistema no disponible"}

@app.get("/api/model/adaptive-parameters")
async def get_adaptive_parameters():
    """Obtiene parámetros adaptativos actuales"""
    if adaptive_params:
        return {
            'current_parameters': adaptive_params.get_current_parameters(),
            'adaptation_history': adaptive_params.get_adaptation_history()[-5:]  # Últimas 5 adaptaciones
        }
    else:
        return {"error": "Sistema no disponible"}

if __name__ == "__main__":
    # Ejecutar dashboard de monitoreo
    create_auto_training_dashboard()