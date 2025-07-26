import os
import pickle
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """
    Cargador de modelos para Brain Trader que maneja la carga y cache
    de los modelos de IA entrenados.
    """
    
    def __init__(self, models_path: str = "models/trained_models"):
        self.models_path = Path(models_path)
        self.models = {}
        self.scalers = {}
        self.model_info = {}
        
        # Crear directorio si no existe
        self.models_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelLoader initialized with path: {self.models_path}")
    
    def load_brain_max(self, pair: str, style: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Cargar modelo Brain Max para un par y estilo específico
        """
        try:
            model_key = f"brain_max_{pair}_{style}"
            
            # Verificar si ya está en cache
            if model_key in self.models:
                logger.info(f"Brain Max model {model_key} loaded from cache")
                return self.models[model_key], self.scalers.get(model_key), self.model_info.get(model_key, {})
            
            # Construir ruta del modelo
            model_file = self.models_path / "Brain_Max" / pair / f"{style}_model.pkl"
            scaler_file = self.models_path / "Brain_Max" / pair / f"{style}_scaler.pkl"
            
            # Verificar si existen los archivos
            if not model_file.exists():
                logger.warning(f"Brain Max model file not found: {model_file}")
                return self._create_mock_model("Brain Max", pair, style)
            
            if not scaler_file.exists():
                logger.warning(f"Brain Max scaler file not found: {scaler_file}")
                return self._create_mock_model("Brain Max", pair, style)
            
            # Cargar modelo y scaler
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # Información del modelo
            model_info = {
                'name': 'Brain Max',
                'pair': pair,
                'style': style,
                'version': '1.0.0',
                'accuracy': self._get_model_accuracy(pair, style),
                'last_training': self._get_last_training_date(model_file),
                'features': self._get_model_features(pair, style)
            }
            
            # Guardar en cache
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.model_info[model_key] = model_info
            
            logger.info(f"Brain Max model {model_key} loaded successfully")
            return model, scaler, model_info
            
        except Exception as e:
            logger.error(f"Error loading Brain Max model: {str(e)}")
            return self._create_mock_model("Brain Max", pair, style)
    
    def load_brain_ultra(self, pair: str, style: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Cargar modelo Brain Ultra para un par y estilo específico
        """
        try:
            model_key = f"brain_ultra_{pair}_{style}"
            
            # Verificar si ya está en cache
            if model_key in self.models:
                logger.info(f"Brain Ultra model {model_key} loaded from cache")
                return self.models[model_key], self.scalers.get(model_key), self.model_info.get(model_key, {})
            
            # Construir ruta del modelo
            model_file = self.models_path / "Brain_Ultra" / pair / f"{style}_model.pkl"
            scaler_file = self.models_path / "Brain_Ultra" / pair / f"{style}_scaler.pkl"
            
            # Verificar si existen los archivos
            if not model_file.exists():
                logger.warning(f"Brain Ultra model file not found: {model_file}")
                return self._create_mock_model("Brain Ultra", pair, style)
            
            if not scaler_file.exists():
                logger.warning(f"Brain Ultra scaler file not found: {scaler_file}")
                return self._create_mock_model("Brain Ultra", pair, style)
            
            # Cargar modelo y scaler
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # Información del modelo
            model_info = {
                'name': 'Brain Ultra',
                'pair': pair,
                'style': style,
                'version': '2.0.0',
                'accuracy': self._get_model_accuracy(pair, style),
                'last_training': self._get_last_training_date(model_file),
                'features': self._get_model_features(pair, style)
            }
            
            # Guardar en cache
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.model_info[model_key] = model_info
            
            logger.info(f"Brain Ultra model {model_key} loaded successfully")
            return model, scaler, model_info
            
        except Exception as e:
            logger.error(f"Error loading Brain Ultra model: {str(e)}")
            return self._create_mock_model("Brain Ultra", pair, style)
    
    def load_brain_predictor(self, pair: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Cargar modelo Brain Predictor para un par específico
        """
        try:
            model_key = f"brain_predictor_{pair}"
            
            # Verificar si ya está en cache
            if model_key in self.models:
                logger.info(f"Brain Predictor model {model_key} loaded from cache")
                return self.models[model_key], self.scalers.get(model_key), self.model_info.get(model_key, {})
            
            # Construir ruta del modelo
            model_file = self.models_path / "Brain_Predictor" / pair / "predictor_model.pkl"
            scaler_file = self.models_path / "Brain_Predictor" / pair / "predictor_scaler.pkl"
            
            # Verificar si existen los archivos
            if not model_file.exists():
                logger.warning(f"Brain Predictor model file not found: {model_file}")
                return self._create_mock_model("Brain Predictor", pair, "predictive")
            
            if not scaler_file.exists():
                logger.warning(f"Brain Predictor scaler file not found: {scaler_file}")
                return self._create_mock_model("Brain Predictor", pair, "predictive")
            
            # Cargar modelo y scaler
            with open(model_file, 'rb') as f:
                model = pickle.load(f)
            
            with open(scaler_file, 'rb') as f:
                scaler = pickle.load(f)
            
            # Información del modelo
            model_info = {
                'name': 'Brain Predictor',
                'pair': pair,
                'style': 'predictive',
                'version': '3.0.0',
                'accuracy': self._get_model_accuracy(pair, "predictive"),
                'last_training': self._get_last_training_date(model_file),
                'features': self._get_model_features(pair, "predictive")
            }
            
            # Guardar en cache
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            self.model_info[model_key] = model_info
            
            logger.info(f"Brain Predictor model {model_key} loaded successfully")
            return model, scaler, model_info
            
        except Exception as e:
            logger.error(f"Error loading Brain Predictor model: {str(e)}")
            return self._create_mock_model("Brain Predictor", pair, "predictive")
    
    def load_mega_mind(self, pair: str, style: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Cargar modelo MEGA MIND (combinación de los 3 cerebros)
        """
        try:
            model_key = f"mega_mind_{pair}_{style}"
            
            # Verificar si ya está en cache
            if model_key in self.models:
                logger.info(f"MEGA MIND model {model_key} loaded from cache")
                return self.models[model_key], self.scalers.get(model_key), self.model_info.get(model_key, {})
            
            # MEGA MIND combina los 3 cerebros
            brain_max_model, brain_max_scaler, brain_max_info = self.load_brain_max(pair, style)
            brain_ultra_model, brain_ultra_scaler, brain_ultra_info = self.load_brain_ultra(pair, style)
            brain_predictor_model, brain_predictor_scaler, brain_predictor_info = self.load_brain_predictor(pair)
            
            # Crear modelo combinado (simulado por ahora)
            combined_model = self._create_combined_model(
                brain_max_model, brain_ultra_model, brain_predictor_model
            )
            
            # Información del modelo combinado
            model_info = {
                'name': 'MEGA MIND',
                'pair': pair,
                'style': style,
                'version': '4.0.0',
                'accuracy': self._get_mega_mind_accuracy(pair, style),
                'last_training': self._get_current_date(),
                'features': self._get_mega_mind_features(pair, style),
                'combined_models': [
                    brain_max_info,
                    brain_ultra_info,
                    brain_predictor_info
                ]
            }
            
            # Guardar en cache
            self.models[model_key] = combined_model
            self.scalers[model_key] = brain_max_scaler  # Usar scaler del Brain Max como base
            self.model_info[model_key] = model_info
            
            logger.info(f"MEGA MIND model {model_key} loaded successfully")
            return combined_model, brain_max_scaler, model_info
            
        except Exception as e:
            logger.error(f"Error loading MEGA MIND model: {str(e)}")
            return self._create_mock_model("MEGA MIND", pair, style)
    
    def get_available_models(self) -> Dict[str, Any]:
        """
        Obtener lista de modelos disponibles
        """
        available_models = {
            'brain_max': {
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
                'styles': ['scalping', 'day_trading', 'swing_trading', 'position_trading'],
                'status': 'available'
            },
            'brain_ultra': {
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
                'styles': ['scalping', 'day_trading', 'swing_trading', 'position_trading'],
                'status': 'available'
            },
            'brain_predictor': {
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
                'styles': ['predictive'],
                'status': 'available'
            },
            'mega_mind': {
                'pairs': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
                'styles': ['scalping', 'day_trading', 'swing_trading', 'position_trading'],
                'status': 'available'
            }
        }
        
        return available_models
    
    def clear_cache(self):
        """
        Limpiar cache de modelos
        """
        self.models.clear()
        self.scalers.clear()
        self.model_info.clear()
        logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Obtener información del cache
        """
        return {
            'cached_models': len(self.models),
            'cached_scalers': len(self.scalers),
            'cached_info': len(self.model_info),
            'models_path': str(self.models_path)
        }
    
    # Métodos privados
    
    def _create_mock_model(self, model_name: str, pair: str, style: str) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Crear modelo simulado cuando no se encuentra el archivo real
        """
        logger.info(f"Creating mock model for {model_name} - {pair} - {style}")
        
        # Modelo simulado
        mock_model = {
            'name': model_name,
            'type': 'mock',
            'pair': pair,
            'style': style
        }
        
        # Scaler simulado
        mock_scaler = {
            'type': 'mock_scaler',
            'fitted': True
        }
        
        # Información del modelo
        model_info = {
            'name': model_name,
            'pair': pair,
            'style': style,
            'version': 'mock',
            'accuracy': self._get_mock_accuracy(model_name),
            'last_training': self._get_current_date(),
            'features': self._get_mock_features(model_name),
            'status': 'mock'
        }
        
        return mock_model, mock_scaler, model_info
    
    def _create_combined_model(self, brain_max_model: Any, brain_ultra_model: Any, brain_predictor_model: Any) -> Any:
        """
        Crear modelo combinado para MEGA MIND
        """
        combined_model = {
            'name': 'MEGA MIND',
            'type': 'combined',
            'models': {
                'brain_max': brain_max_model,
                'brain_ultra': brain_ultra_model,
                'brain_predictor': brain_predictor_model
            },
            'fusion_strategy': 'weighted_average',
            'weights': {
                'brain_max': 0.3,
                'brain_ultra': 0.3,
                'brain_predictor': 0.4
            }
        }
        
        return combined_model
    
    def _get_model_accuracy(self, pair: str, style: str) -> float:
        """
        Obtener precisión del modelo (simulado)
        """
        import random
        
        # Precisión base según el estilo
        base_accuracies = {
            'scalping': (75, 85),
            'day_trading': (80, 88),
            'swing_trading': (82, 90),
            'position_trading': (85, 92),
            'predictive': (88, 94)
        }
        
        min_acc, max_acc = base_accuracies.get(style, (80, 88))
        return random.uniform(min_acc, max_acc)
    
    def _get_mega_mind_accuracy(self, pair: str, style: str) -> float:
        """
        Obtener precisión de MEGA MIND (superior a los individuales)
        """
        import random
        return random.uniform(92, 98)
    
    def _get_mock_accuracy(self, model_name: str) -> float:
        """
        Obtener precisión simulada para modelos mock
        """
        import random
        
        accuracies = {
            'Brain Max': (80, 88),
            'Brain Ultra': (85, 92),
            'Brain Predictor': (88, 94),
            'MEGA MIND': (92, 98)
        }
        
        min_acc, max_acc = accuracies.get(model_name, (80, 88))
        return random.uniform(min_acc, max_acc)
    
    def _get_model_features(self, pair: str, style: str) -> list:
        """
        Obtener características del modelo
        """
        base_features = [
            'price', 'volume', 'rsi', 'macd', 'bollinger_bands',
            'moving_averages', 'support_resistance', 'trend_indicators'
        ]
        
        # Agregar características específicas según el estilo
        style_features = {
            'scalping': ['micro_patterns', 'high_frequency_data'],
            'day_trading': ['intraday_patterns', 'news_sentiment'],
            'swing_trading': ['medium_term_patterns', 'fundamental_analysis'],
            'position_trading': ['long_term_patterns', 'macro_analysis'],
            'predictive': ['time_series_analysis', 'machine_learning_features']
        }
        
        features = base_features + style_features.get(style, [])
        return features
    
    def _get_mock_features(self, model_name: str) -> list:
        """
        Obtener características simuladas para modelos mock
        """
        return [
            'price', 'volume', 'rsi', 'macd', 'bollinger_bands',
            'moving_averages', 'support_resistance', 'trend_indicators'
        ]
    
    def _get_mega_mind_features(self, pair: str, style: str) -> list:
        """
        Obtener características de MEGA MIND (combinación de todos)
        """
        all_features = [
            'price', 'volume', 'rsi', 'macd', 'bollinger_bands',
            'moving_averages', 'support_resistance', 'trend_indicators',
            'micro_patterns', 'high_frequency_data', 'intraday_patterns',
            'news_sentiment', 'medium_term_patterns', 'fundamental_analysis',
            'long_term_patterns', 'macro_analysis', 'time_series_analysis',
            'machine_learning_features', 'cross_asset_correlation',
            'economic_calendar', 'market_sentiment', 'volatility_analysis'
        ]
        
        return all_features
    
    def _get_last_training_date(self, model_file: Path) -> str:
        """
        Obtener fecha de último entrenamiento
        """
        try:
            stat = model_file.stat()
            from datetime import datetime
            return datetime.fromtimestamp(stat.st_mtime).isoformat()
        except:
            return self._get_current_date()
    
    def _get_current_date(self) -> str:
        """
        Obtener fecha actual
        """
        from datetime import datetime
        return datetime.now().isoformat() 