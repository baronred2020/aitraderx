# hyperparameter_optimization.py - Sistema avanzado de optimización de hiperparámetros
import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedHyperparameterOptimizer:
    """Sistema avanzado de optimización de hiperparámetros para trading"""
    
    def __init__(self, data_source, max_trials=100, cv_folds=5):
        self.data_source = data_source
        self.max_trials = max_trials
        self.cv_folds = cv_folds
        self.best_params = {}
        self.optimization_history = []
        self.logger = logging.getLogger(__name__)
        
    def optimize_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                             study_name: str = "random_forest_optimization") -> Dict:
        """Optimiza hiperparámetros para Random Forest"""
        
        def objective(trial):
            # Hiperparámetros a optimizar
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
            }
            
            # Validación temporal (Time Series Cross Validation)
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Escalar features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                # Entrenar modelo
                model = RandomForestClassifier(**params, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Predecir y evaluar
                y_pred = model.predict(X_val_scaled)
                
                # Métricas múltiples
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                # Score compuesto (priorizar precisión en trading)
                composite_score = (accuracy * 0.3 + precision * 0.4 + recall * 0.2 + f1 * 0.1)
                scores.append(composite_score)
            
            return np.mean(scores)
        
        # Crear estudio de optimización
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=None,  # En producción usar SQLite o PostgreSQL
            load_if_exists=True
        )
        
        # Ejecutar optimización
        study.optimize(objective, n_trials=self.max_trials, timeout=3600)  # 1 hora timeout
        
        self.best_params['random_forest'] = study.best_params
        self.optimization_history.append({
            'model': 'RandomForest',
            'best_score': study.best_value,
            'best_params': study.best_params,
            'timestamp': datetime.now()
        })
        
        return study.best_params
    
    def optimize_gradient_boosting(self, X: pd.DataFrame, y: pd.Series,
                                  study_name: str = "gradient_boosting_optimization") -> Dict:
        """Optimiza hiperparámetros para Gradient Boosting"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'loss': trial.suggest_categorical('loss', ['deviance', 'exponential'])
            }
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model = GradientBoostingRegressor(**params, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                y_pred = model.predict(X_val_scaled)
                
                # Para regresión, usar R² y MSE
                from sklearn.metrics import r2_score, mean_squared_error
                r2 = r2_score(y_val, y_pred)
                mse = mean_squared_error(y_val, y_pred)
                
                # Score compuesto para regresión
                composite_score = r2 - (mse / 1000)  # Penalizar MSE
                scores.append(composite_score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.max_trials, timeout=3600)
        
        self.best_params['gradient_boosting'] = study.best_params
        return study.best_params
    
    def optimize_xgboost(self, X: pd.DataFrame, y: pd.Series,
                        study_name: str = "xgboost_optimization") -> Dict:
        """Optimiza hiperparámetros para XGBoost"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                composite_score = (accuracy * 0.3 + precision * 0.4 + recall * 0.2 + f1 * 0.1)
                scores.append(composite_score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.max_trials, timeout=3600)
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params
    
    def optimize_lightgbm(self, X: pd.DataFrame, y: pd.Series,
                         study_name: str = "lightgbm_optimization") -> Dict:
        """Optimiza hiperparámetros para LightGBM"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10)
            }
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                composite_score = (accuracy * 0.3 + precision * 0.4 + recall * 0.2 + f1 * 0.1)
                scores.append(composite_score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=self.max_trials, timeout=3600)
        
        self.best_params['lightgbm'] = study.best_params
        return study.best_params
    
    def optimize_ensemble(self, X: pd.DataFrame, y: pd.Series,
                         study_name: str = "ensemble_optimization") -> Dict:
        """Optimiza pesos de ensemble de modelos"""
        
        def objective(trial):
            # Pesos para cada modelo
            rf_weight = trial.suggest_float('rf_weight', 0.1, 0.5)
            xgb_weight = trial.suggest_float('xgb_weight', 0.1, 0.5)
            lgb_weight = trial.suggest_float('lgb_weight', 0.1, 0.5)
            
            # Normalizar pesos
            total_weight = rf_weight + xgb_weight + lgb_weight
            rf_weight /= total_weight
            xgb_weight /= total_weight
            lgb_weight /= total_weight
            
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Entrenar modelos individuales
                rf_model = RandomForestClassifier(**self.best_params.get('random_forest', {}), random_state=42)
                xgb_model = xgb.XGBClassifier(**self.best_params.get('xgboost', {}), random_state=42)
                lgb_model = lgb.LGBMClassifier(**self.best_params.get('lightgbm', {}), random_state=42)
                
                rf_model.fit(X_train, y_train)
                xgb_model.fit(X_train, y_train)
                lgb_model.fit(X_train, y_train)
                
                # Predicciones ponderadas
                rf_pred = rf_model.predict_proba(X_val)[:, 1]
                xgb_pred = xgb_model.predict_proba(X_val)[:, 1]
                lgb_pred = lgb_model.predict_proba(X_val)[:, 1]
                
                ensemble_pred = (rf_weight * rf_pred + 
                               xgb_weight * xgb_pred + 
                               lgb_weight * lgb_pred)
                
                # Convertir a clases
                y_pred = (ensemble_pred > 0.5).astype(int)
                
                accuracy = accuracy_score(y_val, y_pred)
                precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
                
                composite_score = (accuracy * 0.3 + precision * 0.4 + recall * 0.2 + f1 * 0.1)
                scores.append(composite_score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', study_name=study_name)
        study.optimize(objective, n_trials=50, timeout=1800)  # Menos trials para ensemble
        
        self.best_params['ensemble'] = study.best_params
        return study.best_params
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """Optimiza todos los modelos en secuencia"""
        self.logger.info("🚀 Iniciando optimización completa de hiperparámetros...")
        
        results = {}
        
        # Optimizar cada modelo individual
        models_to_optimize = [
            ('random_forest', self.optimize_random_forest),
            ('gradient_boosting', self.optimize_gradient_boosting),
            ('xgboost', self.optimize_xgboost),
            ('lightgbm', self.optimize_lightgbm)
        ]
        
        for model_name, optimizer_func in models_to_optimize:
            try:
                self.logger.info(f"🔧 Optimizando {model_name}...")
                best_params = optimizer_func(X, y)
                results[model_name] = best_params
                self.logger.info(f"✅ {model_name} optimizado: {best_params}")
            except Exception as e:
                self.logger.error(f"❌ Error optimizando {model_name}: {e}")
        
        # Optimizar ensemble
        if len(results) >= 2:  # Necesitamos al menos 2 modelos para ensemble
            try:
                self.logger.info("🔧 Optimizando ensemble...")
                ensemble_params = self.optimize_ensemble(X, y)
                results['ensemble'] = ensemble_params
                self.logger.info(f"✅ Ensemble optimizado: {ensemble_params}")
            except Exception as e:
                self.logger.error(f"❌ Error optimizando ensemble: {e}")
        
        return results
    
    def create_optimized_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Crea modelos optimizados con los mejores hiperparámetros"""
        optimized_models = {}
        
        # Random Forest optimizado
        if 'random_forest' in self.best_params:
            rf_params = self.best_params['random_forest']
            optimized_models['random_forest'] = RandomForestClassifier(**rf_params, random_state=42)
        
        # XGBoost optimizado
        if 'xgboost' in self.best_params:
            xgb_params = self.best_params['xgboost']
            optimized_models['xgboost'] = xgb.XGBClassifier(**xgb_params, random_state=42)
        
        # LightGBM optimizado
        if 'lightgbm' in self.best_params:
            lgb_params = self.best_params['lightgbm']
            optimized_models['lightgbm'] = lgb.LGBMClassifier(**lgb_params, random_state=42)
        
        # Ensemble optimizado
        if 'ensemble' in self.best_params:
            ensemble_params = self.best_params['ensemble']
            optimized_models['ensemble'] = {
                'weights': ensemble_params,
                'models': [optimized_models.get('random_forest'), 
                          optimized_models.get('xgboost'),
                          optimized_models.get('lightgbm')]
            }
        
        return optimized_models
    
    def save_optimization_results(self, filepath: str = "models/optimization_results.pkl"):
        """Guarda los resultados de optimización"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        results = {
            'best_params': self.best_params,
            'optimization_history': self.optimization_history,
            'timestamp': datetime.now()
        }
        
        joblib.dump(results, filepath)
        self.logger.info(f"💾 Resultados guardados en {filepath}")
    
    def load_optimization_results(self, filepath: str = "models/optimization_results.pkl"):
        """Carga resultados de optimización previos"""
        try:
            results = joblib.load(filepath)
            self.best_params = results['best_params']
            self.optimization_history = results['optimization_history']
            self.logger.info(f"📂 Resultados cargados desde {filepath}")
            return True
        except FileNotFoundError:
            self.logger.warning(f"⚠️ No se encontraron resultados previos en {filepath}")
            return False

# Función de utilidad para integrar con el sistema existente
def integrate_hyperparameter_optimization(ai_model, market_data: pd.DataFrame):
    """Integra la optimización de hiperparámetros con el modelo AI existente"""
    
    # Preparar datos
    from ai_models import AdvancedTradingAI
    
    # Crear features
    ai_model.create_features(market_data)
    
    # Preparar features para optimización
    feature_columns = [
        'rsi', 'macd', 'momentum_5', 'momentum_10', 'momentum_20',
        'volatility_5', 'volatility_10', 'volume_ratio', 'high_low_ratio',
        'trend_5_20', 'trend_20_50', 'day_of_week', 'month'
    ]
    
    X = market_data[feature_columns].fillna(0)
    y = ai_model.create_signals(market_data)
    
    # Filtrar datos válidos
    valid_mask = ~X.isin([np.inf, -np.inf]).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 100:
        raise ValueError("Datos insuficientes para optimización")
    
    # Crear optimizador
    optimizer = AdvancedHyperparameterOptimizer(data_source=None, max_trials=50)
    
    # Optimizar todos los modelos
    best_params = optimizer.optimize_all_models(X, y)
    
    # Crear modelos optimizados
    optimized_models = optimizer.create_optimized_models(X, y)
    
    # Guardar resultados
    optimizer.save_optimization_results()
    
    return optimized_models, best_params 