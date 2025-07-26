#!/usr/bin/env python3
"""
Servicio para cargar y verificar modelos entrenados
==================================================
Servicio que carga los modelos existentes y los integra con el sistema de suscripciones
"""

import os
import json
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ModelLoaderService:
    """Servicio para cargar modelos entrenados existentes"""
    
    def __init__(self, models_dir: str = "backend/models/trained_models"):
        self.models_dir = Path(models_dir)
        self.loaded_models = {}
        self.model_metadata = {}
        
    def scan_existing_models(self) -> Dict[str, Any]:
        """Escanea todos los modelos entrenados existentes"""
        logger.info("🔍 Escaneando modelos entrenados...")
        
        scan_results = {
            'brain_max': {},
            'brain_ultra': {},
            'total_models': 0,
            'total_pairs': 0,
            'total_styles': 0
        }
        
        # Escanear Brain_Max
        brain_max_dir = self.models_dir / "Brain_Max"
        if brain_max_dir.exists():
            scan_results['brain_max'] = self._scan_brain_max(brain_max_dir)
        
        # Escanear Brain_Ultra
        brain_ultra_dir = self.models_dir / "Brain_Ultra"
        if brain_ultra_dir.exists():
            scan_results['brain_ultra'] = self._scan_brain_ultra(brain_ultra_dir)
        
        # Calcular totales
        scan_results['total_models'] = (
            scan_results['brain_max'].get('total_models', 0) +
            scan_results['brain_ultra'].get('total_models', 0)
        )
        scan_results['total_pairs'] = (
            scan_results['brain_max'].get('total_pairs', 0) +
            scan_results['brain_ultra'].get('total_pairs', 0)
        )
        scan_results['total_styles'] = (
            scan_results['brain_max'].get('total_styles', 0) +
            scan_results['brain_ultra'].get('total_styles', 0)
        )
        
        logger.info(f"✅ Escaneo completado: {scan_results['total_models']} modelos encontrados")
        return scan_results
    
    def _scan_brain_max(self, brain_max_dir: Path) -> Dict[str, Any]:
        """Escanea modelos de Brain_Max"""
        results = {
            'pairs': {},
            'total_models': 0,
            'total_pairs': 0,
            'total_styles': 0
        }
        
        for pair_dir in brain_max_dir.iterdir():
            if pair_dir.is_dir():
                pair_name = pair_dir.name
                results['pairs'][pair_name] = {
                    'styles': {},
                    'total_models': 0
                }
                
                for style_dir in pair_dir.iterdir():
                    if style_dir.is_dir():
                        style_name = style_dir.name
                        model_files = list(style_dir.glob("*.pkl"))
                        metadata_file = style_dir / "metadata.json"
                        
                        results['pairs'][pair_name]['styles'][style_name] = {
                            'model_files': len(model_files),
                            'models': [f.name for f in model_files],
                            'has_metadata': metadata_file.exists()
                        }
                        
                        results['pairs'][pair_name]['total_models'] += len(model_files)
                        results['total_models'] += len(model_files)
                        results['total_styles'] += 1
                
                results['total_pairs'] += 1
        
        return results
    
    def _scan_brain_ultra(self, brain_ultra_dir: Path) -> Dict[str, Any]:
        """Escanea modelos de Brain_Ultra"""
        results = {
            'pairs': {},
            'total_models': 0,
            'total_pairs': 0,
            'total_styles': 0
        }
        
        for pair_dir in brain_ultra_dir.iterdir():
            if pair_dir.is_dir():
                pair_name = pair_dir.name
                results['pairs'][pair_name] = {
                    'styles': {},
                    'total_models': 0
                }
                
                # Buscar archivos de modelos por estilo
                model_files = list(pair_dir.glob("*.pkl"))
                style_models = {}
                
                for model_file in model_files:
                    filename = model_file.name
                    if '_strategies_' in filename:
                        # Extraer estilo del nombre del archivo
                        parts = filename.split('_strategies_')
                        if len(parts) > 1:
                            style_part = parts[1]
                            if '_' in style_part:
                                style_name = style_part.split('_')[0]
                                if style_name not in style_models:
                                    style_models[style_name] = []
                                style_models[style_name].append(filename)
                
                for style_name, models in style_models.items():
                    results['pairs'][pair_name]['styles'][style_name] = {
                        'model_files': len(models),
                        'models': models,
                        'has_metadata': any('metadata' in m for m in models)
                    }
                    results['pairs'][pair_name]['total_models'] += len(models)
                    results['total_models'] += len(models)
                    results['total_styles'] += 1
                
                results['total_pairs'] += 1
        
        return results
    
    def load_model(self, brain_type: str, pair: str, style: str, model_name: str) -> Optional[Any]:
        """Carga un modelo específico"""
        try:
            if brain_type == "brain_max":
                model_path = self.models_dir / "Brain_Max" / pair / style / f"{model_name}.pkl"
            elif brain_type == "brain_ultra":
                model_path = self.models_dir / "Brain_Ultra" / pair / f"{pair}_strategies_{style}_{model_name}.pkl"
            else:
                logger.error(f"Tipo de cerebro no válido: {brain_type}")
                return None
            
            if not model_path.exists():
                logger.warning(f"Modelo no encontrado: {model_path}")
                return None
            
            model = joblib.load(model_path)
            logger.info(f"✅ Modelo cargado: {brain_type}/{pair}/{style}/{model_name}")
            return model
            
        except Exception as e:
            logger.error(f"❌ Error cargando modelo {model_path}: {e}")
            return None
    
    def load_style_models(self, brain_type: str, pair: str, style: str) -> Dict[str, Any]:
        """Carga todos los modelos de un estilo específico"""
        models = {}
        
        if brain_type == "brain_max":
            style_dir = self.models_dir / "Brain_Max" / pair / style
            if style_dir.exists():
                for model_file in style_dir.glob("*.pkl"):
                    model_name = model_file.stem
                    model = self.load_model(brain_type, pair, style, model_name)
                    if model:
                        models[model_name] = model
        elif brain_type == "brain_ultra":
            pair_dir = self.models_dir / "Brain_Ultra" / pair
            if pair_dir.exists():
                # Buscar modelos del estilo específico
                pattern = f"{pair}_strategies_{style}_*.pkl"
                for model_file in pair_dir.glob(pattern):
                    if not model_file.name.endswith('_scaler.pkl') and not model_file.name.endswith('_metadata.pkl'):
                        model_name = model_file.name.replace(f"{pair}_strategies_{style}_", "").replace(".pkl", "")
                        model = self.load_model(brain_type, pair, style, model_name)
                        if model:
                            models[model_name] = model
        
        logger.info(f"✅ Cargados {len(models)} modelos para {brain_type}/{pair}/{style}")
        return models
    
    def get_model_info(self, brain_type: str, pair: str, style: str) -> Dict[str, Any]:
        """Obtiene información de los modelos de un estilo"""
        info = {
            'brain_type': brain_type,
            'pair': pair,
            'style': style,
            'models_available': [],
            'metadata': None,
            'performance': None
        }
        
        try:
            if brain_type == "brain_max":
                metadata_path = self.models_dir / "Brain_Max" / pair / style / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        info['metadata'] = json.load(f)
                        info['performance'] = info['metadata'].get('trading_results', {})
                
                style_dir = self.models_dir / "Brain_Max" / pair / style
                if style_dir.exists():
                    info['models_available'] = [f.stem for f in style_dir.glob("*.pkl") if not f.name.endswith('_metadata.pkl')]
            
            elif brain_type == "brain_ultra":
                pair_dir = self.models_dir / "Brain_Ultra" / pair
                if pair_dir.exists():
                    # Buscar metadata
                    metadata_pattern = f"{pair}_strategies_{style}_metadata.pkl"
                    metadata_files = list(pair_dir.glob(metadata_pattern))
                    if metadata_files:
                        try:
                            metadata = joblib.load(metadata_files[0])
                            info['metadata'] = metadata
                        except:
                            pass
                    
                    # Buscar modelos
                    pattern = f"{pair}_strategies_{style}_*.pkl"
                    model_files = list(pair_dir.glob(pattern))
                    info['models_available'] = [
                        f.name.replace(f"{pair}_strategies_{style}_", "").replace(".pkl", "")
                        for f in model_files
                        if not f.name.endswith('_scaler.pkl') and not f.name.endswith('_metadata.pkl')
                    ]
        
        except Exception as e:
            logger.error(f"Error obteniendo información del modelo: {e}")
        
        return info
    
    def verify_model_compatibility(self) -> Dict[str, Any]:
        """Verifica la compatibilidad de los modelos con el sistema actual"""
        logger.info("🔍 Verificando compatibilidad de modelos...")
        
        compatibility_report = {
            'brain_max': {
                'compatible': True,
                'issues': [],
                'recommendations': []
            },
            'brain_ultra': {
                'compatible': True,
                'issues': [],
                'recommendations': []
            },
            'overall_compatible': True
        }
        
        # Verificar Brain_Max
        brain_max_dir = self.models_dir / "Brain_Max"
        if brain_max_dir.exists():
            for pair_dir in brain_max_dir.iterdir():
                if pair_dir.is_dir():
                    for style_dir in pair_dir.iterdir():
                        if style_dir.is_dir():
                            # Verificar que tenga modelos
                            model_files = list(style_dir.glob("*.pkl"))
                            if not model_files:
                                compatibility_report['brain_max']['issues'].append(
                                    f"No hay modelos en {pair_dir.name}/{style_dir.name}"
                                )
                            
                            # Verificar metadata
                            metadata_file = style_dir / "metadata.json"
                            if not metadata_file.exists():
                                compatibility_report['brain_max']['recommendations'].append(
                                    f"Agregar metadata.json en {pair_dir.name}/{style_dir.name}"
                                )
        
        # Verificar Brain_Ultra
        brain_ultra_dir = self.models_dir / "Brain_Ultra"
        if brain_ultra_dir.exists():
            for pair_dir in brain_ultra_dir.iterdir():
                if pair_dir.is_dir():
                    # Verificar que tenga modelos
                    model_files = list(pair_dir.glob("*.pkl"))
                    if not model_files:
                        compatibility_report['brain_ultra']['issues'].append(
                            f"No hay modelos en {pair_dir.name}"
                        )
        
        # Determinar compatibilidad general
        if (compatibility_report['brain_max']['issues'] or 
            compatibility_report['brain_ultra']['issues']):
            compatibility_report['overall_compatible'] = False
        
        logger.info(f"✅ Verificación completada: {'Compatible' if compatibility_report['overall_compatible'] else 'Incompatible'}")
        return compatibility_report

# Función de utilidad para mostrar el estado de los modelos
def show_models_status():
    """Muestra el estado actual de los modelos entrenados"""
    loader = ModelLoaderService()
    
    print("🧠 ESTADO DE MODELOS ENTRENADOS")
    print("=" * 50)
    
    # Escanear modelos
    scan_results = loader.scan_existing_models()
    
    print(f"📊 Total de modelos: {scan_results['total_models']}")
    print(f"📈 Total de pares: {scan_results['total_pairs']}")
    print(f"🎯 Total de estilos: {scan_results['total_styles']}")
    
    print("\n🧠 Brain_Max:")
    for pair, pair_info in scan_results['brain_max'].get('pairs', {}).items():
        print(f"  📈 {pair}: {pair_info['total_models']} modelos")
        for style, style_info in pair_info['styles'].items():
            print(f"    🎯 {style}: {style_info['model_files']} archivos")
    
    print("\n🧠 Brain_Ultra:")
    for pair, pair_info in scan_results['brain_ultra'].get('pairs', {}).items():
        print(f"  📈 {pair}: {pair_info['total_models']} modelos")
        for style, style_info in pair_info['styles'].items():
            print(f"    🎯 {style}: {style_info['model_files']} archivos")
    
    # Verificar compatibilidad
    compatibility = loader.verify_model_compatibility()
    print(f"\n✅ Compatibilidad: {'OK' if compatibility['overall_compatible'] else 'ISSUES'}")
    
    if not compatibility['overall_compatible']:
        print("\n⚠️ Problemas encontrados:")
        for brain_type, report in compatibility.items():
            if brain_type != 'overall_compatible' and report['issues']:
                print(f"  {brain_type}:")
                for issue in report['issues']:
                    print(f"    - {issue}")

if __name__ == "__main__":
    show_models_status() 