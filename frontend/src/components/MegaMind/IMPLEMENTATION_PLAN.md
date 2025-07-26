# 🧠 Mega Mind - Plan de Implementación
## Sistema de Cerebros Colaborativos Multi-Institucional

---

## 📋 Resumen Ejecutivo

### Objetivo
Implementar un sistema de entrenamiento de cerebros colaborativos que permita a múltiples instituciones entrenar y personalizar los 3 modelos base (Brain_Max, Brain_Ultra, Brain_Predictor) manteniendo aislamiento completo entre instituciones.

### Arquitectura
- **Modelos Base**: 3 cerebros pre-entrenados de la aplicación
- **Personalización**: Cada institución entrena versiones personalizadas
- **Almacenamiento**: Cloud Storage (AWS S3) + Base de datos para metadatos
- **Escalabilidad**: Sistema multi-tenant con aislamiento completo

---

## 🏗️ Arquitectura del Sistema

### 1. Estructura de Almacenamiento

```
aitraderx/
├── models/
│   ├── base_models/ (MODELOS ORIGINALES - NO SE TOCAN)
│   │   ├── Brain_Max/
│   │   │   ├── original_model.pkl
│   │   │   ├── base_config.json
│   │   │   └── metadata.json
│   │   ├── Brain_Ultra/
│   │   └── Brain_Predictor/
│   └── institutions/ (VERSIONES PERSONALIZADAS)
│       ├── institution_001/
│       │   ├── trained_models/
│       │   │   ├── brain_max_trained.pkl
│       │   │   ├── brain_ultra_trained.pkl
│       │   │   └── brain_predictor_trained.pkl
│       │   ├── training_data/
│       │   │   ├── market_data.csv
│       │   │   └── preferences.json
│       │   └── consensus_config.json
│       ├── institution_002/
│       └── institution_003/
```

### 2. Cloud Storage (AWS S3)

```
s3://aitraderx-models/
├── base-models/
│   ├── brain_max_base.pkl
│   ├── brain_ultra_base.pkl
│   └── brain_predictor_base.pkl
├── institutions/
│   ├── institution_001/
│   │   ├── brain_max_trained.pkl
│   │   ├── brain_ultra_trained.pkl
│   │   └── brain_predictor_trained.pkl
│   ├── institution_002/
│   └── institution_003/
└── shared/
    ├── training_datasets/
    └── common_configurations/
```

### 3. Base de Datos (Metadatos)

```sql
-- Instituciones
CREATE TABLE institutions (
    id VARCHAR(50) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    subscription_plan VARCHAR(50) DEFAULT 'institutional',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- Configuraciones de cerebros por institución
CREATE TABLE institution_brain_configs (
    id VARCHAR(100) PRIMARY KEY,
    institution_id VARCHAR(50) NOT NULL,
    brain_type VARCHAR(20) NOT NULL, -- 'brain_max', 'brain_ultra', 'brain_predictor'
    trading_params JSONB, -- stop_loss, take_profit, lot_size
    market_preferences JSONB, -- markets, timeframes, risk_profile
    specializations JSONB, -- specific strategies, indicators
    consensus_weight DECIMAL(3,2) DEFAULT 0.33,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (institution_id) REFERENCES institutions(id)
);

-- Modelos entrenados por institución
CREATE TABLE institution_trained_models (
    id VARCHAR(100) PRIMARY KEY,
    institution_id VARCHAR(50) NOT NULL,
    brain_type VARCHAR(20) NOT NULL,
    config_id VARCHAR(100) NOT NULL, -- Referencia a la configuración usada
    s3_path VARCHAR(500) NOT NULL,
    file_size BIGINT NOT NULL, -- en bytes
    accuracy DECIMAL(5,2),
    training_duration INTEGER, -- en segundos
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    version VARCHAR(20) DEFAULT '1.0.0',
    status VARCHAR(20) DEFAULT 'active',
    FOREIGN KEY (institution_id) REFERENCES institutions(id),
    FOREIGN KEY (config_id) REFERENCES institution_brain_configs(id)
);

-- Configuración de consenso por institución
CREATE TABLE institution_consensus_config (
    institution_id VARCHAR(50) PRIMARY KEY,
    brain_max_weight DECIMAL(3,2) DEFAULT 0.35,
    brain_ultra_weight DECIMAL(3,2) DEFAULT 0.40,
    brain_predictor_weight DECIMAL(3,2) DEFAULT 0.25,
    consensus_threshold DECIMAL(3,2) DEFAULT 0.75,
    min_confidence DECIMAL(3,2) DEFAULT 0.60,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (institution_id) REFERENCES institutions(id)
);

-- Historial de entrenamientos
CREATE TABLE training_history (
    id VARCHAR(100) PRIMARY KEY,
    institution_id VARCHAR(50) NOT NULL,
    brain_type VARCHAR(20) NOT NULL,
    config_id VARCHAR(100) NOT NULL,
    training_start TIMESTAMP,
    training_end TIMESTAMP,
    accuracy_before DECIMAL(5,2),
    accuracy_after DECIMAL(5,2),
    improvement DECIMAL(5,2),
    status VARCHAR(20), -- 'completed', 'failed', 'in_progress'
    error_message TEXT,
    FOREIGN KEY (institution_id) REFERENCES institutions(id),
    FOREIGN KEY (config_id) REFERENCES institution_brain_configs(id)
);
```

---

## 🔧 Implementación Técnica

### 1. Backend API Endpoints

```python
# Configurar cerebro específico
POST /api/institutions/{institution_id}/brains/{brain_type}/configure
{
    "trading_params": {
        "stop_loss": 2.0,
        "take_profit": 4.0,
        "lot_size": 0.1,
        "max_drawdown": 10.0
    },
    "market_preferences": {
        "markets": ["forex", "crypto"],
        "timeframes": ["1h", "4h", "1d"],
        "risk_profile": "moderate"
    },
    "specializations": {
        "indicators": ["RSI", "MACD", "Bollinger"],
        "strategies": ["trend_following", "mean_reversion"],
        "custom_indicators": []
    },
    "consensus_weight": 0.35
}

# Entrenar cerebro con configuración existente
POST /api/institutions/{institution_id}/brains/{brain_type}/train
{
    "config_id": "config_123", // ID de la configuración a usar
    "training_params": {
        "epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 32
    }
}

# Obtener configuraciones de cerebros
GET /api/institutions/{institution_id}/brain-configs

# Obtener estado de entrenamiento
GET /api/institutions/{institution_id}/training-status

# Obtener modelos entrenados
GET /api/institutions/{institution_id}/trained-models

# Configurar consenso
PUT /api/institutions/{institution_id}/consensus-config
{
    "brain_max_weight": 0.35,
    "brain_ultra_weight": 0.40,
    "brain_predictor_weight": 0.25,
    "consensus_threshold": 0.75
}
```

### 2. Frontend Integration

```typescript
// Funcionalidad de botones en MegaMind.tsx
const handleConfigureBrain = async (brainType: string) => {
    try {
        // Abrir modal de configuración
        setConfigModalOpen(true);
        setSelectedBrain(brainType);
        
        // Cargar configuración existente si existe
        const existingConfig = await fetch(`/api/institutions/${institutionId}/brain-configs/${brainType}`);
        if (existingConfig.ok) {
            const config = await existingConfig.json();
            setBrainConfig(config);
        }
    } catch (error) {
        console.error('Error loading brain config:', error);
    }
};

const handleSaveConfiguration = async (brainType: string, config: any) => {
    try {
        const response = await fetch(`/api/institutions/${institutionId}/brains/${brainType}/configure`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            const savedConfig = await response.json();
            setConfigId(savedConfig.id);
            setConfigModalOpen(false);
            // Mostrar notificación de éxito
        }
    } catch (error) {
        console.error('Error saving configuration:', error);
    }
};

const handleTrainBrain = async (brainType: string) => {
    try {
        if (!configId) {
            alert('Debe configurar el cerebro antes de entrenarlo');
            return;
        }
        
        const response = await fetch(`/api/institutions/${institutionId}/brains/${brainType}/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                config_id: configId,
                training_params: {
                    epochs: 100,
                    learning_rate: 0.001
                }
            })
        });
        
        if (response.ok) {
            // Actualizar UI con progreso
            setTrainingStatus('in_progress');
            pollTrainingStatus();
        }
    } catch (error) {
        console.error('Error training brain:', error);
    }
};
```

### 3. Sistema de Entrenamiento (Algoritmos Existentes)

```python
class CollaborativeBrainTrainer:
    def __init__(self, institution_id: str):
        self.institution_id = institution_id
        self.s3_client = boto3.client('s3')
        self.db = Database()
    
    def train_brain_with_config(self, brain_type: str, config_id: str, training_params: dict) -> dict:
        """Entrena un cerebro usando configuración específica"""
        
        # 1. Obtener configuración de la base de datos
        config = self.db.get_brain_config(config_id)
        
        # 2. Descargar modelo base
        base_model = self.download_base_model(brain_type)
        
        # 3. Aplicar configuración personalizada
        customized_model = self.apply_configuration(base_model, config)
        
        # 4. Entrenar con algoritmos existentes
        trained_model = self.train_with_existing_algorithms(
            customized_model, 
            config, 
            training_params
        )
        
        # 5. Evaluar precisión
        accuracy = self.evaluate_model(trained_model)
        
        # 6. Subir a S3
        s3_path = self.upload_to_s3(trained_model, brain_type)
        
        # 7. Guardar metadatos en DB
        self.save_trained_model_metadata(brain_type, config_id, s3_path, accuracy)
        
        return {
            'status': 'completed',
            'accuracy': accuracy,
            's3_path': s3_path,
            'config_used': config_id
        }
    
    def apply_configuration(self, base_model, config: dict):
        """Aplica configuración personalizada al modelo base"""
        # Usar algoritmos existentes para personalizar
        return self.existing_algorithms.customize_model(base_model, config)
    
    def train_with_existing_algorithms(self, model, config: dict, training_params: dict):
        """Usa algoritmos de entrenamiento existentes"""
        # Los algoritmos ya están implementados
        return self.existing_algorithms.train_model(model, config, training_params)
```

### 4. Modal de Configuración

```typescript
// Componente de configuración de cerebro
const BrainConfigModal = ({ brainType, isOpen, onClose, onSave }) => {
    const [config, setConfig] = useState({
        trading_params: {
            stop_loss: 2.0,
            take_profit: 4.0,
            lot_size: 0.1,
            max_drawdown: 10.0
        },
        market_preferences: {
            markets: ['forex', 'crypto'],
            timeframes: ['1h', '4h', '1d'],
            risk_profile: 'moderate'
        },
        specializations: {
            indicators: ['RSI', 'MACD', 'Bollinger'],
            strategies: ['trend_following', 'mean_reversion'],
            custom_indicators: []
        },
        consensus_weight: 0.33
    });

    return (
        <Modal isOpen={isOpen} onClose={onClose}>
            <div className="bg-gray-800 rounded-lg p-6 max-w-2xl">
                <h3 className="text-xl font-semibold text-white mb-4">
                    Configurar {brainType}
                </h3>
                
                {/* Trading Parameters */}
                <div className="mb-6">
                    <h4 className="text-lg font-medium text-white mb-3">Parámetros de Trading</h4>
                    <div className="grid grid-cols-2 gap-4">
                        <div>
                            <label className="text-gray-300 text-sm">Stop Loss (%)</label>
                            <input 
                                type="number" 
                                value={config.trading_params.stop_loss}
                                onChange={(e) => setConfig({
                                    ...config,
                                    trading_params: {
                                        ...config.trading_params,
                                        stop_loss: parseFloat(e.target.value)
                                    }
                                })}
                                className="w-full bg-gray-700 text-white rounded px-3 py-2"
                            />
                        </div>
                        <div>
                            <label className="text-gray-300 text-sm">Take Profit (%)</label>
                            <input 
                                type="number" 
                                value={config.trading_params.take_profit}
                                onChange={(e) => setConfig({
                                    ...config,
                                    trading_params: {
                                        ...config.trading_params,
                                        take_profit: parseFloat(e.target.value)
                                    }
                                })}
                                className="w-full bg-gray-700 text-white rounded px-3 py-2"
                            />
                        </div>
                    </div>
                </div>

                {/* Market Preferences */}
                <div className="mb-6">
                    <h4 className="text-lg font-medium text-white mb-3">Preferencias de Mercado</h4>
                    <div className="space-y-3">
                        <div>
                            <label className="text-gray-300 text-sm">Mercados</label>
                            <select 
                                multiple
                                value={config.market_preferences.markets}
                                onChange={(e) => setConfig({
                                    ...config,
                                    market_preferences: {
                                        ...config.market_preferences,
                                        markets: Array.from(e.target.selectedOptions, option => option.value)
                                    }
                                })}
                                className="w-full bg-gray-700 text-white rounded px-3 py-2"
                            >
                                <option value="forex">Forex</option>
                                <option value="crypto">Crypto</option>
                                <option value="stocks">Stocks</option>
                                <option value="commodities">Commodities</option>
                            </select>
                        </div>
                        <div>
                            <label className="text-gray-300 text-sm">Perfil de Riesgo</label>
                            <select 
                                value={config.market_preferences.risk_profile}
                                onChange={(e) => setConfig({
                                    ...config,
                                    market_preferences: {
                                        ...config.market_preferences,
                                        risk_profile: e.target.value
                                    }
                                })}
                                className="w-full bg-gray-700 text-white rounded px-3 py-2"
                            >
                                <option value="conservative">Conservador</option>
                                <option value="moderate">Moderado</option>
                                <option value="aggressive">Agresivo</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Specializations */}
                <div className="mb-6">
                    <h4 className="text-lg font-medium text-white mb-3">Especializaciones</h4>
                    <div className="space-y-3">
                        <div>
                            <label className="text-gray-300 text-sm">Indicadores</label>
                            <select 
                                multiple
                                value={config.specializations.indicators}
                                onChange={(e) => setConfig({
                                    ...config,
                                    specializations: {
                                        ...config.specializations,
                                        indicators: Array.from(e.target.selectedOptions, option => option.value)
                                    }
                                })}
                                className="w-full bg-gray-700 text-white rounded px-3 py-2"
                            >
                                <option value="RSI">RSI</option>
                                <option value="MACD">MACD</option>
                                <option value="Bollinger">Bollinger Bands</option>
                                <option value="SMA">SMA</option>
                                <option value="EMA">EMA</option>
                            </select>
                        </div>
                    </div>
                </div>

                {/* Consensus Weight */}
                <div className="mb-6">
                    <h4 className="text-lg font-medium text-white mb-3">Peso en Consenso</h4>
                    <div>
                        <label className="text-gray-300 text-sm">Peso (%)</label>
                        <input 
                            type="range" 
                            min="0" 
                            max="100" 
                            value={config.consensus_weight * 100}
                            onChange={(e) => setConfig({
                                ...config,
                                consensus_weight: parseFloat(e.target.value) / 100
                            })}
                            className="w-full"
                        />
                        <span className="text-white">{Math.round(config.consensus_weight * 100)}%</span>
                    </div>
                </div>

                {/* Action Buttons */}
                <div className="flex justify-end space-x-3">
                    <button 
                        onClick={onClose}
                        className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-500"
                    >
                        Cancelar
                    </button>
                    <button 
                        onClick={() => onSave(brainType, config)}
                        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-500"
                    >
                        Guardar Configuración
                    </button>
                </div>
            </div>
        </Modal>
    );
};
```

---

## 📊 Gestión de Recursos

### 1. Tamaños de Archivos Estimados

```
Modelos Base:
├── Brain_Max: 150 MB
├── Brain_Ultra: 300 MB
└── Brain_Predictor: 200 MB

Modelos Entrenados (por institución):
├── Brain_Max_trained: 200 MB
├── Brain_Ultra_trained: 400 MB
└── Brain_Predictor_trained: 250 MB

Total por institución: 850 MB
Con 100 instituciones: 85 GB
```

### 2. Costos Estimados (AWS)

```
Almacenamiento S3:
├── 85 GB: $2.04/mes
├── Transferencia: $1.70/mes
└── Total almacenamiento: $3.74/mes

Computación (entrenamiento):
├── EC2 t3.large (2 vCPU, 8 GB RAM)
├── 1 hora por entrenamiento
├── 100 instituciones × 3 cerebros × 1 hora = 300 horas/mes
└── Costo computación: $30/mes

Total estimado: $33.74/mes
```

### 3. Optimizaciones

```
Compresión de Modelos:
├── Antes: 850 MB por institución
├── Después: 425 MB por institución
└── Ahorro: 50%

Caché Local:
├── Modelos más usados en servidor
├── Reducción de latencia
└── Menor costo de transferencia
```

---

## 🔒 Seguridad y Aislamiento

### 1. Autenticación por Institución

```python
class InstitutionAuth:
    def __init__(self):
        self.jwt_secret = os.getenv('JWT_SECRET')
    
    def verify_institution_access(self, institution_id: str, user_token: str) -> bool:
        """Verifica que el usuario pertenece a la institución"""
        try:
            payload = jwt.decode(user_token, self.jwt_secret, algorithms=['HS256'])
            return payload['institution_id'] == institution_id
        except:
            return False
    
    def get_institution_data_path(self, institution_id: str) -> str:
        """Genera path seguro para datos de institución"""
        return f"institutions/{institution_id}/"
```

### 2. Encriptación de Modelos

```python
class ModelEncryption:
    def __init__(self):
        self.encryption_key = os.getenv('MODEL_ENCRYPTION_KEY')
    
    def encrypt_model(self, model_path: str) -> str:
        """Encripta modelo antes de subir a S3"""
        with open(model_path, 'rb') as f:
            model_data = f.read()
        
        encrypted_data = Fernet(self.encryption_key).encrypt(model_data)
        
        encrypted_path = model_path + '.encrypted'
        with open(encrypted_path, 'wb') as f:
            f.write(encrypted_data)
        
        return encrypted_path
    
    def decrypt_model(self, encrypted_path: str) -> str:
        """Desencripta modelo para uso"""
        with open(encrypted_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = Fernet(self.encryption_key).decrypt(encrypted_data)
        
        decrypted_path = encrypted_path.replace('.encrypted', '_decrypted')
        with open(decrypted_path, 'wb') as f:
            f.write(decrypted_data)
        
        return decrypted_path
```

---

## 🚀 Fases de Implementación

### Fase 1: Infraestructura Base (Semana 1-2)
- [ ] Configurar AWS S3 bucket
- [ ] Crear tablas de base de datos
- [ ] Implementar autenticación por institución
- [ ] Configurar sistema de encriptación

### Fase 2: Backend API (Semana 3-4)
- [ ] Implementar endpoints de configuración
- [ ] Crear sistema de descarga/subida a S3
- [ ] Implementar tracking de progreso
- [ ] Crear sistema de manejo de errores

### Fase 3: Frontend Integration (Semana 5-6)
- [ ] Conectar botones "Configurar" y "Entrenar"
- [ ] Implementar modales de configuración
- [ ] Crear sistema de progreso en tiempo real
- [ ] Implementar notificaciones

### Fase 4: Testing y Optimización (Semana 7-8)
- [ ] Pruebas de carga con múltiples instituciones
- [ ] Optimización de rendimiento
- [ ] Pruebas de seguridad
- [ ] Documentación final

---

## 📈 Métricas de Éxito

### 1. Performance
- ✅ Tiempo de entrenamiento < 30 minutos por cerebro
- ✅ Precisión mejorada > 5% después del entrenamiento
- ✅ Disponibilidad del sistema > 99.9%

### 2. Escalabilidad
- ✅ Soporte para 100+ instituciones simultáneas
- ✅ Procesamiento paralelo de entrenamientos
- ✅ Caché inteligente de modelos

### 3. Seguridad
- ✅ Aislamiento completo entre instituciones
- ✅ Encriptación de todos los modelos
- ✅ Auditoría completa de accesos

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **Python**: FastAPI, scikit-learn, TensorFlow
- **Base de Datos**: PostgreSQL
- **Cloud Storage**: AWS S3
- **Autenticación**: JWT, OAuth2

### Frontend
- **React**: TypeScript, Tailwind CSS
- **Estado**: React Context, useState
- **HTTP**: Axios, Fetch API
- **UI**: Lucide React Icons

### DevOps
- **Contenedores**: Docker
- **Orquestación**: Kubernetes
- **CI/CD**: GitHub Actions
- **Monitoreo**: AWS CloudWatch

---

## 🧠 Parámetros de Algoritmos Compatibles

### Análisis de Algoritmos Existentes

Después de revisar los algoritmos de entrenamiento existentes, hemos identificado los parámetros configurables para cada cerebro:

#### Brain Max (Modelo_Brain_Max.py)
**Algoritmos utilizados:**
- **RandomForest**: `n_estimators`, `max_depth`, `min_samples_split`, `random_state`, `n_jobs`
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, `random_state`, `n_jobs`
- **LightGBM**: `n_estimators`, `max_depth`, `learning_rate`, `num_leaves`, `random_state`, `n_jobs`, `verbose`
- **LSTM** (opcional): `hidden_layer_sizes`, `max_iter`, `learning_rate_init`

#### Brain Ultra (Modelo_Brain_Ultra.py)
**Algoritmos utilizados:**
- **LightGBM**: `objective`, `num_leaves`, `learning_rate`, `n_estimators`, `device`, `num_threads`, `verbose`, `random_state`
- **XGBoost**: `n_estimators`, `max_depth`, `learning_rate`, `n_jobs`, `random_state`
- **CatBoost**: `iterations`, `depth`, `learning_rate`, `task_type`, `thread_count`, `silent`, `random_state`
- **RandomForest**: `n_estimators`, `max_depth`, `n_jobs`, `random_state`
- **GradientBoosting**: `n_estimators`, `max_depth`, `learning_rate`, `random_state`

#### Brain Predictor (Brain_predictor.py)
**Algoritmos utilizados:**
- **GradientBoostingRegressor**: `n_estimators`, `learning_rate`, `max_depth`, `random_state`
- **RandomForestRegressor**: `n_estimators`, `max_depth`, `random_state`

### Configuración Compatible

La configuración del modal ahora incluye parámetros específicos para cada algoritmo:

```typescript
algorithm_params: {
  // Brain Max - RandomForest, XGBoost, LightGBM
  random_forest: {
    n_estimators: 100,
    max_depth: 10,
    min_samples_split: 5,
    random_state: 42
  },
  xgboost: {
    n_estimators: 100,
    max_depth: 6,
    learning_rate: 0.1,
    random_state: 42
  },
  lightgbm: {
    n_estimators: 100,
    max_depth: 6,
    learning_rate: 0.1,
    num_leaves: 31,
    random_state: 42
  },
  // Brain Ultra - CatBoost, GradientBoosting
  catboost: {
    iterations: 200,
    depth: 10,
    learning_rate: 0.03,
    random_state: 42
  },
  gradient_boosting: {
    n_estimators: 150,
    max_depth: 8,
    learning_rate: 0.05,
    random_state: 42
  },
  // Brain Predictor - Forecasting
  forecast_horizons: [1, 3, 7, 14, 30],
  lstm_enabled: false
}
```

### Integración con Algoritmos Existentes

El sistema de entrenamiento utiliza los parámetros configurados para personalizar los algoritmos existentes:

1. **Cargar modelo base** del cerebro correspondiente
2. **Aplicar configuración** de parámetros de algoritmo
3. **Entrenar con datos** de la institución
4. **Guardar modelo personalizado** en S3
5. **Actualizar metadatos** en base de datos

---

## 📞 Contacto y Soporte

### Equipo de Desarrollo
- **Arquitecto**: [Nombre]
- **Backend**: [Nombre]
- **Frontend**: [Nombre]
- **DevOps**: [Nombre]

### Documentación
- **API Docs**: `/api/docs`
- **Guía de Usuario**: `/docs/user-guide`
- **Troubleshooting**: `/docs/troubleshooting`

---

*Documento creado: [Fecha]*
*Versión: 1.0*
*Última actualización: [Fecha]* 