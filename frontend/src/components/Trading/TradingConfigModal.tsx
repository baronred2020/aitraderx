import React, { useState, useEffect } from 'react';
import { X, Save, RefreshCw, AlertTriangle, Info } from 'lucide-react';

interface TradingConfig {
  // Configuración de órdenes
  defaultOrderAmount: number;
  defaultOrderType: 'market' | 'limit' | 'stop';
  autoSetStopLoss: boolean;
  defaultStopLossPercent: number;
  autoSetTakeProfit: boolean;
  defaultTakeProfitPercent: number;
  
  // Configuración de riesgo
  maxPositionSize: number;
  maxDailyLoss: number;
  maxOpenPositions: number;
  
  // Configuración de notificaciones
  enableNotifications: boolean;
  priceAlerts: boolean;
  orderExecutedAlerts: boolean;
  stopLossAlerts: boolean;
  
  // Configuración de interfaz
  autoRefreshInterval: number;
  showAdvancedOptions: boolean;
  defaultTimeframe: string;
  
  // Configuración de datos
  dataSource: 'yahoo' | 'fallback';
  updateFrequency: number;
}

interface TradingConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (config: TradingConfig) => void;
  currentConfig: TradingConfig;
}

const TradingConfigModal: React.FC<TradingConfigModalProps> = ({
  isOpen,
  onClose,
  onSave,
  currentConfig
}) => {
  const [config, setConfig] = useState<TradingConfig>(currentConfig);
  const [activeTab, setActiveTab] = useState<'orders' | 'risk' | 'notifications' | 'interface' | 'data'>('orders');
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    setConfig(currentConfig);
    setHasChanges(false);
  }, [currentConfig, isOpen]);

  const handleConfigChange = (key: keyof TradingConfig, value: any) => {
    setConfig(prev => ({ ...prev, [key]: value }));
    setHasChanges(true);
  };

  const handleSave = () => {
    onSave(config);
    setHasChanges(false);
    onClose();
  };

  const handleReset = () => {
    setConfig(currentConfig);
    setHasChanges(false);
  };

  if (!isOpen) return null;

  const tabs = [
    { id: 'orders', name: 'Órdenes', icon: '📋' },
    { id: 'risk', name: 'Riesgo', icon: '🛡️' },
    { id: 'notifications', name: 'Notificaciones', icon: '🔔' },
    { id: 'interface', name: 'Interfaz', icon: '⚙️' },
    { id: 'data', name: 'Datos', icon: '📊' }
  ] as const;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className="bg-gray-900 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden border border-gray-700">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-blue-500 rounded-lg flex items-center justify-center">
              <Info className="w-4 h-4 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">Configuración de Trading</h2>
              <p className="text-sm text-gray-400">Personaliza tu experiencia de trading virtual</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-gray-700">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center space-x-2 px-4 py-3 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-400 border-b-2 border-blue-400 bg-blue-500/10'
                  : 'text-gray-400 hover:text-gray-300'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.name}</span>
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-6 overflow-y-auto max-h-[60vh]">
          {activeTab === 'orders' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Configuración de Órdenes</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Cantidad por defecto
                    </label>
                    <input
                      type="number"
                      value={config.defaultOrderAmount}
                      onChange={(e) => handleConfigChange('defaultOrderAmount', parseFloat(e.target.value))}
                      className="w-full trading-input"
                      min="100"
                      max="100000"
                      step="100"
                    />
                    <p className="text-xs text-gray-400 mt-1">Cantidad predeterminada para nuevas órdenes</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Tipo de orden por defecto
                    </label>
                    <select
                      value={config.defaultOrderType}
                      onChange={(e) => handleConfigChange('defaultOrderType', e.target.value)}
                      className="w-full trading-input"
                    >
                      <option value="market">Mercado</option>
                      <option value="limit">Límite</option>
                      <option value="stop">Stop</option>
                    </select>
                  </div>

                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="autoSetStopLoss"
                      checked={config.autoSetStopLoss}
                      onChange={(e) => handleConfigChange('autoSetStopLoss', e.target.checked)}
                      className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="autoSetStopLoss" className="text-sm text-gray-300">
                      Configurar Stop Loss automáticamente
                    </label>
                  </div>

                  {config.autoSetStopLoss && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Stop Loss por defecto (%)
                      </label>
                      <input
                        type="number"
                        value={config.defaultStopLossPercent}
                        onChange={(e) => handleConfigChange('defaultStopLossPercent', parseFloat(e.target.value))}
                        className="w-full trading-input"
                        min="0.1"
                        max="10"
                        step="0.1"
                      />
                    </div>
                  )}

                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="autoSetTakeProfit"
                      checked={config.autoSetTakeProfit}
                      onChange={(e) => handleConfigChange('autoSetTakeProfit', e.target.checked)}
                      className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="autoSetTakeProfit" className="text-sm text-gray-300">
                      Configurar Take Profit automáticamente
                    </label>
                  </div>

                  {config.autoSetTakeProfit && (
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Take Profit por defecto (%)
                      </label>
                      <input
                        type="number"
                        value={config.defaultTakeProfitPercent}
                        onChange={(e) => handleConfigChange('defaultTakeProfitPercent', parseFloat(e.target.value))}
                        className="w-full trading-input"
                        min="0.1"
                        max="20"
                        step="0.1"
                      />
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'risk' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Gestión de Riesgo</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Tamaño máximo de posición (%)
                    </label>
                    <input
                      type="number"
                      value={config.maxPositionSize}
                      onChange={(e) => handleConfigChange('maxPositionSize', parseFloat(e.target.value))}
                      className="w-full trading-input"
                      min="1"
                      max="100"
                      step="1"
                    />
                    <p className="text-xs text-gray-400 mt-1">Porcentaje máximo del balance por posición</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Pérdida máxima diaria (%)
                    </label>
                    <input
                      type="number"
                      value={config.maxDailyLoss}
                      onChange={(e) => handleConfigChange('maxDailyLoss', parseFloat(e.target.value))}
                      className="w-full trading-input"
                      min="1"
                      max="50"
                      step="1"
                    />
                    <p className="text-xs text-gray-400 mt-1">Detener trading si se alcanza esta pérdida</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Máximo de posiciones abiertas
                    </label>
                    <input
                      type="number"
                      value={config.maxOpenPositions}
                      onChange={(e) => handleConfigChange('maxOpenPositions', parseInt(e.target.value))}
                      className="w-full trading-input"
                      min="1"
                      max="20"
                      step="1"
                    />
                  </div>
                </div>

                <div className="mt-6 p-4 bg-yellow-500/10 border border-yellow-500/20 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <AlertTriangle className="w-4 h-4 text-yellow-400" />
                    <span className="text-sm font-medium text-yellow-400">Importante</span>
                  </div>
                  <p className="text-xs text-yellow-300">
                    Estas configuraciones ayudan a gestionar el riesgo en el trading virtual. 
                    Recuerda que esto es solo para aprendizaje y práctica.
                  </p>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'notifications' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Notificaciones</h3>
                
                <div className="space-y-4">
                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="enableNotifications"
                      checked={config.enableNotifications}
                      onChange={(e) => handleConfigChange('enableNotifications', e.target.checked)}
                      className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="enableNotifications" className="text-sm text-gray-300">
                      Habilitar notificaciones
                    </label>
                  </div>

                  {config.enableNotifications && (
                    <div className="ml-7 space-y-3">
                      <div className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          id="priceAlerts"
                          checked={config.priceAlerts}
                          onChange={(e) => handleConfigChange('priceAlerts', e.target.checked)}
                          className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                        />
                        <label htmlFor="priceAlerts" className="text-sm text-gray-300">
                          Alertas de precio
                        </label>
                      </div>

                      <div className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          id="orderExecutedAlerts"
                          checked={config.orderExecutedAlerts}
                          onChange={(e) => handleConfigChange('orderExecutedAlerts', e.target.checked)}
                          className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                        />
                        <label htmlFor="orderExecutedAlerts" className="text-sm text-gray-300">
                          Órdenes ejecutadas
                        </label>
                      </div>

                      <div className="flex items-center space-x-3">
                        <input
                          type="checkbox"
                          id="stopLossAlerts"
                          checked={config.stopLossAlerts}
                          onChange={(e) => handleConfigChange('stopLossAlerts', e.target.checked)}
                          className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                        />
                        <label htmlFor="stopLossAlerts" className="text-sm text-gray-300">
                          Stop Loss activado
                        </label>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'interface' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Configuración de Interfaz</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Intervalo de actualización (segundos)
                    </label>
                    <select
                      value={config.autoRefreshInterval}
                      onChange={(e) => handleConfigChange('autoRefreshInterval', parseInt(e.target.value))}
                      className="w-full trading-input"
                    >
                      <option value={5}>5 segundos</option>
                      <option value={10}>10 segundos</option>
                      <option value={30}>30 segundos</option>
                      <option value={60}>1 minuto</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Timeframe por defecto
                    </label>
                    <select
                      value={config.defaultTimeframe}
                      onChange={(e) => handleConfigChange('defaultTimeframe', e.target.value)}
                      className="w-full trading-input"
                    >
                      <option value="1M">1 minuto</option>
                      <option value="5M">5 minutos</option>
                      <option value="15M">15 minutos</option>
                      <option value="1H">1 hora</option>
                      <option value="4H">4 horas</option>
                      <option value="1D">1 día</option>
                    </select>
                  </div>

                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="showAdvancedOptions"
                      checked={config.showAdvancedOptions}
                      onChange={(e) => handleConfigChange('showAdvancedOptions', e.target.checked)}
                      className="w-4 h-4 text-blue-500 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                    />
                    <label htmlFor="showAdvancedOptions" className="text-sm text-gray-300">
                      Mostrar opciones avanzadas
                    </label>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'data' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Configuración de Datos</h3>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Fuente de datos
                    </label>
                    <select
                      value={config.dataSource}
                      onChange={(e) => handleConfigChange('dataSource', e.target.value)}
                      className="w-full trading-input"
                    >
                      <option value="yahoo">Yahoo Finance (Recomendado)</option>
                      <option value="fallback">Datos simulados</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Frecuencia de actualización (segundos)
                    </label>
                    <select
                      value={config.updateFrequency}
                      onChange={(e) => handleConfigChange('updateFrequency', parseInt(e.target.value))}
                      className="w-full trading-input"
                    >
                      <option value={5}>5 segundos</option>
                      <option value={10}>10 segundos</option>
                      <option value={30}>30 segundos</option>
                      <option value={60}>1 minuto</option>
                    </select>
                  </div>
                </div>

                <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
                  <div className="flex items-center space-x-2 mb-2">
                    <Info className="w-4 h-4 text-blue-400" />
                    <span className="text-sm font-medium text-blue-400">Información</span>
                  </div>
                  <p className="text-xs text-blue-300">
                    Los datos de Yahoo Finance proporcionan precios reales del mercado. 
                    Los datos simulados se usan cuando el mercado está cerrado o hay problemas de conexión.
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-6 border-t border-gray-700">
          <button
            onClick={handleReset}
            className="flex items-center space-x-2 px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Restablecer</span>
          </button>

          <div className="flex items-center space-x-3">
            <button
              onClick={onClose}
              className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
            >
              Cancelar
            </button>
            <button
              onClick={handleSave}
              disabled={!hasChanges}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                hasChanges
                  ? 'bg-blue-500 hover:bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-400 cursor-not-allowed'
              }`}
            >
              <Save className="w-4 h-4" />
              <span>Guardar</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingConfigModal; 