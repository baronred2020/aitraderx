import React, { useState } from 'react';

interface StrategyConfigProps {
  onClose: () => void;
  onCreate: (strategy: any) => void;
  isLoading: boolean;
}

interface StrategyForm {
  name: string;
  brainType: string;
  pair: string;
  style: string;
  lotSize: number;
  stopLossPips: number;
  takeProfitPips: number;
  minConfidence: number;
  maxPositions: number;
  riskPerTrade: number;
}

const StrategyConfig: React.FC<StrategyConfigProps> = ({ onClose, onCreate, isLoading }) => {
  const [form, setForm] = useState<StrategyForm>({
    name: '',
    brainType: 'Brain_Ultra',
    pair: 'EURUSD',
    style: 'scalping',
    lotSize: 0.1,
    stopLossPips: 20,
    takeProfitPips: 40,
    minConfidence: 75,
    maxPositions: 1,
    riskPerTrade: 2
  });

  const brainTypes = [
    { id: 'Brain_Ultra', name: 'Brain Ultra', description: 'Cerebro especializado en scalping y day trading' },
    { id: 'Brain_Max', name: 'Brain Max', description: 'Cerebro avanzado con análisis técnico completo' },
    { id: 'Brain_Predictor', name: 'Brain Predictor', description: 'Cerebro predictivo con machine learning' },
    { id: 'Mega_Mind', name: 'Mega Mind', description: 'Cerebro de alto rendimiento para swing trading' }
  ];

  const pairs = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'NZDUSD', 'USDCAD', 'EURGBP'
  ];

  const styles = [
    { id: 'scalping', name: 'Scalping', description: 'Operaciones rápidas de 1-5 minutos' },
    { id: 'day_trading', name: 'Day Trading', description: 'Operaciones intradiarias' },
    { id: 'swing_trading', name: 'Swing Trading', description: 'Operaciones de varios días' }
  ];

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onCreate(form);
  };

  const handleInputChange = (field: keyof StrategyForm, value: any) => {
    setForm(prev => ({ ...prev, [field]: value }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-bold text-white">⚙️ Configurar Nueva Estrategia</h3>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Basic Information */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Nombre de la Estrategia
                </label>
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => handleInputChange('name', e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white placeholder-gray-400 focus:outline-none focus:border-blue-500"
                  placeholder="Ej: Scalping EURUSD Brain Ultra"
                  required
                />
              </div>

              {/* Brain Type Selection */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Cerebro IA
                </label>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {brainTypes.map((brain) => (
                    <div
                      key={brain.id}
                      className={`p-4 rounded-lg border-2 cursor-pointer transition-colors ${
                        form.brainType === brain.id
                          ? 'border-blue-500 bg-blue-500/10'
                          : 'border-gray-600 bg-gray-700 hover:border-gray-500'
                      }`}
                      onClick={() => handleInputChange('brainType', brain.id)}
                    >
                      <div className="font-medium text-white">{brain.name}</div>
                      <div className="text-sm text-gray-400">{brain.description}</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Pair and Style Selection */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Par de Divisas
                  </label>
                  <select
                    value={form.pair}
                    onChange={(e) => handleInputChange('pair', e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  >
                    {pairs.map((pair) => (
                      <option key={pair} value={pair}>{pair}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Estilo de Trading
                  </label>
                  <div className="space-y-2">
                    {styles.map((style) => (
                      <div
                        key={style.id}
                        className={`p-3 rounded-lg border-2 cursor-pointer transition-colors ${
                          form.style === style.id
                            ? 'border-blue-500 bg-blue-500/10'
                            : 'border-gray-600 bg-gray-700 hover:border-gray-500'
                        }`}
                        onClick={() => handleInputChange('style', style.id)}
                      >
                        <div className="font-medium text-white">{style.name}</div>
                        <div className="text-sm text-gray-400">{style.description}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Risk Management */}
            <div className="bg-gray-700 rounded-lg p-4">
              <h4 className="text-lg font-medium text-white mb-4">🛡️ Gestión de Riesgo</h4>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Tamaño de Lote
                  </label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.01"
                    max="10"
                    value={form.lotSize}
                    onChange={(e) => handleInputChange('lotSize', parseFloat(e.target.value))}
                    className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Riesgo por Trade (%)
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0.1"
                    max="10"
                    value={form.riskPerTrade}
                    onChange={(e) => handleInputChange('riskPerTrade', parseFloat(e.target.value))}
                    className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Stop Loss (Pips)
                  </label>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    max="1000"
                    value={form.stopLossPips}
                    onChange={(e) => handleInputChange('stopLossPips', parseInt(e.target.value))}
                    className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Take Profit (Pips)
                  </label>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    max="1000"
                    value={form.takeProfitPips}
                    onChange={(e) => handleInputChange('takeProfitPips', parseInt(e.target.value))}
                    className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Confianza Mínima (%)
                  </label>
                  <input
                    type="number"
                    step="1"
                    min="50"
                    max="100"
                    value={form.minConfidence}
                    onChange={(e) => handleInputChange('minConfidence', parseInt(e.target.value))}
                    className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Máx. Posiciones Simultáneas
                  </label>
                  <input
                    type="number"
                    step="1"
                    min="1"
                    max="10"
                    value={form.maxPositions}
                    onChange={(e) => handleInputChange('maxPositions', parseInt(e.target.value))}
                    className="w-full bg-gray-600 border border-gray-500 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Risk/Reward Ratio Display */}
              <div className="mt-4 p-3 bg-gray-600 rounded-lg">
                <div className="text-sm text-gray-300">
                  Ratio Riesgo/Recompensa: <span className="text-white font-medium">
                    {form.takeProfitPips / form.stopLossPips}:1
                  </span>
                </div>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex space-x-4 pt-4">
              <button
                type="button"
                onClick={onClose}
                className="flex-1 bg-gray-600 hover:bg-gray-700 text-white py-3 px-6 rounded-lg font-medium transition-colors"
              >
                Cancelar
              </button>
              <button
                type="submit"
                disabled={isLoading}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white py-3 px-6 rounded-lg font-medium transition-colors flex items-center justify-center space-x-2"
              >
                {isLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                    <span>Creando...</span>
                  </>
                ) : (
                  <>
                    <span>🚀</span>
                    <span>Crear Estrategia</span>
                  </>
                )}
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default StrategyConfig; 