import React, { useState, useEffect } from 'react';
import { 
  Brain, 
  Users, 
  Zap, 
  Trophy, 
  TrendingUp, 
  Target, 
  Settings, 
  RotateCcw,
  Crown,
  Star,
  Activity,
  BarChart3,
  X,
  Save,
  Edit3
} from 'lucide-react';

interface Brain {
  id: string;
  name: string;
  type: 'max' | 'ultra' | 'predictor';
  description: string;
  accuracy: number;
  status: 'active' | 'training' | 'inactive';
  icon: React.ComponentType<{ className?: string }>;
}

interface ConsensusVote {
  brainId: string;
  vote: 'buy' | 'sell' | 'hold';
  confidence: number;
  timestamp: Date;
}

interface BrainArena {
  brainId: string;
  performance: number;
  wins: number;
  losses: number;
  rank: number;
}

interface BrainEvolution {
  brainId: string;
  generation: number;
  fitness: number;
  mutations: string[];
}

interface BrainConfig {
  trading_params: {
    stop_loss: number;
    take_profit: number;
    lot_size: number;
    max_drawdown: number;
  };
  trading_styles: string[];
  market_preferences: {
    markets: string[];
    timeframes: string[];
    risk_profile: 'conservative' | 'moderate' | 'aggressive';
  };
  specializations: {
    indicators: string[];
    strategies: string[];
    custom_indicators: string[];
  };
  algorithm_params: {
    // Brain Max - Modelos de Clasificación
    random_forest: {
      n_estimators: number;
      max_depth: number;
      min_samples_split: number;
      random_state: number;
    };
    xgboost: {
      n_estimators: number;
      max_depth: number;
      learning_rate: number;
      random_state: number;
    };
    lightgbm: {
      n_estimators: number;
      max_depth: number;
      learning_rate: number;
      num_leaves: number;
      random_state: number;
    };
    // Brain Ultra - Modelos de Regresión
    lightgbm_regressor: {
      n_estimators: number;
      max_depth: number;
      learning_rate: number;
      num_leaves: number;
      random_state: number;
    };
    xgboost_regressor: {
      n_estimators: number;
      max_depth: number;
      learning_rate: number;
      random_state: number;
    };
    catboost_regressor: {
      iterations: number;
      depth: number;
      learning_rate: number;
      random_state: number;
    };
    random_forest_regressor: {
      n_estimators: number;
      max_depth: number;
      min_samples_split: number;
      random_state: number;
    };
    gradient_boosting_regressor: {
      n_estimators: number;
      max_depth: number;
      learning_rate: number;
      random_state: number;
    };
    // Brain Predictor - Solo Modelos Originales
    gradient_boosting_predictor: {
      n_estimators: number;
      learning_rate: number;
      max_depth: number;
      random_state: number;
    };
    random_forest_predictor: {
      n_estimators: number;
      max_depth: number;
      random_state: number;
    };
    forecast_horizons: number[];
  };
  consensus_weight: number;
}

interface BrainConfigModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSave: (config: BrainConfig) => void;
  selectedBrain: Brain | null;
  initialConfig?: BrainConfig;
}

const BrainConfigModal: React.FC<BrainConfigModalProps> = ({
  isOpen,
  onClose,
  onSave,
  selectedBrain,
  initialConfig
}) => {
  const [config, setConfig] = useState<BrainConfig>({
    trading_params: {
      stop_loss: 50,
      take_profit: 100,
      lot_size: 0.1,
      max_drawdown: 20
    },
    trading_styles: ['trend_following', 'mean_reversion'],
    market_preferences: {
      markets: ['forex', 'crypto'],
      timeframes: ['1h', '4h'],
      risk_profile: 'moderate'
    },
    specializations: {
      indicators: ['RSI', 'MACD', 'Bollinger'],
      strategies: ['trend_following', 'mean_reversion'],
      custom_indicators: []
    },
    algorithm_params: {
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
          // Brain Ultra - Modelos de Regresión
    lightgbm_regressor: {
      n_estimators: 100,
      max_depth: 6,
      learning_rate: 0.1,
      num_leaves: 31,
      random_state: 42
    },
    xgboost_regressor: {
      n_estimators: 100,
      max_depth: 6,
      learning_rate: 0.1,
      random_state: 42
    },
    catboost_regressor: {
      iterations: 200,
      depth: 10,
      learning_rate: 0.03,
      random_state: 42
    },
    random_forest_regressor: {
      n_estimators: 100,
      max_depth: 10,
      min_samples_split: 5,
      random_state: 42
    },
    gradient_boosting_regressor: {
      n_estimators: 150,
      max_depth: 8,
      learning_rate: 0.05,
      random_state: 42
    },
    // Brain Predictor - Solo Modelos Originales
    gradient_boosting_predictor: {
      n_estimators: 100,
      learning_rate: 0.1,
      max_depth: 5,
      random_state: 42
    },
    random_forest_predictor: {
      n_estimators: 100,
      max_depth: 5,
      random_state: 42
    },
      forecast_horizons: [1, 3, 7, 14, 30]
    },
    consensus_weight: 0.33
  });

  useEffect(() => {
    if (initialConfig) {
      setConfig(initialConfig);
    }
  }, [initialConfig]);

  const tradingStyles = [
    { id: 'trend_following', name: 'Trend Following', description: 'Seguir tendencias del mercado' },
    { id: 'mean_reversion', name: 'Mean Reversion', description: 'Reversión a la media' },
    { id: 'scalping', name: 'Scalping', description: 'Operaciones de corta duración' },
    { id: 'swing_trading', name: 'Swing Trading', description: 'Operaciones de mediano plazo' }
  ];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold text-white">
            Configurar {selectedBrain?.name}
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-6">
          {/* Trading Parameters */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-white mb-3">Parámetros de Trading</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-gray-300 text-xs">Stop Loss (pips)</label>
                <input 
                  type="number"
                  value={config.trading_params.stop_loss}
                  onChange={(e) => setConfig({
                    ...config,
                    trading_params: {
                      ...config.trading_params,
                      stop_loss: parseInt(e.target.value)
                    }
                  })}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                />
              </div>
              <div>
                <label className="text-gray-300 text-xs">Take Profit (pips)</label>
                <input 
                  type="number"
                  value={config.trading_params.take_profit}
                  onChange={(e) => setConfig({
                    ...config,
                    trading_params: {
                      ...config.trading_params,
                      take_profit: parseInt(e.target.value)
                    }
                  })}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                />
              </div>
              <div>
                <label className="text-gray-300 text-xs">Lot Size</label>
                <input 
                  type="number"
                  step="0.01"
                  value={config.trading_params.lot_size}
                  onChange={(e) => setConfig({
                    ...config,
                    trading_params: {
                      ...config.trading_params,
                      lot_size: parseFloat(e.target.value)
                    }
                  })}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                />
              </div>
              <div>
                <label className="text-gray-300 text-xs">Max Drawdown (%)</label>
                <input 
                  type="number"
                  value={config.trading_params.max_drawdown}
                  onChange={(e) => setConfig({
                    ...config,
                    trading_params: {
                      ...config.trading_params,
                      max_drawdown: parseInt(e.target.value)
                    }
                  })}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                />
              </div>
            </div>
          </div>

          {/* Trading Styles */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-white mb-3">Estilos de Trading</h4>
            <div className="grid grid-cols-2 gap-2">
              {tradingStyles.map((style) => (
                <label key={style.id} className="flex items-center space-x-2">
                  <input 
                    type="checkbox"
                    checked={config.trading_styles.includes(style.id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setConfig({
                          ...config,
                          trading_styles: [...config.trading_styles, style.id]
                        });
                      } else {
                        setConfig({
                          ...config,
                          trading_styles: config.trading_styles.filter(s => s !== style.id)
                        });
                      }
                    }}
                    className="text-purple-500"
                  />
                  <span className="text-white text-sm">{style.name}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Market Preferences */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-white mb-3">Preferencias de Mercado</h4>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="text-gray-300 text-xs">Mercados</label>
                <select 
                  multiple
                  value={config.market_preferences.markets}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions, option => option.value);
                    setConfig({
                      ...config,
                      market_preferences: {
                        ...config.market_preferences,
                        markets: selected
                      }
                    });
                  }}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                >
                  <option value="forex">Forex</option>
                  <option value="crypto">Crypto</option>
                  <option value="stocks">Stocks</option>
                  <option value="commodities">Commodities</option>
                  <option value="indices">Indices</option>
                </select>
              </div>
              <div>
                <label className="text-gray-300 text-xs">Timeframes</label>
                <select 
                  multiple
                  value={config.market_preferences.timeframes}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions, option => option.value);
                    setConfig({
                      ...config,
                      market_preferences: {
                        ...config.market_preferences,
                        timeframes: selected
                      }
                    });
                  }}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                >
                  <option value="1m">1m</option>
                  <option value="5m">5m</option>
                  <option value="15m">15m</option>
                  <option value="1h">1h</option>
                  <option value="4h">4h</option>
                  <option value="1d">1d</option>
                </select>
              </div>
              <div>
                <label className="text-gray-300 text-xs">Risk Profile</label>
                <select 
                  value={config.market_preferences.risk_profile}
                  onChange={(e) => setConfig({
                    ...config,
                    market_preferences: {
                      ...config.market_preferences,
                      risk_profile: e.target.value as 'conservative' | 'moderate' | 'aggressive'
                    }
                  })}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                >
                  <option value="conservative">Conservative</option>
                  <option value="moderate">Moderate</option>
                  <option value="aggressive">Aggressive</option>
                </select>
              </div>
            </div>
          </div>

          {/* Specializations */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-white mb-3">Especializaciones</h4>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-gray-300 text-xs">Indicadores Técnicos</label>
                <select 
                  multiple
                  value={config.specializations.indicators}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions, option => option.value);
                    setConfig({
                      ...config,
                      specializations: {
                        ...config.specializations,
                        indicators: selected
                      }
                    });
                  }}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                >
                  <option value="RSI">RSI</option>
                  <option value="MACD">MACD</option>
                  <option value="Bollinger">Bollinger</option>
                  <option value="SMA">SMA</option>
                  <option value="EMA">EMA</option>
                  <option value="Stochastic">Stochastic</option>
                  <option value="Williams %R">Williams %R</option>
                  <option value="CCI">CCI</option>
                  <option value="ATR">ATR</option>
                  <option value="ADX">ADX</option>
                </select>
              </div>
              <div>
                <label className="text-gray-300 text-xs">Estrategias</label>
                <select 
                  multiple
                  value={config.specializations.strategies}
                  onChange={(e) => {
                    const selected = Array.from(e.target.selectedOptions, option => option.value);
                    setConfig({
                      ...config,
                      specializations: {
                        ...config.specializations,
                        strategies: selected
                      }
                    });
                  }}
                  className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                >
                  <option value="trend_following">Trend Following</option>
                  <option value="mean_reversion">Mean Reversion</option>
                  <option value="breakout_trading">Breakout Trading</option>
                  <option value="scalping">Scalping</option>
                  <option value="swing_trading">Swing Trading</option>
                  <option value="grid_trading">Grid Trading</option>
                  <option value="arbitrage">Arbitrage</option>
                  <option value="hedging">Hedging</option>
                </select>
              </div>
            </div>
          </div>

          {/* Algorithm Parameters */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-white mb-3">Parámetros de Algoritmos</h4>
            <div className="grid grid-cols-2 gap-4">
              {/* Brain Max - Modelos de Clasificación */}
              {selectedBrain?.type === 'max' && (
                <>
                  {/* Random Forest */}
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <h5 className="text-white font-medium mb-2">Random Forest</h5>
                    <div className="space-y-2">
                      <div>
                        <label className="text-gray-300 text-xs">N Estimators</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.random_forest.n_estimators}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              random_forest: {
                                ...config.algorithm_params.random_forest,
                                n_estimators: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Max Depth</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.random_forest.max_depth}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              random_forest: {
                                ...config.algorithm_params.random_forest,
                                max_depth: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Min Samples Split</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.random_forest.min_samples_split}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              random_forest: {
                                ...config.algorithm_params.random_forest,
                                min_samples_split: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                    </div>
                  </div>
                  {/* XGBoost */}
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <h5 className="text-white font-medium mb-2">XGBoost</h5>
                    <div className="space-y-2">
                      <div>
                        <label className="text-gray-300 text-xs">N Estimators</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.xgboost.n_estimators}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              xgboost: {
                                ...config.algorithm_params.xgboost,
                                n_estimators: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Max Depth</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.xgboost.max_depth}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              xgboost: {
                                ...config.algorithm_params.xgboost,
                                max_depth: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Learning Rate</label>
                        <input 
                          type="number"
                          step="0.01"
                          value={config.algorithm_params.xgboost.learning_rate}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              xgboost: {
                                ...config.algorithm_params.xgboost,
                                learning_rate: parseFloat(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                    </div>
                  </div>

                  {/* LightGBM */}
                  <div className="bg-gray-700/50 rounded-lg p-4">
                    <h5 className="text-white font-medium mb-2">LightGBM</h5>
                    <div className="space-y-2">
                      <div>
                        <label className="text-gray-300 text-xs">N Estimators</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.lightgbm.n_estimators}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              lightgbm: {
                                ...config.algorithm_params.lightgbm,
                                n_estimators: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Max Depth</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.lightgbm.max_depth}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              lightgbm: {
                                ...config.algorithm_params.lightgbm,
                                max_depth: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Learning Rate</label>
                        <input 
                          type="number"
                          step="0.01"
                          value={config.algorithm_params.lightgbm.learning_rate}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              lightgbm: {
                                ...config.algorithm_params.lightgbm,
                                learning_rate: parseFloat(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                      <div>
                        <label className="text-gray-300 text-xs">Num Leaves</label>
                        <input 
                          type="number"
                          value={config.algorithm_params.lightgbm.num_leaves}
                          onChange={(e) => setConfig({
                            ...config,
                            algorithm_params: {
                              ...config.algorithm_params,
                              lightgbm: {
                                ...config.algorithm_params.lightgbm,
                                num_leaves: parseInt(e.target.value)
                              }
                            }
                          })}
                          className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                        />
                      </div>
                    </div>
                  </div>
                </>
              )}

              {/* Brain Ultra - Modelos de Regresión */}
              {selectedBrain?.type === 'ultra' && (
                <>
                  {/* LightGBM Regressor */}
                  <div className="bg-gray-700/50 rounded-lg p-4">
                 <h5 className="text-white font-medium mb-2">LightGBM Regressor</h5>
                 <div className="space-y-2">
                   <div>
                     <label className="text-gray-300 text-xs">N Estimators</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.lightgbm_regressor.n_estimators}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           lightgbm_regressor: {
                             ...config.algorithm_params.lightgbm_regressor,
                             n_estimators: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Max Depth</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.lightgbm_regressor.max_depth}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           lightgbm_regressor: {
                             ...config.algorithm_params.lightgbm_regressor,
                             max_depth: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Learning Rate</label>
                     <input 
                       type="number"
                       step="0.01"
                       value={config.algorithm_params.lightgbm_regressor.learning_rate}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           lightgbm_regressor: {
                             ...config.algorithm_params.lightgbm_regressor,
                             learning_rate: parseFloat(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Num Leaves</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.lightgbm_regressor.num_leaves}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           lightgbm_regressor: {
                             ...config.algorithm_params.lightgbm_regressor,
                             num_leaves: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                 </div>
               </div>

               {/* XGBoost Regressor */}
               <div className="bg-gray-700/50 rounded-lg p-4">
                 <h5 className="text-white font-medium mb-2">XGBoost Regressor</h5>
                 <div className="space-y-2">
                   <div>
                     <label className="text-gray-300 text-xs">N Estimators</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.xgboost_regressor.n_estimators}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           xgboost_regressor: {
                             ...config.algorithm_params.xgboost_regressor,
                             n_estimators: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Max Depth</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.xgboost_regressor.max_depth}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           xgboost_regressor: {
                             ...config.algorithm_params.xgboost_regressor,
                             max_depth: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Learning Rate</label>
                     <input 
                       type="number"
                       step="0.01"
                       value={config.algorithm_params.xgboost_regressor.learning_rate}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           xgboost_regressor: {
                             ...config.algorithm_params.xgboost_regressor,
                             learning_rate: parseFloat(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                 </div>
               </div>

               {/* CatBoost Regressor */}
               <div className="bg-gray-700/50 rounded-lg p-4">
                 <h5 className="text-white font-medium mb-2">CatBoost Regressor</h5>
                 <div className="space-y-2">
                   <div>
                     <label className="text-gray-300 text-xs">Iterations</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.catboost_regressor.iterations}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           catboost_regressor: {
                             ...config.algorithm_params.catboost_regressor,
                             iterations: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Depth</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.catboost_regressor.depth}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           catboost_regressor: {
                             ...config.algorithm_params.catboost_regressor,
                             depth: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Learning Rate</label>
                     <input 
                       type="number"
                       step="0.01"
                       value={config.algorithm_params.catboost_regressor.learning_rate}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           catboost_regressor: {
                             ...config.algorithm_params.catboost_regressor,
                             learning_rate: parseFloat(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                 </div>
               </div>

               {/* Random Forest Regressor */}
               <div className="bg-gray-700/50 rounded-lg p-4">
                 <h5 className="text-white font-medium mb-2">Random Forest Regressor</h5>
                 <div className="space-y-2">
                   <div>
                     <label className="text-gray-300 text-xs">N Estimators</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.random_forest_regressor.n_estimators}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           random_forest_regressor: {
                             ...config.algorithm_params.random_forest_regressor,
                             n_estimators: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Max Depth</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.random_forest_regressor.max_depth}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           random_forest_regressor: {
                             ...config.algorithm_params.random_forest_regressor,
                             max_depth: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                   <div>
                     <label className="text-gray-300 text-xs">Min Samples Split</label>
                     <input 
                       type="number"
                       value={config.algorithm_params.random_forest_regressor.min_samples_split}
                       onChange={(e) => setConfig({
                         ...config,
                         algorithm_params: {
                           ...config.algorithm_params,
                           random_forest_regressor: {
                             ...config.algorithm_params.random_forest_regressor,
                             min_samples_split: parseInt(e.target.value)
                           }
                         }
                       })}
                       className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                     />
                   </div>
                 </div>
               </div>

                   {/* Gradient Boosting Regressor */}
                   <div className="bg-gray-700/50 rounded-lg p-4">
                     <h5 className="text-white font-medium mb-2">Gradient Boosting Regressor</h5>
                     <div className="space-y-2">
                       <div>
                         <label className="text-gray-300 text-xs">N Estimators</label>
                         <input 
                           type="number"
                           value={config.algorithm_params.gradient_boosting_regressor.n_estimators}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               gradient_boosting_regressor: {
                                 ...config.algorithm_params.gradient_boosting_regressor,
                                 n_estimators: parseInt(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                       <div>
                         <label className="text-gray-300 text-xs">Max Depth</label>
                         <input 
                           type="number"
                           value={config.algorithm_params.gradient_boosting_regressor.max_depth}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               gradient_boosting_regressor: {
                                 ...config.algorithm_params.gradient_boosting_regressor,
                                 max_depth: parseInt(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                       <div>
                         <label className="text-gray-300 text-xs">Learning Rate</label>
                         <input 
                           type="number"
                           step="0.01"
                           value={config.algorithm_params.gradient_boosting_regressor.learning_rate}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               gradient_boosting_regressor: {
                                 ...config.algorithm_params.gradient_boosting_regressor,
                                 learning_rate: parseFloat(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                     </div>
                   </div>
                 </>
               )}

               {/* Brain Predictor - Solo Modelos Originales */}
               {selectedBrain?.type === 'predictor' && (
                 <>
                   <div className="bg-gray-700/50 rounded-lg p-4">
                     <h5 className="text-white font-medium mb-2">Brain Predictor - Gradient Boosting (Precio)</h5>
                     <div className="space-y-2">
                       <div>
                         <label className="text-gray-300 text-xs">N Estimators</label>
                         <input 
                           type="number"
                           value={config.algorithm_params.gradient_boosting_predictor.n_estimators}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               gradient_boosting_predictor: {
                                 ...config.algorithm_params.gradient_boosting_predictor,
                                 n_estimators: parseInt(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                       <div>
                         <label className="text-gray-300 text-xs">Learning Rate</label>
                         <input 
                           type="number"
                           step="0.01"
                           value={config.algorithm_params.gradient_boosting_predictor.learning_rate}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               gradient_boosting_predictor: {
                                 ...config.algorithm_params.gradient_boosting_predictor,
                                 learning_rate: parseFloat(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                       <div>
                         <label className="text-gray-300 text-xs">Max Depth</label>
                         <input 
                           type="number"
                           value={config.algorithm_params.gradient_boosting_predictor.max_depth}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               gradient_boosting_predictor: {
                                 ...config.algorithm_params.gradient_boosting_predictor,
                                 max_depth: parseInt(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                     </div>
                   </div>

                   <div className="bg-gray-700/50 rounded-lg p-4">
                     <h5 className="text-white font-medium mb-2">Brain Predictor - Random Forest (Dirección)</h5>
                     <div className="space-y-2">
                       <div>
                         <label className="text-gray-300 text-xs">N Estimators</label>
                         <input 
                           type="number"
                           value={config.algorithm_params.random_forest_predictor.n_estimators}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               random_forest_predictor: {
                                 ...config.algorithm_params.random_forest_predictor,
                                 n_estimators: parseInt(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                       <div>
                         <label className="text-gray-300 text-xs">Max Depth</label>
                         <input 
                           type="number"
                           value={config.algorithm_params.random_forest_predictor.max_depth}
                           onChange={(e) => setConfig({
                             ...config,
                             algorithm_params: {
                               ...config.algorithm_params,
                               random_forest_predictor: {
                                 ...config.algorithm_params.random_forest_predictor,
                                 max_depth: parseInt(e.target.value)
                               }
                             }
                           })}
                           className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
                         />
                       </div>
                     </div>
                   </div>
                 </>
               )}
            </div>
          </div>

          {/* Consensus Weight */}
          <div className="mb-6">
            <h4 className="text-lg font-medium text-white mb-3">Peso de Consenso</h4>
            <div>
              <label className="text-gray-300 text-xs">Peso en Votación (0.0 - 1.0)</label>
              <input 
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={config.consensus_weight}
                onChange={(e) => setConfig({
                  ...config,
                  consensus_weight: parseFloat(e.target.value)
                })}
                className="w-full bg-gray-600 text-white rounded px-2 py-1 text-sm"
              />
            </div>
          </div>
        </div>

        <div className="flex justify-end space-x-3 mt-6">
          <button 
            onClick={onClose}
            className="px-4 py-2 text-gray-300 hover:text-white"
          >
            Cancelar
          </button>
          <button 
            onClick={() => onSave(config)}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 flex items-center space-x-2"
          >
            <Save className="w-4 h-4" />
            <span>Guardar Configuración</span>
          </button>
        </div>
      </div>
    </div>
  );
};

const MegaMind: React.FC = () => {
  const [activeTab, setActiveTab] = useState('collaboration');
  const [autoTrading, setAutoTrading] = useState(false);
  const [autoTraining, setAutoTraining] = useState(false);
  const [configModalOpen, setConfigModalOpen] = useState(false);
  const [selectedBrain, setSelectedBrain] = useState<Brain | null>(null);
  const [brainConfigs, setBrainConfigs] = useState<{[key: string]: any}>({});
  const [trainingStatus, setTrainingStatus] = useState<{[key: string]: string}>({});
  const [configId, setConfigId] = useState<string | null>(null);

  const brains: Brain[] = [
    {
      id: 'brain-max',
      name: 'Brain Max',
      type: 'max',
      description: 'Análisis técnico + patrones de mercado',
      accuracy: 87.5,
      status: 'active',
      icon: Brain
    },
    {
      id: 'brain-ultra',
      name: 'Brain Ultra',
      type: 'ultra',
      description: 'Multi-estrategia + adaptación dinámica',
      accuracy: 89.2,
      status: 'active',
      icon: Zap
    },
    {
      id: 'brain-predictor',
      name: 'Brain Predictor',
      type: 'predictor',
      description: 'Forecasting + eventos económicos',
      accuracy: 85.8,
      status: 'active',
      icon: TrendingUp
    }
  ];

  const consensusVotes: ConsensusVote[] = [
    { brainId: 'brain-max', vote: 'buy', confidence: 0.87, timestamp: new Date() },
    { brainId: 'brain-ultra', vote: 'buy', confidence: 0.89, timestamp: new Date() },
    { brainId: 'brain-predictor', vote: 'hold', confidence: 0.76, timestamp: new Date() }
  ];

  const brainArena: BrainArena[] = [
    { brainId: 'brain-max', performance: 87.5, wins: 45, losses: 12, rank: 1 },
    { brainId: 'brain-ultra', performance: 89.2, wins: 52, losses: 8, rank: 2 },
    { brainId: 'brain-predictor', performance: 85.8, wins: 38, losses: 15, rank: 3 }
  ];

  const brainEvolution: BrainEvolution[] = [
    { brainId: 'brain-max', generation: 15, fitness: 0.875, mutations: ['Optimized RSI', 'Enhanced MACD'] },
    { brainId: 'brain-ultra', generation: 12, fitness: 0.892, mutations: ['Multi-strategy fusion', 'Dynamic adaptation'] },
    { brainId: 'brain-predictor', generation: 18, fitness: 0.858, mutations: ['Economic event integration', 'Time series optimization'] }
  ];

  const handleConfigureBrain = async (brain: Brain) => {
    setSelectedBrain(brain);
    setConfigModalOpen(true);
    
    // Intentar cargar configuración existente
    try {
      const response = await fetch(`/api/institutions/001/brain-configs/${brain.type}`);
      if (response.ok) {
        const existingConfig = await response.json();
        setBrainConfigs(prev => ({
          ...prev,
          [brain.type]: existingConfig
        }));
        setConfigId(existingConfig.config_id);
      }
    } catch (error) {
      console.log('No existing configuration found');
    }
  };

  const handleSaveConfiguration = async (config: BrainConfig) => {
    if (!selectedBrain) return;

    try {
      const response = await fetch(`/api/institutions/001/brains/${selectedBrain.type}/configure`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        const result = await response.json();
        setConfigId(result.config_id);
        setBrainConfigs(prev => ({
          ...prev,
          [selectedBrain.type]: config
        }));
        setConfigModalOpen(false);
        alert('Configuración guardada exitosamente');
      }
    } catch (error) {
      console.error('Error saving configuration:', error);
      alert('Error al guardar la configuración');
    }
  };

  const handleTrainBrain = async (brain: Brain) => {
    if (!brainConfigs[brain.type]) {
      alert('Debe configurar el cerebro antes de entrenarlo');
      return;
    }

    setTrainingStatus(prev => ({
      ...prev,
      [brain.type]: 'in_progress'
    }));

    try {
      const response = await fetch(`/api/institutions/001/brains/${brain.type}/train`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          config_id: configId,
          training_params: {
            epochs: 100,
            batch_size: 32,
            validation_split: 0.2
          }
        })
      });

      if (response.ok) {
        // Simular entrenamiento
        setTimeout(() => {
          setTrainingStatus(prev => ({
            ...prev,
            [brain.type]: 'completed'
          }));
          alert('Entrenamiento completado exitosamente');
        }, 3000);
      }
    } catch (error) {
      console.error('Error training brain:', error);
      setTrainingStatus(prev => ({
        ...prev,
        [brain.type]: 'failed'
      }));
      alert('Error en el entrenamiento');
    }
  };

  const getTabIcon = (tabId: string) => {
    const icons: {[key: string]: React.ComponentType<{ className?: string }>} = {
      collaboration: Users,
      fusion: Zap,
      arena: Trophy,
      evolution: TrendingUp,
      specialization: Target,
      orchestration: Settings,
      gamification: Star,
      personalization: Crown
    };
    return icons[tabId] || Brain;
  };

  const getTabColor = (tabId: string) => {
    const colors: {[key: string]: string} = {
      collaboration: 'from-blue-500 to-cyan-500',
      fusion: 'from-purple-500 to-pink-500',
      arena: 'from-yellow-500 to-orange-500',
      evolution: 'from-green-500 to-emerald-500',
      specialization: 'from-red-500 to-pink-500',
      orchestration: 'from-indigo-500 to-purple-500',
      gamification: 'from-yellow-400 to-orange-400',
      personalization: 'from-purple-600 to-pink-600'
    };
    return colors[tabId] || 'from-gray-500 to-gray-600';
  };

  const tabs = [
    { id: 'collaboration', name: 'Cerebros Colaborativos', description: 'Los 3 cerebros trabajan en conjunto' },
    { id: 'fusion', name: 'Brain Fusion', description: 'Fusión de cerebros en tiempo real' },
    { id: 'arena', name: 'Brain Arena', description: 'Competencia entre cerebros' },
    { id: 'evolution', name: 'Brain Evolution', description: 'Evolución continua' },
    { id: 'specialization', name: 'Brain Specialization', description: 'Especialización por condiciones' },
    { id: 'orchestration', name: 'Brain Orchestration', description: 'Orquestación inteligente' },
    { id: 'gamification', name: 'Brain Gamification', description: 'Gamificación de cerebros' },
    { id: 'personalization', name: 'Brain Personalization', description: 'Personalización de cerebros' }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
            🧠 Mega Mind
          </h1>
          <p className="text-gray-400 text-lg">
            Sistema de Inteligencia Artificial Colaborativa para Trading Institucional
          </p>
        </div>

        {/* Tabs */}
        <div className="mb-6">
          <div className="flex flex-wrap gap-2">
            {tabs.map((tab) => {
              const Icon = getTabIcon(tab.id);
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                    isActive
                      ? `bg-gradient-to-r ${getTabColor(tab.id)} text-white shadow-lg`
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  <span className="font-medium">{tab.name}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* Content */}
        <div className="glass-effect rounded-lg p-6">
          {activeTab === 'collaboration' && (
            <div className="space-y-6">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold mb-4">🧠 Cerebros Colaborativos</h2>
                <p className="text-gray-400 text-lg">
                  Los 3 cerebros trabajan en conjunto para maximizar la precisión y reducir el riesgo
                </p>
              </div>

              {/* Brain Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
                {brains.map((brain) => {
                  const Icon = brain.icon;
                  const isTraining = trainingStatus[brain.type] === 'in_progress';
                  return (
                    <div key={brain.id} className="bg-gray-800 rounded-lg p-6 border border-gray-700">
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center space-x-3">
                          <Icon className="w-6 h-6 text-purple-400" />
                          <h3 className="text-xl font-bold">{brain.name}</h3>
                        </div>
                        <div className={`px-2 py-1 rounded text-xs ${
                          brain.status === 'active' ? 'bg-green-500/20 text-green-400' :
                          brain.status === 'training' ? 'bg-yellow-500/20 text-yellow-400' :
                          'bg-red-500/20 text-red-400'
                        }`}>
                          {brain.status === 'active' ? 'Activo' :
                           brain.status === 'training' ? 'Entrenando' : 'Inactivo'}
                        </div>
                      </div>
                      
                      <p className="text-gray-400 mb-4">{brain.description}</p>
                      
                      <div className="mb-4">
                        <div className="flex justify-between text-sm mb-1">
                          <span>Precisión</span>
                          <span>{brain.accuracy}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-2">
                          <div 
                            className="bg-gradient-to-r from-green-400 to-blue-400 h-2 rounded-full"
                            style={{ width: `${brain.accuracy}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="flex space-x-2">
                        <button
                          onClick={() => handleConfigureBrain(brain)}
                          className="flex-1 bg-purple-600 hover:bg-purple-700 text-white px-3 py-2 rounded text-sm flex items-center justify-center space-x-1"
                        >
                          <Settings className="w-4 h-4" />
                          <span>Configurar</span>
                        </button>
                        <button
                          onClick={() => handleTrainBrain(brain)}
                          disabled={!brainConfigs[brain.type] || isTraining}
                          className={`flex-1 px-3 py-2 rounded text-sm flex items-center justify-center space-x-1 ${
                            !brainConfigs[brain.type] || isTraining
                              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                              : 'bg-green-600 hover:bg-green-700 text-white'
                          }`}
                        >
                          {isTraining ? (
                            <>
                              <RotateCcw className="w-4 h-4 animate-spin" />
                              <span>Entrenando...</span>
                            </>
                          ) : (
                            <>
                              <Brain className="w-4 h-4" />
                              <span>Entrenar</span>
                            </>
                          )}
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Consensus Voting */}
              <div className="bg-gray-800 rounded-lg p-6">
                <h3 className="text-xl font-bold mb-4">🗳️ Votación de Consenso</h3>
                <div className="space-y-3">
                  {consensusVotes.map((vote) => {
                    const brain = brains.find(b => b.id === vote.brainId);
                    if (!brain) return null;
                    
                    return (
                      <div key={vote.brainId} className="flex items-center justify-between p-3 bg-gray-700 rounded">
                        <div className="flex items-center space-x-3">
                          <span className="font-medium">{brain.name}</span>
                          <span className={`px-2 py-1 rounded text-xs ${
                            vote.vote === 'buy' ? 'bg-green-500/20 text-green-400' :
                            vote.vote === 'sell' ? 'bg-red-500/20 text-red-400' :
                            'bg-yellow-500/20 text-yellow-400'
                          }`}>
                            {vote.vote.toUpperCase()}
                          </span>
                        </div>
                        <div className="text-right">
                          <div className="text-sm text-gray-400">Confianza</div>
                          <div className="font-bold">{(vote.confidence * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'fusion' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">⚡ Brain Fusion</h2>
              <p className="text-gray-400">Fusión de cerebros en tiempo real</p>
            </div>
          )}

          {activeTab === 'arena' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">🏆 Brain Arena</h2>
              <p className="text-gray-400">Competencia entre cerebros</p>
            </div>
          )}

          {activeTab === 'evolution' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">🔄 Brain Evolution</h2>
              <p className="text-gray-400">Evolución continua</p>
            </div>
          )}

          {activeTab === 'specialization' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">🎯 Brain Specialization</h2>
              <p className="text-gray-400">Especialización por condiciones</p>
            </div>
          )}

          {activeTab === 'orchestration' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">🎼 Brain Orchestration</h2>
              <p className="text-gray-400">Orquestación inteligente</p>
            </div>
          )}

          {activeTab === 'gamification' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">🎮 Brain Gamification</h2>
              <p className="text-gray-400">Gamificación de cerebros</p>
            </div>
          )}

          {activeTab === 'personalization' && (
            <div className="text-center">
              <h2 className="text-3xl font-bold mb-4">👑 Brain Personalization</h2>
              <p className="text-gray-400">Personalización de cerebros</p>
            </div>
          )}
        </div>
      </div>

      {/* Configuration Modal */}
      <BrainConfigModal
        isOpen={configModalOpen}
        onClose={() => setConfigModalOpen(false)}
        onSave={handleSaveConfiguration}
        selectedBrain={selectedBrain}
        initialConfig={selectedBrain ? brainConfigs[selectedBrain.type] : undefined}
      />
    </div>
  );
};

export default MegaMind; 