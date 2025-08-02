import { useState, useEffect, useCallback } from 'react';

export interface TradingConfig {
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

// Configuración por defecto
const DEFAULT_CONFIG: TradingConfig = {
  // Órdenes
  defaultOrderAmount: 10000,
  defaultOrderType: 'market',
  autoSetStopLoss: false,
  defaultStopLossPercent: 2.0,
  autoSetTakeProfit: false,
  defaultTakeProfitPercent: 3.0,
  
  // Riesgo
  maxPositionSize: 20, // 20% del balance
  maxDailyLoss: 10, // 10% del balance
  maxOpenPositions: 5,
  
  // Notificaciones
  enableNotifications: true,
  priceAlerts: true,
  orderExecutedAlerts: true,
  stopLossAlerts: true,
  
  // Interfaz
  autoRefreshInterval: 10, // 10 segundos
  showAdvancedOptions: false,
  defaultTimeframe: '1H',
  
  // Datos
  dataSource: 'yahoo',
  updateFrequency: 10, // 10 segundos
};

export const useTradingConfig = () => {
  const [config, setConfig] = useState<TradingConfig>(DEFAULT_CONFIG);
  const [loading, setLoading] = useState(true);

  // Cargar configuración desde localStorage
  const loadConfig = useCallback(() => {
    try {
      const savedConfig = localStorage.getItem('trading_config');
      if (savedConfig) {
        const parsedConfig = JSON.parse(savedConfig);
        // Combinar con configuración por defecto para asegurar que todos los campos estén presentes
        setConfig({ ...DEFAULT_CONFIG, ...parsedConfig });
      } else {
        setConfig(DEFAULT_CONFIG);
      }
    } catch (error) {
      console.error('Error loading trading config:', error);
      setConfig(DEFAULT_CONFIG);
    } finally {
      setLoading(false);
    }
  }, []);

  // Guardar configuración en localStorage
  const saveConfig = useCallback((newConfig: TradingConfig) => {
    try {
      localStorage.setItem('trading_config', JSON.stringify(newConfig));
      setConfig(newConfig);
      return true;
    } catch (error) {
      console.error('Error saving trading config:', error);
      return false;
    }
  }, []);

  // Restablecer configuración por defecto
  const resetConfig = useCallback(() => {
    localStorage.removeItem('trading_config');
    setConfig(DEFAULT_CONFIG);
  }, []);

  // Actualizar una configuración específica
  const updateConfig = useCallback((key: keyof TradingConfig, value: any) => {
    const newConfig = { ...config, [key]: value };
    setConfig(newConfig);
    saveConfig(newConfig);
  }, [config, saveConfig]);

  // Cargar configuración al inicializar
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  // Función para aplicar configuración automática a órdenes
  const applyOrderDefaults = useCallback((orderData: any) => {
    const enhancedOrder = { ...orderData };
    
    // Aplicar cantidad por defecto si no se especifica
    if (!enhancedOrder.amount) {
      enhancedOrder.amount = config.defaultOrderAmount;
    }
    
    // Aplicar tipo de orden por defecto si no se especifica
    if (!enhancedOrder.type) {
      enhancedOrder.type = config.defaultOrderType;
    }
    
    // Aplicar Stop Loss automático si está habilitado
    if (config.autoSetStopLoss && !enhancedOrder.stopLoss) {
      const currentPrice = enhancedOrder.price || 1.0;
      const stopLossPercent = config.defaultStopLossPercent / 100;
      
      if (enhancedOrder.side === 'buy') {
        enhancedOrder.stopLoss = currentPrice * (1 - stopLossPercent);
      } else {
        enhancedOrder.stopLoss = currentPrice * (1 + stopLossPercent);
      }
    }
    
    // Aplicar Take Profit automático si está habilitado
    if (config.autoSetTakeProfit && !enhancedOrder.takeProfit) {
      const currentPrice = enhancedOrder.price || 1.0;
      const takeProfitPercent = config.defaultTakeProfitPercent / 100;
      
      if (enhancedOrder.side === 'buy') {
        enhancedOrder.takeProfit = currentPrice * (1 + takeProfitPercent);
      } else {
        enhancedOrder.takeProfit = currentPrice * (1 - takeProfitPercent);
      }
    }
    
    return enhancedOrder;
  }, [config]);

  // Función para validar límites de riesgo
  const validateRiskLimits = useCallback((orderAmount: number, currentBalance: number, openPositions: any[]) => {
    const validations = {
      isValid: true,
      errors: [] as string[]
    };

    // Validar tamaño máximo de posición
    const positionSizePercent = (orderAmount / currentBalance) * 100;
    if (positionSizePercent > config.maxPositionSize) {
      validations.isValid = false;
      validations.errors.push(`El tamaño de la posición (${positionSizePercent.toFixed(1)}%) excede el límite máximo (${config.maxPositionSize}%)`);
    }

    // Validar número máximo de posiciones abiertas
    if (openPositions.length >= config.maxOpenPositions) {
      validations.isValid = false;
      validations.errors.push(`Ya tienes ${openPositions.length} posiciones abiertas (máximo: ${config.maxOpenPositions})`);
    }

    return validations;
  }, [config]);

  // Función para verificar pérdida diaria
  const checkDailyLoss = useCallback((dailyPnL: number, currentBalance: number) => {
    const dailyLossPercent = Math.abs(Math.min(dailyPnL, 0)) / currentBalance * 100;
    
    if (dailyLossPercent >= config.maxDailyLoss) {
      return {
        exceeded: true,
        currentLoss: dailyLossPercent,
        maxLoss: config.maxDailyLoss,
        message: `Pérdida diaria (${dailyLossPercent.toFixed(1)}%) excede el límite máximo (${config.maxDailyLoss}%)`
      };
    }
    
    return {
      exceeded: false,
      currentLoss: dailyLossPercent,
      maxLoss: config.maxDailyLoss
    };
  }, [config]);

  return {
    config,
    loading,
    saveConfig,
    resetConfig,
    updateConfig,
    applyOrderDefaults,
    validateRiskLimits,
    checkDailyLoss,
    DEFAULT_CONFIG
  };
}; 