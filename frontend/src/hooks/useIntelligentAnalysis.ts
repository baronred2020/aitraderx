import { useState } from 'react';

export interface TradingType {
  id: string;
  name: string;
  description: string;
  timeframe: string;
  reason: string;
  color: string;
}

export interface AnalysisResult {
  tradingType: TradingType;
  recommendations: string[];
  riskLevel: 'low' | 'medium' | 'high';
  confidence: number;
  timestamp: Date;
  technicalAnalysis?: {
    rsi: number | null;
    macd: number | null;
    sma20: number | null;
    ema50: number | null;
    adx: number | null;
    currentPrice: number;
    trend: 'bullish' | 'bearish' | 'neutral';
    volatility: number;
    volumeRatio: number;
  };
}

// Funciones de análisis técnico
const calculateRSI = (data: any[], period: number = 14): number | null => {
  if (data.length < period + 1) return null;
  
  let gains = 0;
  let losses = 0;
  
  for (let i = 1; i <= period; i++) {
    const change = data[data.length - i].close - data[data.length - i - 1].close;
    if (change > 0) {
      gains += change;
    } else {
      losses += Math.abs(change);
    }
  }
  
  const avgGain = gains / period;
  const avgLoss = losses / period;
  
  if (avgLoss === 0) return 50; // Neutral en lugar de 100
  
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  
  // Asegurar que RSI esté en rango válido
  return Math.max(0, Math.min(100, rsi));
};

const calculateMACD = (data: any[], fastPeriod: number = 12, slowPeriod: number = 26): number | null => {
  if (data.length < Math.max(fastPeriod, slowPeriod) + 1) return null;
  
  // Calcular EMA rápida
  let emaFast = data[0].close;
  const fastMultiplier = 2 / (fastPeriod + 1);
  for (let i = 1; i < data.length; i++) {
    emaFast = (data[i].close * fastMultiplier) + (emaFast * (1 - fastMultiplier));
  }
  
  // Calcular EMA lenta
  let emaSlow = data[0].close;
  const slowMultiplier = 2 / (slowPeriod + 1);
  for (let i = 1; i < data.length; i++) {
    emaSlow = (data[i].close * slowMultiplier) + (emaSlow * (1 - slowMultiplier));
  }
  
  const macd = emaFast - emaSlow;
  
  // Asegurar que MACD no sea extremo
  return Math.max(-1, Math.min(1, macd));
};

const calculateSMA = (data: any[], period: number): number | null => {
  if (data.length < period) return null;
  
  const sum = data.slice(-period).reduce((acc, candle) => acc + candle.close, 0);
  return sum / period;
};

const calculateEMA = (data: any[], period: number): number | null => {
  if (data.length < period) return null;
  
  const multiplier = 2 / (period + 1);
  let ema = data[0].close;
  
  for (let i = 1; i < data.length; i++) {
    ema = (data[i].close * multiplier) + (ema * (1 - multiplier));
  }
  
  return ema;
};

const calculateADX = (data: any[], period: number = 14): number | null => {
  if (data.length < period + 1) return null;
  
  let plusDM = 0;
  let minusDM = 0;
  let trueRange = 0;
  
  for (let i = 1; i <= period; i++) {
    const highDiff = data[data.length - i].high - data[data.length - i - 1].high;
    const lowDiff = data[data.length - i - 1].low - data[data.length - i].low;
    
    if (highDiff > lowDiff && highDiff > 0) {
      plusDM += highDiff;
    }
    if (lowDiff > highDiff && lowDiff > 0) {
      minusDM += lowDiff;
    }
    
    const tr = Math.max(
      data[data.length - i].high - data[data.length - i].low,
      Math.abs(data[data.length - i].high - data[data.length - i - 1].close),
      Math.abs(data[data.length - i].low - data[data.length - i - 1].close)
    );
    trueRange += tr;
  }
  
  if (trueRange === 0) return 0; // Evitar división por cero
  
  const plusDI = (plusDM / trueRange) * 100;
  const minusDI = (minusDM / trueRange) * 100;
  
  if (plusDI + minusDI === 0) return 0; // Evitar división por cero
  
  const dx = Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100;
  
  // Asegurar que ADX esté en rango válido
  return Math.max(0, Math.min(100, dx));
};

const calculateVolatility = (data: any[], period: number = 20): number => {
  if (data.length < period) return 0;
  
  // Usar solo los últimos 'period' datos para el cálculo
  const recentData = data.slice(-period);
  const returns = [];
  
  for (let i = 1; i < recentData.length; i++) {
    const return_ = (recentData[i].close - recentData[i - 1].close) / recentData[i - 1].close;
    returns.push(return_);
  }
  
  const mean = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
  const volatility = Math.sqrt(variance) * 100; // Convertir a porcentaje
  
  // Asegurar que volatilidad esté en rango razonable
  return Math.max(0, Math.min(20, volatility));
};

const calculateVolumeRatio = (data: any[], period: number = 20): number => {
  if (data.length < period) return 1;
  
  const currentVolume = data[data.length - 1].volume;
  const avgVolume = data.slice(-period).reduce((sum, candle) => sum + candle.volume, 0) / period;
  
  const ratio = currentVolume / avgVolume;
  
  // Asegurar que ratio esté en rango razonable
  return Math.max(0.1, Math.min(10, ratio));
};

export const useIntelligentAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<AnalysisResult | null>(null);

  const executeAnalysis = async (tradingType: TradingType, symbol: string): Promise<AnalysisResult> => {
    setIsAnalyzing(true);
    
    try {
      console.log(`Ejecutando análisis inteligente para ${symbol} con tipo: ${tradingType.name}`);
      
      // Obtener datos reales de Yahoo Finance
      const response = await fetch(`http://localhost:8000/api/candles?symbol=${symbol}&interval=15&count=100`);
      
      if (!response.ok) {
        throw new Error(`Error obteniendo datos: ${response.status}`);
      }
      
      const candleData = await response.json();
      
      if (!candleData.values || candleData.values.length < 50) {
        throw new Error('Datos insuficientes para análisis');
      }
      
      // Convertir datos al formato necesario
      const chartData = candleData.values.map((item: any) => ({
        open: parseFloat(item.open),
        high: parseFloat(item.high),
        low: parseFloat(item.low),
        close: parseFloat(item.close),
        volume: parseFloat(item.volume),
      }));
      
      // Ajustar parámetros según el tipo de trading
      let rsiPeriod = 14;
      let smaPeriod = 20;
      let emaPeriod = 50;
      let macdFastPeriod = 12;
      let macdSlowPeriod = 26;
      let adxPeriod = 14;
      let volatilityPeriod = 20;
      let volumePeriod = 20;
      
      switch (tradingType.id) {
        case 'scalping':
          rsiPeriod = 7;
          smaPeriod = 10;
          emaPeriod = 20;
          macdFastPeriod = 6;
          macdSlowPeriod = 13;
          adxPeriod = 7;
          volatilityPeriod = 10;
          volumePeriod = 10;
          break;
        case 'day_trading':
          rsiPeriod = 14;
          smaPeriod = 20;
          emaPeriod = 50;
          macdFastPeriod = 12;
          macdSlowPeriod = 26;
          adxPeriod = 14;
          volatilityPeriod = 20;
          volumePeriod = 20;
          break;
        case 'swing_trading':
          rsiPeriod = 21;
          smaPeriod = 50;
          emaPeriod = 100;
          macdFastPeriod = 21;
          macdSlowPeriod = 52;
          adxPeriod = 21;
          volatilityPeriod = 50;
          volumePeriod = 50;
          break;
        case 'position_trading':
          rsiPeriod = 30;
          smaPeriod = 100;
          emaPeriod = 200;
          macdFastPeriod = 30;
          macdSlowPeriod = 60;
          adxPeriod = 30;
          volatilityPeriod = 100;
          volumePeriod = 100;
          break;
      }
      
      // Calcular indicadores técnicos con parámetros específicos por tipo
      const rsi = calculateRSI(chartData, rsiPeriod);
      const macd = calculateMACD(chartData, macdFastPeriod, macdSlowPeriod);
      const sma20 = calculateSMA(chartData, smaPeriod);
      const ema50 = calculateEMA(chartData, emaPeriod);
      const adx = calculateADX(chartData, adxPeriod);
      const currentPrice = chartData[chartData.length - 1].close;
      const volatility = calculateVolatility(chartData, volatilityPeriod);
      const volumeRatio = calculateVolumeRatio(chartData, volumePeriod);
      
      // Log de debugging para indicadores técnicos
      console.log('🔍 Technical Indicators Debug:', {
        symbol,
        tradingType: tradingType.name,
        rsi,
        macd,
        sma20,
        ema50,
        adx,
        currentPrice,
        volatility,
        volumeRatio,
        dataLength: chartData.length
      });
      
      // Determinar tendencia
      let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
      if (sma20 !== null && ema50 !== null) {
        if (currentPrice > sma20 && sma20 > ema50) {
          trend = 'bullish';
        } else if (currentPrice < sma20 && sma20 < ema50) {
          trend = 'bearish';
        }
      }
      
      console.log('🔍 Trend Analysis:', { currentPrice, sma20, ema50, trend });
      
      // Generar recomendaciones basadas en análisis técnico real
      const recommendations = generateRealRecommendations({
        tradingType,
        rsi,
        macd,
        sma20,
        ema50,
        adx,
        currentPrice,
        trend,
        volatility,
        volumeRatio,
        symbol
      });
      
      // Calcular nivel de riesgo basado en volatilidad real
      const riskLevel = calculateRealRiskLevel(volatility, tradingType);
      
      // Calcular confianza basada en múltiples factores
      const confidence = calculateRealConfidence({
        rsi,
        macd,
        adx,
        trend,
        volatility,
        volumeRatio,
        tradingType
      });
      
      const result: AnalysisResult = {
        tradingType,
        recommendations,
        riskLevel,
        confidence,
        timestamp: new Date(),
        technicalAnalysis: {
          rsi,
          macd,
          sma20,
          ema50,
          adx,
          currentPrice,
          trend,
          volatility,
          volumeRatio
        }
      };
      
      setLastAnalysis(result);
      return result;
      
    } catch (error) {
      console.error('Error en análisis inteligente:', error);
      throw error;
    } finally {
      setIsAnalyzing(false);
    }
  };

  const generateRealRecommendations = (analysis: {
    tradingType: TradingType;
    rsi: number | null;
    macd: number | null;
    sma20: number | null;
    ema50: number | null;
    adx: number | null;
    currentPrice: number;
    trend: 'bullish' | 'bearish' | 'neutral';
    volatility: number;
    volumeRatio: number;
    symbol: string;
  }): string[] => {
    const recommendations: string[] = [];
    
    // Análisis específico por tipo de trading
    switch (analysis.tradingType.id) {
      case 'scalping':
        // Análisis específico para Scalping
        if (analysis.rsi !== null) {
          if (analysis.rsi < 20) {
            recommendations.push(`RSI extremadamente bajo (${analysis.rsi.toFixed(1)}). Oportunidad de scalping alcista`);
          } else if (analysis.rsi > 80) {
            recommendations.push(`RSI extremadamente alto (${analysis.rsi.toFixed(1)}). Oportunidad de scalping bajista`);
          } else if (analysis.rsi < 35) {
            recommendations.push(`RSI en sobreventa (${analysis.rsi.toFixed(1)}). Buscar entradas largas rápidas`);
          } else if (analysis.rsi > 65) {
            recommendations.push(`RSI en sobrecompra (${analysis.rsi.toFixed(1)}). Buscar entradas cortas rápidas`);
          }
        }
        
        if (analysis.volatility > 3) {
          recommendations.push(`Alta volatilidad (${analysis.volatility.toFixed(1)}%). Perfecto para scalping - usar stop-loss de 0.3%`);
        } else if (analysis.volatility < 1) {
          recommendations.push(`Baja volatilidad (${analysis.volatility.toFixed(1)}%). Mercado tranquilo - posiciones más pequeñas`);
        }
        
        if (analysis.volumeRatio > 2) {
          recommendations.push('Volumen muy alto. Confirmación fuerte para scalping');
        } else if (analysis.volumeRatio < 0.5) {
          recommendations.push('Volumen bajo. Evitar scalping - falta de liquidez');
        }
        
        recommendations.push('Scalping: Usar timeframes de 1-5 minutos');
        recommendations.push('Scalping: Mantener stop-loss muy ajustado (0.3-0.5%)');
        recommendations.push('Scalping: Tomar beneficios rápidos (0.5-1%)');
        recommendations.push('Scalping: Monitorear spread y comisiones');
        break;
        
      case 'day_trading':
        // Análisis específico para Day Trading
        if (analysis.rsi !== null) {
          if (analysis.rsi < 25) {
            recommendations.push(`RSI muy bajo (${analysis.rsi.toFixed(1)}). Oportunidad de day trading alcista`);
          } else if (analysis.rsi > 75) {
            recommendations.push(`RSI muy alto (${analysis.rsi.toFixed(1)}). Oportunidad de day trading bajista`);
          } else if (analysis.rsi > 45 && analysis.rsi < 55) {
            recommendations.push(`RSI neutral (${analysis.rsi.toFixed(1)}). Mercado lateral - usar rangos`);
          }
        }
        
        if (analysis.trend === 'bullish') {
          recommendations.push('Tendencia alcista clara. Buscar entradas largas en pullbacks');
        } else if (analysis.trend === 'bearish') {
          recommendations.push('Tendencia bajista clara. Buscar entradas cortas en rallies');
        } else {
          recommendations.push('Tendencia neutral. Usar estrategias de rango');
        }
        
        if (analysis.volatility > 2.5) {
          recommendations.push(`Volatilidad alta (${analysis.volatility.toFixed(1)}%). Usar stop-loss de 1-1.5%`);
        } else if (analysis.volatility < 1.5) {
          recommendations.push(`Volatilidad baja (${analysis.volatility.toFixed(1)}%). Posiciones más grandes`);
        }
        
        recommendations.push('Day Trading: Usar múltiples timeframes (M15, H1, H4)');
        recommendations.push('Day Trading: Cerrar posiciones antes del fin de día');
        recommendations.push('Day Trading: Establecer objetivos 1:2 o 1:3');
        recommendations.push('Day Trading: Evitar operar en noticias importantes');
        break;
        
      case 'swing_trading':
        // Análisis específico para Swing Trading
        if (analysis.rsi !== null) {
          if (analysis.rsi < 30) {
            recommendations.push(`RSI en sobreventa (${analysis.rsi.toFixed(1)}). Oportunidad de swing largo`);
          } else if (analysis.rsi > 70) {
            recommendations.push(`RSI en sobrecompra (${analysis.rsi.toFixed(1)}). Oportunidad de swing corto`);
          } else if (analysis.rsi > 40 && analysis.rsi < 60) {
            recommendations.push(`RSI neutral (${analysis.rsi.toFixed(1)}). Esperar mejor entrada`);
          }
        }
        
        if (analysis.adx !== null) {
          if (analysis.adx > 30) {
            recommendations.push(`Tendencia fuerte (ADX: ${analysis.adx.toFixed(1)}). Ideal para swing trading`);
          } else if (analysis.adx < 20) {
            recommendations.push(`Tendencia débil (ADX: ${analysis.adx.toFixed(1)}). Evitar swing trading`);
          }
        }
        
        if (analysis.volatility > 4) {
          recommendations.push(`Alta volatilidad (${analysis.volatility.toFixed(1)}%). Usar stop-loss de 2-3%`);
        } else if (analysis.volatility < 2) {
          recommendations.push(`Baja volatilidad (${analysis.volatility.toFixed(1)}%). Mercado estable para swing`);
        }
        
        recommendations.push('Swing Trading: Mantener posiciones por días/semanas');
        recommendations.push('Swing Trading: Aplicar gestión de riesgo 1:2 o 1:3');
        recommendations.push('Swing Trading: Usar análisis fundamental complementario');
        recommendations.push('Swing Trading: Considerar niveles de soporte/resistencia');
        break;
        
      case 'position_trading':
        // Análisis específico para Position Trading
        if (analysis.rsi !== null) {
          if (analysis.rsi < 35) {
            recommendations.push(`RSI bajo (${analysis.rsi.toFixed(1)}). Oportunidad de posición larga`);
          } else if (analysis.rsi > 65) {
            recommendations.push(`RSI alto (${analysis.rsi.toFixed(1)}). Oportunidad de posición corta`);
          } else {
            recommendations.push(`RSI neutral (${analysis.rsi.toFixed(1)}). Posición conservadora`);
          }
        }
        
        if (analysis.trend === 'bullish') {
          recommendations.push('Tendencia alcista de largo plazo. Posición larga recomendada');
        } else if (analysis.trend === 'bearish') {
          recommendations.push('Tendencia bajista de largo plazo. Posición corta recomendada');
        } else {
          recommendations.push('Tendencia neutral. Mantener posiciones existentes');
        }
        
        if (analysis.volatility > 5) {
          recommendations.push(`Alta volatilidad (${analysis.volatility.toFixed(1)}%). Usar stop-loss de 5-10%`);
        } else if (analysis.volatility < 3) {
          recommendations.push(`Baja volatilidad (${analysis.volatility.toFixed(1)}%). Mercado estable para posiciones`);
        }
        
        recommendations.push('Position Trading: Análisis fundamental profundo requerido');
        recommendations.push('Position Trading: Considerar factores macroeconómicos');
        recommendations.push('Position Trading: Gestión de riesgo conservadora (1:1.5)');
        recommendations.push('Position Trading: Mantener posiciones por semanas/meses');
        break;
    }
    
    // Recomendaciones basadas en MACD (específicas por tipo)
    if (analysis.macd !== null) {
      const macdThreshold = analysis.tradingType.id === 'scalping' ? 0.0005 : 
                           analysis.tradingType.id === 'day_trading' ? 0.001 : 
                           analysis.tradingType.id === 'swing_trading' ? 0.002 : 0.005;
      
      if (analysis.macd > macdThreshold) {
        recommendations.push(`MACD alcista (${analysis.macd.toFixed(4)}). Momentum positivo`);
      } else if (analysis.macd < -macdThreshold) {
        recommendations.push(`MACD bajista (${analysis.macd.toFixed(4)}). Momentum negativo`);
      } else {
        recommendations.push(`MACD neutral (${analysis.macd.toFixed(4)}). Momentum débil`);
      }
    }
    
    // Recomendaciones basadas en volumen (específicas por tipo)
    const volumeThreshold = analysis.tradingType.id === 'scalping' ? 1.8 : 
                           analysis.tradingType.id === 'day_trading' ? 1.5 : 
                           analysis.tradingType.id === 'swing_trading' ? 1.3 : 1.2;
    
    if (analysis.volumeRatio > volumeThreshold) {
      recommendations.push(`Volumen alto (${analysis.volumeRatio.toFixed(1)}x). Confirmación de movimiento`);
    } else if (analysis.volumeRatio < 0.7) {
      recommendations.push(`Volumen bajo (${analysis.volumeRatio.toFixed(1)}x). Falta de confirmación`);
    } else {
      recommendations.push(`Volumen normal (${analysis.volumeRatio.toFixed(1)}x). Mercado equilibrado`);
    }
    
    return recommendations.slice(0, 10); // Aumentar a 10 recomendaciones
  };

  const calculateRealRiskLevel = (volatility: number, tradingType: TradingType): 'low' | 'medium' | 'high' => {
    // Umbrales de volatilidad específicos por tipo de trading
    const volatilityThresholds = {
      scalping: {
        low: 1.0,    // Baja volatilidad para scalping
        medium: 2.5, // Volatilidad media para scalping
        high: 4.0    // Alta volatilidad para scalping
      },
      day_trading: {
        low: 1.5,    // Baja volatilidad para day trading
        medium: 3.0, // Volatilidad media para day trading
        high: 5.0    // Alta volatilidad para day trading
      },
      swing_trading: {
        low: 2.0,    // Baja volatilidad para swing trading
        medium: 4.0, // Volatilidad media para swing trading
        high: 6.0    // Alta volatilidad para swing trading
      },
      position_trading: {
        low: 2.5,    // Baja volatilidad para position trading
        medium: 5.0, // Volatilidad media para position trading
        high: 8.0    // Alta volatilidad para position trading
      }
    };
    
    const thresholds = volatilityThresholds[tradingType.id as keyof typeof volatilityThresholds];
    
    // Base de riesgo por tipo de trading
    const baseRisk = {
      scalping: 'high',
      day_trading: 'medium',
      swing_trading: 'medium',
      position_trading: 'low'
    } as const;
    
    let riskLevel = baseRisk[tradingType.id as keyof typeof baseRisk];
    
    // Ajustar basado en volatilidad específica del tipo
    if (volatility > thresholds.high) {
      riskLevel = 'high';
    } else if (volatility < thresholds.low) {
      riskLevel = 'low';
    } else if (volatility > thresholds.medium) {
      riskLevel = 'high';
    } else {
      riskLevel = 'medium';
    }
    
    // Log para debugging
    console.log('🔍 Risk Level Debug:', {
      tradingType: tradingType.name,
      volatility,
      thresholds,
      baseRisk: baseRisk[tradingType.id as keyof typeof baseRisk],
      finalRiskLevel: riskLevel
    });
    
    return riskLevel;
  };

  const calculateRealConfidence = (analysis: {
    rsi: number | null;
    macd: number | null;
    adx: number | null;
    trend: 'bullish' | 'bearish' | 'neutral';
    volatility: number;
    volumeRatio: number;
    tradingType: TradingType;
  }): number => {
    // Base específica por tipo de trading
    let confidence = 0.70; // Base 70%
    
    // Umbrales específicos por tipo de trading
    const thresholds = {
      scalping: {
        rsiExtreme: { low: 20, high: 80 },
        rsiSignal: { low: 35, high: 65 },
        macdThreshold: 0.0005,
        volumeThreshold: 1.8,
        volatilityIdeal: { min: 2, max: 5 }
      },
      day_trading: {
        rsiExtreme: { low: 25, high: 75 },
        rsiSignal: { low: 30, high: 70 },
        macdThreshold: 0.001,
        volumeThreshold: 1.5,
        volatilityIdeal: { min: 1.5, max: 4 }
      },
      swing_trading: {
        rsiExtreme: { low: 30, high: 70 },
        rsiSignal: { low: 35, high: 65 },
        macdThreshold: 0.002,
        volumeThreshold: 1.3,
        volatilityIdeal: { min: 2, max: 6 }
      },
      position_trading: {
        rsiExtreme: { low: 35, high: 65 },
        rsiSignal: { low: 40, high: 60 },
        macdThreshold: 0.005,
        volumeThreshold: 1.2,
        volatilityIdeal: { min: 1, max: 8 }
      }
    };
    
    const t = thresholds[analysis.tradingType.id as keyof typeof thresholds];
    
    // Factor RSI específico por tipo (0-20 puntos)
    if (analysis.rsi !== null) {
      if (analysis.rsi < t.rsiExtreme.low || analysis.rsi > t.rsiExtreme.high) {
        confidence += 0.20; // Señales extremas
      } else if (analysis.rsi < t.rsiSignal.low || analysis.rsi > t.rsiSignal.high) {
        confidence += 0.15; // Señales claras
      } else if (analysis.rsi > 40 && analysis.rsi < 60) {
        confidence += 0.10; // Neutral estable
      } else {
        confidence += 0.05; // Zona intermedia
      }
    }
    
    // Factor MACD específico por tipo (0-15 puntos)
    if (analysis.macd !== null) {
      if (Math.abs(analysis.macd) > t.macdThreshold * 2) {
        confidence += 0.15; // Señal muy clara
      } else if (Math.abs(analysis.macd) > t.macdThreshold) {
        confidence += 0.10; // Señal clara
      } else if (Math.abs(analysis.macd) > t.macdThreshold * 0.5) {
        confidence += 0.05; // Señal débil
      }
    }
    
    // Factor tendencia (0-15 puntos)
    if (analysis.trend === 'bullish' || analysis.trend === 'bearish') {
      confidence += 0.15; // Tendencia clara
    } else {
      confidence += 0.05; // Neutral
    }
    
    // Factor ADX específico por tipo (0-15 puntos)
    if (analysis.adx !== null) {
      const adxThreshold = analysis.tradingType.id === 'scalping' ? 20 :
                          analysis.tradingType.id === 'day_trading' ? 25 :
                          analysis.tradingType.id === 'swing_trading' ? 30 : 15;
      
      if (analysis.adx > adxThreshold + 10) {
        confidence += 0.15; // Tendencia muy fuerte
      } else if (analysis.adx > adxThreshold) {
        confidence += 0.10; // Tendencia fuerte
      } else if (analysis.adx > adxThreshold - 5) {
        confidence += 0.05; // Tendencia moderada
      }
    }
    
    // Factor volumen específico por tipo (0-10 puntos)
    if (analysis.volumeRatio > t.volumeThreshold) {
      confidence += 0.10; // Volumen muy alto
    } else if (analysis.volumeRatio > t.volumeThreshold * 0.8) {
      confidence += 0.08; // Volumen alto
    } else if (analysis.volumeRatio > 1.0) {
      confidence += 0.05; // Volumen normal
    } else if (analysis.volumeRatio < 0.5) {
      confidence -= 0.05; // Volumen muy bajo
    }
    
    // Factor volatilidad específico por tipo (0-10 puntos)
    if (analysis.volatility >= t.volatilityIdeal.min && analysis.volatility <= t.volatilityIdeal.max) {
      confidence += 0.10; // Volatilidad ideal para el tipo
    } else if (analysis.volatility < t.volatilityIdeal.min) {
      confidence += 0.05; // Volatilidad baja
    } else if (analysis.volatility > t.volatilityIdeal.max * 1.5) {
      confidence -= 0.10; // Volatilidad excesiva
    } else if (analysis.volatility > t.volatilityIdeal.max) {
      confidence -= 0.05; // Volatilidad alta
    }
    
    // Multiplicadores específicos por tipo de trading
    const tradingTypeMultiplier = {
      scalping: 0.85,      // Menor confianza para scalping (más difícil)
      day_trading: 0.90,   // Confianza moderada para day trading
      swing_trading: 0.95, // Alta confianza para swing
      position_trading: 1.0 // Máxima confianza para position
    };
    
    const multiplier = tradingTypeMultiplier[analysis.tradingType.id as keyof typeof tradingTypeMultiplier];
    confidence = confidence * multiplier;
    
    // Límites específicos por tipo
    const limits = {
      scalping: { min: 0.35, max: 0.90 },
      day_trading: { min: 0.40, max: 0.92 },
      swing_trading: { min: 0.45, max: 0.95 },
      position_trading: { min: 0.50, max: 0.95 }
    };
    
    const l = limits[analysis.tradingType.id as keyof typeof limits];
    confidence = Math.max(l.min, Math.min(l.max, confidence));
    
    // Log para debugging
    console.log('🔍 Confidence Debug:', {
      tradingType: analysis.tradingType.name,
      base: 0.70,
      rsi: analysis.rsi,
      macd: analysis.macd,
      trend: analysis.trend,
      adx: analysis.adx,
      volumeRatio: analysis.volumeRatio,
      volatility: analysis.volatility,
      multiplier,
      finalConfidence: confidence,
      finalConfidencePercent: (confidence * 100).toFixed(1) + '%'
    });
    
    return confidence;
  };

  return {
    isAnalyzing,
    lastAnalysis,
    executeAnalysis
  };
}; 
