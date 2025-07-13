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
  
  if (avgLoss === 0) return 100;
  
  const rs = avgGain / avgLoss;
  const rsi = 100 - (100 / (1 + rs));
  
  return rsi;
};

const calculateMACD = (data: any[]): number | null => {
  if (data.length < 26) return null;
  
  // Calcular EMA 12
  let ema12 = data[0].close;
  for (let i = 1; i < data.length; i++) {
    ema12 = (data[i].close * 0.15) + (ema12 * 0.85);
  }
  
  // Calcular EMA 26
  let ema26 = data[0].close;
  for (let i = 1; i < data.length; i++) {
    ema26 = (data[i].close * 0.075) + (ema26 * 0.925);
  }
  
  return ema12 - ema26;
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
  
  const plusDI = (plusDM / trueRange) * 100;
  const minusDI = (minusDM / trueRange) * 100;
  const dx = Math.abs(plusDI - minusDI) / (plusDI + minusDI) * 100;
  
  return dx;
};

const calculateVolatility = (data: any[]): number => {
  if (data.length < 2) return 0;
  
  const returns = [];
  for (let i = 1; i < data.length; i++) {
    returns.push((data[i].close - data[i-1].close) / data[i-1].close);
  }
  
  const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
  const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
  
  return Math.sqrt(variance) * 100; // Volatilidad en porcentaje
};

const calculateVolumeRatio = (data: any[]): number => {
  if (data.length < 20) return 1;
  
  const currentVolume = data[data.length - 1].volume;
  const avgVolume = data.slice(-20).reduce((sum, candle) => sum + candle.volume, 0) / 20;
  
  return currentVolume / avgVolume;
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
      
      switch (tradingType.id) {
        case 'scalping':
          rsiPeriod = 7;
          smaPeriod = 10;
          emaPeriod = 20;
          break;
        case 'day_trading':
          rsiPeriod = 14;
          smaPeriod = 20;
          emaPeriod = 50;
          break;
        case 'swing_trading':
          rsiPeriod = 21;
          smaPeriod = 50;
          emaPeriod = 100;
          break;
        case 'position_trading':
          rsiPeriod = 30;
          smaPeriod = 100;
          emaPeriod = 200;
          break;
      }
      
      // Calcular indicadores técnicos con datos reales
      const rsi = calculateRSI(chartData, rsiPeriod);
      const macd = calculateMACD(chartData);
      const sma20 = calculateSMA(chartData, smaPeriod);
      const ema50 = calculateEMA(chartData, emaPeriod);
      const adx = calculateADX(chartData);
      const currentPrice = chartData[chartData.length - 1].close;
      const volatility = calculateVolatility(chartData);
      const volumeRatio = calculateVolumeRatio(chartData);
      
      // Determinar tendencia
      let trend: 'bullish' | 'bearish' | 'neutral' = 'neutral';
      if (sma20 !== null && ema50 !== null) {
        if (currentPrice > sma20 && sma20 > ema50) {
          trend = 'bullish';
        } else if (currentPrice < sma20 && sma20 < ema50) {
          trend = 'bearish';
        }
      }
      
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
    
    // Recomendaciones basadas en RSI
    if (analysis.rsi !== null) {
      if (analysis.rsi < 30) {
        recommendations.push(`RSI en sobreventa (${analysis.rsi.toFixed(1)}). Considerar compras con stop-loss ajustado`);
      } else if (analysis.rsi > 70) {
        recommendations.push(`RSI en sobrecompra (${analysis.rsi.toFixed(1)}). Considerar ventas o tomar beneficios`);
      } else if (analysis.rsi > 50) {
        recommendations.push(`RSI neutral (${analysis.rsi.toFixed(1)}). Mercado en equilibrio`);
      } else {
        recommendations.push(`RSI débil (${analysis.rsi.toFixed(1)}). Mantener cautela`);
      }
    }
    
    // Recomendaciones basadas en MACD
    if (analysis.macd !== null) {
      if (analysis.macd > 0) {
        recommendations.push('MACD alcista. Momentum positivo en el mercado');
      } else {
        recommendations.push('MACD bajista. Momentum negativo, considerar ventas');
      }
    }
    
    // Recomendaciones basadas en tendencia
    if (analysis.trend === 'bullish') {
      recommendations.push('Tendencia alcista confirmada. Buscar oportunidades de compra');
    } else if (analysis.trend === 'bearish') {
      recommendations.push('Tendencia bajista confirmada. Considerar ventas o esperar');
    } else {
      recommendations.push('Tendencia neutral. Mercado lateral, usar rangos');
    }
    
    // Recomendaciones basadas en volatilidad
    if (analysis.volatility > 3) {
      recommendations.push(`Alta volatilidad (${analysis.volatility.toFixed(1)}%). Usar stop-loss más amplios`);
    } else if (analysis.volatility < 1) {
      recommendations.push(`Baja volatilidad (${analysis.volatility.toFixed(1)}%). Mercado tranquilo, posiciones más pequeñas`);
    } else {
      recommendations.push(`Volatilidad normal (${analysis.volatility.toFixed(1)}%). Operar con parámetros estándar`);
    }
    
    // Recomendaciones basadas en volumen
    if (analysis.volumeRatio > 1.5) {
      recommendations.push('Volumen alto. Confirmación de movimiento de precio');
    } else if (analysis.volumeRatio < 0.7) {
      recommendations.push('Volumen bajo. Falta de confirmación, ser cauteloso');
    } else {
      recommendations.push('Volumen normal. Mercado equilibrado');
    }
    
    // Recomendaciones específicas por tipo de trading
    switch (analysis.tradingType.id) {
      case 'scalping':
        recommendations.push('Para scalping: Usar timeframes de 1-5 minutos');
        recommendations.push('Para scalping: Mantener stop-loss muy ajustado (0.5-1%)');
        recommendations.push('Para scalping: Monitorear volumen en tiempo real');
        break;
      case 'day_trading':
        recommendations.push('Para day trading: Usar múltiples timeframes (M15, H1)');
        recommendations.push('Para day trading: Cerrar posiciones antes del fin de día');
        recommendations.push('Para day trading: Establecer objetivos claros');
        break;
      case 'swing_trading':
        recommendations.push('Para swing trading: Mantener posiciones por días/semanas');
        recommendations.push('Para swing trading: Aplicar gestión de riesgo 1:2');
        recommendations.push('Para swing trading: Usar análisis fundamental complementario');
        break;
      case 'position_trading':
        recommendations.push('Para position trading: Análisis fundamental profundo');
        recommendations.push('Para position trading: Considerar factores macroeconómicos');
        recommendations.push('Para position trading: Gestión de riesgo conservadora');
        break;
    }
    
    // Recomendaciones basadas en ADX
    if (analysis.adx !== null) {
      if (analysis.adx > 25) {
        recommendations.push(`Tendencia fuerte (ADX: ${analysis.adx.toFixed(1)}). Seguir la tendencia`);
      } else {
        recommendations.push(`Tendencia débil (ADX: ${analysis.adx.toFixed(1)}). Mercado lateral`);
      }
    }
    
    return recommendations.slice(0, 8); // Limitar a 8 recomendaciones
  };

  const calculateRealRiskLevel = (volatility: number, tradingType: TradingType): 'low' | 'medium' | 'high' => {
    // Base de riesgo por tipo de trading
    const baseRisk = {
      scalping: 'high',
      day_trading: 'medium',
      swing_trading: 'medium',
      position_trading: 'low'
    } as const;
    
    let riskLevel = baseRisk[tradingType.id as keyof typeof baseRisk];
    
    // Ajustar basado en volatilidad
    if (volatility > 4) {
      riskLevel = 'high';
    } else if (volatility < 1.5) {
      riskLevel = 'low';
    }
    
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
    let confidence = 0.5; // Base 50%
    
    // Factor RSI
    if (analysis.rsi !== null) {
      if (analysis.rsi < 30 || analysis.rsi > 70) {
        confidence += 0.15; // Señales claras
      } else if (analysis.rsi > 40 && analysis.rsi < 60) {
        confidence += 0.05; // Neutral
      }
    }
    
    // Factor MACD
    if (analysis.macd !== null) {
      if (Math.abs(analysis.macd) > 0.001) {
        confidence += 0.10; // Señal clara
      }
    }
    
    // Factor tendencia
    if (analysis.trend !== 'neutral') {
      confidence += 0.15; // Tendencia clara
    }
    
    // Factor ADX
    if (analysis.adx !== null) {
      if (analysis.adx > 25) {
        confidence += 0.10; // Tendencia fuerte
      }
    }
    
    // Factor volumen
    if (analysis.volumeRatio > 1.2) {
      confidence += 0.05; // Confirmación de volumen
    }
    
    // Factor volatilidad (menor volatilidad = mayor confianza)
    if (analysis.volatility < 2) {
      confidence += 0.05;
    } else if (analysis.volatility > 4) {
      confidence -= 0.05;
    }
    
    // Ajuste por tipo de trading
    const tradingTypeConfidence = {
      scalping: 0.65,
      day_trading: 0.75,
      swing_trading: 0.80,
      position_trading: 0.85
    };
    
    confidence = Math.min(confidence, tradingTypeConfidence[analysis.tradingType.id as keyof typeof tradingTypeConfidence]);
    
    return Math.max(0.1, Math.min(0.95, confidence)); // Entre 10% y 95%
  };

  return {
    isAnalyzing,
    lastAnalysis,
    executeAnalysis
  };
}; 