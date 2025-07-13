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
}

export const useIntelligentAnalysis = () => {
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [lastAnalysis, setLastAnalysis] = useState<AnalysisResult | null>(null);

  const executeAnalysis = async (tradingType: TradingType, symbol: string): Promise<AnalysisResult> => {
    setIsAnalyzing(true);
    
    try {
      // Simular llamada al backend para análisis inteligente
      console.log(`Ejecutando análisis inteligente para ${symbol} con tipo: ${tradingType.name}`);
      
      // Simular delay de análisis
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Generar resultados simulados según el tipo de trading
      const recommendations = generateRecommendations(tradingType, symbol);
      const riskLevel = calculateRiskLevel(tradingType);
      const confidence = calculateConfidence(tradingType);
      
      const result: AnalysisResult = {
        tradingType,
        recommendations,
        riskLevel,
        confidence,
        timestamp: new Date()
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

  const generateRecommendations = (tradingType: TradingType, symbol: string): string[] => {
    const recommendations: string[] = [];
    
    switch (tradingType.id) {
      case 'scalping':
        recommendations.push(
          'Usar indicadores de momentum (RSI, MACD)',
          'Mantener stop-loss muy ajustado',
          'Monitorear volumen en tiempo real',
          'Evitar operar en noticias importantes'
        );
        break;
      case 'day_trading':
        recommendations.push(
          'Analizar soportes y resistencias',
          'Usar múltiples timeframes (M15, H1)',
          'Establecer objetivos claros',
          'Cerrar posiciones antes del fin de día'
        );
        break;
      case 'swing_trading':
        recommendations.push(
          'Identificar tendencias principales',
          'Usar análisis fundamental',
          'Mantener posiciones por días/semanas',
          'Aplicar gestión de riesgo 1:2'
        );
        break;
      case 'position_trading':
        recommendations.push(
          'Análisis fundamental profundo',
          'Considerar factores macroeconómicos',
          'Posiciones de largo plazo',
          'Gestión de riesgo conservadora'
        );
        break;
    }
    
    return recommendations;
  };

  const calculateRiskLevel = (tradingType: TradingType): 'low' | 'medium' | 'high' => {
    switch (tradingType.id) {
      case 'scalping':
        return 'high';
      case 'day_trading':
        return 'medium';
      case 'swing_trading':
        return 'medium';
      case 'position_trading':
        return 'low';
      default:
        return 'medium';
    }
  };

  const calculateConfidence = (tradingType: TradingType): number => {
    // Simular nivel de confianza basado en el tipo de trading
    const baseConfidence = {
      scalping: 0.65,
      day_trading: 0.75,
      swing_trading: 0.80,
      position_trading: 0.85
    };
    
    return baseConfidence[tradingType.id as keyof typeof baseConfidence] || 0.70;
  };

  return {
    isAnalyzing,
    lastAnalysis,
    executeAnalysis
  };
}; 