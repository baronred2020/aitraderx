// API Service for Brain Trader and Mega Mind
const API_BASE_URL = 'http://localhost:8000/api/v1';

export interface BrainTraderPrediction {
  pair: string;
  direction: 'up' | 'down' | 'sideways';
  confidence: number;
  target_price: number;
  timeframe: string;
  reasoning: string;
  brain_type: string;
  timestamp: string;
}

export interface BrainTraderSignal {
  pair: string;
  type: 'buy' | 'sell' | 'hold';
  strength: 'strong' | 'medium' | 'weak';
  confidence: number;
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  brain_type: string;
  timestamp: string;
}

export interface BrainTraderTrend {
  pair: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  strength: number;
  timeframe: string;
  support: number;
  resistance: number;
  description: string;
  brain_type: string;
  timestamp: string;
}

export interface MegaMindPrediction extends BrainTraderPrediction {
  fusion_method: string;
  collaboration_score: number;
  fusion_details: {
    brain_max_confidence: number;
    brain_ultra_confidence: number;
    brain_predictor_confidence: number;
    consensus_level: number;
    collaboration_boost: number;
  };
}

export interface MegaMindCollaboration {
  pair: string;
  collaboration_score: number;
  brain_contributions: {
    brain_max: { contribution: number; confidence: number };
    brain_ultra: { contribution: number; confidence: number };
    brain_predictor: { contribution: number; confidence: number };
  };
  consensus_level: number;
  collaboration_status: 'optimal' | 'good' | 'improving' | 'needs_attention';
  timestamp: string;
}

export interface MegaMindArena {
  pair: string;
  arena_results: {
    brain_max: { wins: number; accuracy: number; performance: number };
    brain_ultra: { wins: number; accuracy: number; performance: number };
    brain_predictor: { wins: number; accuracy: number; performance: number };
    mega_mind: { wins: number; accuracy: number; performance: number };
  };
  winner: string;
  total_rounds: number;
  timestamp: string;
}

export interface MegaMindPerformance {
  overall_accuracy: number;
  fusion_effectiveness: number;
  collaboration_score: number;
  brain_performance: {
    brain_max: { accuracy: number; reliability: number };
    brain_ultra: { accuracy: number; reliability: number };
    brain_predictor: { accuracy: number; reliability: number };
  };
  evolution_status: 'evolving' | 'stable' | 'optimizing';
  last_optimization: string;
}

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API Error (${endpoint}):`, error);
      throw error;
    }
  }

  // Brain Trader APIs
  async getAvailableBrains(): Promise<{ available_brains: string[]; default_brain: string }> {
    return this.request('/brain-trader/available-brains');
  }

  async getPredictions(
    brainType: string,
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    limit: number = 5
  ): Promise<BrainTraderPrediction[]> {
    return this.request(`/brain-trader/predictions/${brainType}?pair=${pair}&style=${style}&limit=${limit}`);
  }

  async getSignals(
    brainType: string,
    pair: string = 'EURUSD',
    limit: number = 5
  ): Promise<BrainTraderSignal[]> {
    return this.request(`/brain-trader/signals/${brainType}?pair=${pair}&limit=${limit}`);
  }

  async getTrends(
    brainType: string,
    pair: string = 'EURUSD',
    limit: number = 3
  ): Promise<BrainTraderTrend[]> {
    return this.request(`/brain-trader/trends/${brainType}?pair=${pair}&limit=${limit}`);
  }

  // Mega Mind APIs
  async getMegaMindPredictions(
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    limit: number = 5
  ): Promise<MegaMindPrediction[]> {
    return this.request(`/mega-mind/predictions?pair=${pair}&style=${style}&limit=${limit}`);
  }

  async getMegaMindCollaboration(pair: string = 'EURUSD'): Promise<MegaMindCollaboration> {
    return this.request(`/mega-mind/collaboration?pair=${pair}`);
  }

  async getMegaMindArena(pair: string = 'EURUSD'): Promise<MegaMindArena> {
    return this.request(`/mega-mind/arena?pair=${pair}`);
  }

  async getMegaMindPerformance(): Promise<MegaMindPerformance> {
    return this.request('/mega-mind/performance');
  }

  // Health check
  async getHealth(): Promise<{ status: string; service: string; version: string; timestamp: string }> {
    return this.request('/health');
  }
}

export const apiService = new ApiService(); 