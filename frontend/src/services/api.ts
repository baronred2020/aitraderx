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

// ===== INTERFACES PARA AGENTES DE MONITOREO =====

export interface MonitoringAlert {
  id: string;
  agent_type: 'technical' | 'ai' | 'risk' | 'temporal' | 'fundamental';
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: string;
  title: string;
  description: string;
  pair?: string;
  brain_type?: string;
  timestamp: string;
  is_read: boolean;
  action_required: boolean;
  metadata?: Record<string, any>;
}

export interface MonitoringAgentStatus {
  agent_type: 'technical' | 'ai' | 'risk' | 'temporal' | 'fundamental';
  is_active: boolean;
  last_check: string;
  alerts_count: number;
  performance_score: number;
  status: 'monitoring' | 'idle' | 'error' | 'maintenance';
}

export interface MonitoringSystemStatus {
  overall_status: 'healthy' | 'warning' | 'critical';
  active_agents: number;
  total_alerts: number;
  unread_alerts: number;
  critical_alerts: number;
  last_update: string;
  agents_status: MonitoringAgentStatus[];
}

export interface MonitoringConfig {
  enabled: boolean;
  check_interval: number; // segundos
  alert_retention_days: number;
  max_alerts_per_agent: number;
  subscription_limits: {
    starter: { max_alerts: number; max_agents: number };
    trader: { max_alerts: number; max_agents: number };
    expert: { max_alerts: number; max_agents: number };
    premium: { max_alerts: number; max_agents: number };
    institutional: { max_alerts: number; max_agents: number };
  };
}

// ===== FIN INTERFACES MONITOREO =====

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

  // ===== AGENTES DE MONITOREO APIs =====

  // Obtener alertas de monitoreo
  async getMonitoringAlerts(
    agent_type?: string,
    severity?: string,
    limit: number = 50
  ): Promise<MonitoringAlert[]> {
    const params = new URLSearchParams();
    if (agent_type) params.append('agent_type', agent_type);
    if (severity) params.append('severity', severity);
    params.append('limit', limit.toString());
    
    return this.request(`/brain-trader/monitoring/alerts?${params.toString()}`);
  }

  // Marcar alerta como leída
  async markAlertAsRead(alert_id: string): Promise<{ success: boolean }> {
    return this.request(`/brain-trader/monitoring/alerts/${alert_id}/read`, {
      method: 'PUT'
    });
  }

  // Obtener estado del sistema de monitoreo
  async getMonitoringSystemStatus(): Promise<MonitoringSystemStatus> {
    return this.request('/brain-trader/monitoring/status');
  }

  // Obtener configuración de monitoreo
  async getMonitoringConfig(): Promise<MonitoringConfig> {
    return this.request('/brain-trader/monitoring/config');
  }

  // Actualizar configuración de monitoreo
  async updateMonitoringConfig(config: Partial<MonitoringConfig>): Promise<MonitoringConfig> {
    return this.request('/brain-trader/monitoring/config', {
      method: 'PUT',
      body: JSON.stringify(config)
    });
  }

  // Iniciar monitoreo para un par específico
  async startMonitoring(pair: string, brain_type?: string): Promise<{ success: boolean; message: string }> {
    const params = new URLSearchParams({ pair });
    if (brain_type) params.append('brain_type', brain_type);
    
    return this.request(`/brain-trader/monitoring/start?${params.toString()}`, {
      method: 'POST'
    });
  }

  // Detener monitoreo para un par específico
  async stopMonitoring(pair: string): Promise<{ success: boolean; message: string }> {
    return this.request(`/brain-trader/monitoring/stop?pair=${pair}`, {
      method: 'POST'
    });
  }

  // ===== FIN AGENTES DE MONITOREO APIs =====
}

export const apiService = new ApiService(); 