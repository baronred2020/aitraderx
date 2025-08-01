// API Service for Brain Trader and Mega Mind
const API_BASE_URL = 'http://localhost:8000/api/v1';

export interface BrainTraderPrediction {
  pair: string;
  direction: 'up' | 'down' | 'sideways';
  confidence: number;
  precision: number;
  win_rate: number;
  timeframe: string;
  reasoning: string;
  brain_type: string;
  timestamp: string;
  expires_at: string;
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

export interface PredictionHistoryItem {
  id: number;
  pair: string;
  direction: string;
  current_price: number;
  confidence: number;
  precision: number;
  win_rate: number;
  timeframe: string;
  reasoning: string;
  brain_type: string;  // ✅ Agregada propiedad brain_type
  created_at: string;
  expires_at: string;
  is_completed: boolean;
  actual_price_at_expiry?: number;
  prediction_success?: boolean;
  success_percentage?: number;
}

export interface PredictionLimits {
  can_generate: boolean;
  remaining_predictions: number;
  max_predictions_per_day: number;
  has_active_prediction: boolean;
  active_prediction_expires?: string;
  plan_type: string;
  analysis_type: string;
  timeframe: string;
  duration_minutes: number;
  has_unlimited?: boolean;
}

export interface SignalLimits {
  can_generate: boolean;
  remaining_signals: number;
  max_signals_per_day: number;
  plan_type: string;
  style: string;
  timeframe: string;
  has_unlimited: boolean;
}

export interface UserStats {
  total_predictions: number;
  successful_predictions: number;
  success_rate: number;
  average_success_percentage: number;
  best_pair?: string;
  total_predictions_today: number;
}

export interface RealMetrics {
  total_predictions: number;
  successful_predictions: number;
  win_rate: number;
  precision: number;
  average_confidence: number;
  average_success_percentage: number;
  best_pair: string | null;
  best_brain_type: string | null;
  recent_performance: Array<{
    id: number;
    pair: string;
    direction: string;
    prediction_success: boolean;
    success_percentage: number;
    confidence: number;
    brain_type: string;
    created_at: string;
  }>;
  metrics_by_pair: Record<string, {
    total_predictions: number;
    successful_predictions: number;
    win_rate: number;
    precision: number;
    average_confidence: number;
  }>;
  metrics_by_brain: Record<string, {
    total_predictions: number;
    successful_predictions: number;
    win_rate: number;
    precision: number;
    average_confidence: number;
  }>;
}

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    // Asegurar que el endpoint no comience con /api/v1 para evitar duplicación
    const cleanEndpoint = endpoint.startsWith('/api/v1') ? endpoint.substring(7) : endpoint;
    const url = `${API_BASE_URL}${cleanEndpoint}`;
    
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
  async getAvailableBrains(planType: string = 'starter'): Promise<{ available_brains: string[]; default_brain: string }> {
    return this.request(`/brain-trader/available-brains?plan_type=${planType}`);
  }

  async getPredictions(
    brainType: string,
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    limit: number = 5,
    planType: string = 'starter'
  ): Promise<BrainTraderPrediction[]> {
    return this.request(`/brain-trader/predictions/${brainType}?pair=${pair}&style=${style}&limit=${limit}&plan_type=${planType}`);
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

  // ===== PREDICTION-SPECIFIC APIs =====

  async generatePrediction(
    pair: string,
    brainType: string = 'brain_max',
    style: string = 'day_trading'
  ): Promise<{ success: boolean; prediction?: any; limits?: PredictionLimits; error?: string }> {
    try {
      const response = await this.request<{
        success: boolean;
        prediction: any;
        limits: PredictionLimits;
        error?: string;
      }>('/api/v1/predictions/generate', {
        method: 'POST',
        body: JSON.stringify({
          pair,
          brain_type: brainType,
          style
        })
      });
      
      if (response && response.success) {
        return {
          success: true,
          prediction: response.prediction,
          limits: response.limits
        };
      } else {
        return {
          success: false,
          error: response?.error || 'No se pudo generar la predicción'
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Error generando predicción'
      };
    }
  }

  async getPredictionHistory(limit: number = 20): Promise<PredictionHistoryItem[]> {
    try {
      const response = await this.request<PredictionHistoryItem[]>(`/api/v1/predictions/history?limit=${limit}`);
      return response || [];
    } catch (error) {
      console.error('Error getting prediction history:', error);
      return [];
    }
  }

  async getPredictionLimits(style: string = 'day_trading'): Promise<PredictionLimits> {
    return this.request(`/api/v1/predictions/limits?style=${style}`);
  }

  async getActivePrediction(style: string = 'day_trading'): Promise<any> {
    return this.request(`/api/v1/predictions/active?style=${style}`);
  }

  async getUserStats(): Promise<UserStats> {
    return this.request('/api/v1/predictions/stats');
  }

  // ✅ Completar predicciones expiradas
  async completeExpiredPredictions(): Promise<{ success: boolean; message: string; completed: number; total: number }> {
    return this.request('/api/v1/predictions/complete-expired', {
      method: 'POST'
    });
  }

  // Generate manual signal
  async generateSignal(
    brainType: string,
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    userId?: string,
    planType: string = 'starter'
  ): Promise<{
    success: boolean;
    signal?: BrainTraderSignal;
    signal_quality?: number;
    quality_score?: number;
    reasoning?: string;
    message?: string;
    next_interval?: string;
    indicators_used?: string[];
    generated_at?: string;
    style?: string;
    timeframe?: string;
    signal_type?: string;
    current_price?: number;
    stop_loss?: number;
    take_profit?: string;
    brain_type?: string;
    timestamp?: string;
    // Campos adicionales que devuelve el backend
    pair?: string;
    strength?: string;
    confidence?: number;
    entry_price?: number;
    // Campos de límites de señales
    remaining_signals?: number;
    max_signals_per_day?: number;
    upgrade_required?: boolean;
  }> {
    const params = new URLSearchParams({
      pair,
      style,
      ...(userId && { user_id: userId }),
      plan_type: planType
    });
    
    return this.request(`/brain-trader/signals/${brainType}/generate?${params.toString()}`, {
      method: 'POST'
    });
  }

  // Get signal intervals information
  async getSignalLimits(
    brainType: string,
    userId: string,
    planType: string = 'starter',
    style: string = 'day_trading'
  ): Promise<SignalLimits> {
    const params = new URLSearchParams({
      user_id: userId,
      plan_type: planType,
      style
    });
    return this.request(`/brain-trader/signals/${brainType}/limits?${params.toString()}`);
  }

  async getSignalIntervals(
    brainType: string,
    style: string = 'day_trading'
  ): Promise<{
    style: string;
    timeframe: string;
    current_time: string;
    is_valid_time: boolean;
    next_interval: string;
    upcoming_intervals: string[];
    duration_minutes: number;
  }> {
    return this.request(`/brain-trader/signals/${brainType}/intervals?style=${style}`);
  }

  // Nuevas funciones para predicciones con intervalos
  async getPredictionsWithIntervals(
    brainType: string,
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    limit: number = 5,
    planType: string = 'starter'
  ): Promise<BrainTraderPrediction[]> {
    return this.request(`/brain-trader/predictions/${brainType}/with-intervals?pair=${pair}&style=${style}&limit=${limit}&plan_type=${planType}`);
  }

  async getNextPredictionTime(
    brainType: string,
    style: string = 'day_trading'
  ): Promise<{
    current_time: string;
    next_interval: string;
    time_until_next_seconds: number;
    time_until_next_minutes: number;
    is_valid_now: boolean;
    style: string;
    timeframe: string;
  }> {
    return this.request(`/brain-trader/predictions/${brainType}/next-interval?style=${style}`);
  }

  // ===== MÉTRICAS REALES APIs =====

  async getRealMetrics(
    brainType?: string,
    pair?: string,
    style?: string
  ): Promise<RealMetrics> {
    const params = new URLSearchParams();
    if (brainType) params.append('brain_type', brainType);
    if (pair) params.append('pair', pair);
    if (style) params.append('style', style);
    
    return this.request(`/predictions/real-metrics?${params.toString()}`);
  }

  async completeExpiredPredictionsWithRealResults(): Promise<{
    total_expired: number;
    completed: number;
    failed: number;
  }> {
    return this.request('/predictions/complete-expired-with-real-results', {
      method: 'POST'
    });
  }

  // ===== FIN PREDICTION-SPECIFIC APIs =====
}

export const apiService = new ApiService(); 