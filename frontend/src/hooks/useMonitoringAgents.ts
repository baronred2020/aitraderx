import { useState, useEffect, useCallback } from 'react';
import { 
  apiService, 
  MonitoringAlert, 
  MonitoringSystemStatus, 
  MonitoringConfig,
  MonitoringAgentStatus 
} from '../services/api';
import { useAuth } from '../contexts/AuthContext';
import { useFeatureAccess } from './useFeatureAccess';

export interface UseMonitoringAgentsReturn {
  // Data states
  alerts: MonitoringAlert[];
  systemStatus: MonitoringSystemStatus | null;
  config: MonitoringConfig | null;
  agentsStatus: MonitoringAgentStatus[];
  
  // Loading states
  loading: {
    alerts: boolean;
    systemStatus: boolean;
    config: boolean;
    startMonitoring: boolean;
    stopMonitoring: boolean;
  };
  
  // Error states
  errors: {
    alerts: string | null;
    systemStatus: string | null;
    config: string | null;
    startMonitoring: string | null;
    stopMonitoring: string | null;
  };
  
  // API functions
  loadAlerts: (agent_type?: string, severity?: string, limit?: number) => Promise<void>;
  loadSystemStatus: () => Promise<void>;
  loadConfig: () => Promise<void>;
  markAlertAsRead: (alert_id: string) => Promise<void>;
  startMonitoring: (pair: string, brain_type?: string) => Promise<void>;
  stopMonitoring: (pair: string) => Promise<void>;
  updateConfig: (config: Partial<MonitoringConfig>) => Promise<void>;
  
  // Utility functions
  clearErrors: () => void;
  refreshAll: () => Promise<void>;
  getAlertsBySeverity: (severity: string) => MonitoringAlert[];
  getAlertsByAgent: (agent_type: string) => MonitoringAlert[];
  getUnreadAlertsCount: () => number;
  getCriticalAlertsCount: () => number;
  isMonitoringAvailable: () => boolean;
  getSubscriptionLimits: () => { max_alerts: number; max_agents: number } | null;
}

export const useMonitoringAgents = (): UseMonitoringAgentsReturn => {
  const { subscription } = useAuth();
  const { checkFeature } = useFeatureAccess();
  
  // Data states
  const [alerts, setAlerts] = useState<MonitoringAlert[]>([]);
  const [systemStatus, setSystemStatus] = useState<MonitoringSystemStatus | null>(null);
  const [config, setConfig] = useState<MonitoringConfig | null>(null);
  const [agentsStatus, setAgentsStatus] = useState<MonitoringAgentStatus[]>([]);
  
  // Loading states
  const [loading, setLoading] = useState({
    alerts: false,
    systemStatus: false,
    config: false,
    startMonitoring: false,
    stopMonitoring: false,
  });
  
  // Error states
  const [errors, setErrors] = useState<{
    alerts: string | null;
    systemStatus: string | null;
    config: string | null;
    startMonitoring: string | null;
    stopMonitoring: string | null;
  }>({
    alerts: null,
    systemStatus: null,
    config: null,
    startMonitoring: null,
    stopMonitoring: null,
  });
  
  // Load alerts
  const loadAlerts = useCallback(async (
    agent_type?: string, 
    severity?: string, 
    limit: number = 50
  ) => {
    if (!isMonitoringAvailable()) return;
    
    setLoading(prev => ({ ...prev, alerts: true }));
    setErrors(prev => ({ ...prev, alerts: null }));
    
    try {
      const alertsData = await apiService.getMonitoringAlerts(agent_type, severity, limit);
      setAlerts(alertsData);
    } catch (error) {
      setErrors(prev => ({ 
        ...prev, 
        alerts: error instanceof Error ? error.message : 'Error loading alerts' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, alerts: false }));
    }
  }, []);
  
  // Load system status
  const loadSystemStatus = useCallback(async () => {
    if (!isMonitoringAvailable()) return;
    
    setLoading(prev => ({ ...prev, systemStatus: true }));
    setErrors(prev => ({ ...prev, systemStatus: null }));
    
    try {
      const statusData = await apiService.getMonitoringSystemStatus();
      setSystemStatus(statusData);
      setAgentsStatus(statusData.agents_status);
    } catch (error) {
      setErrors(prev => ({ 
        ...prev, 
        systemStatus: error instanceof Error ? error.message : 'Error loading system status' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, systemStatus: false }));
    }
  }, []);
  
  // Load config
  const loadConfig = useCallback(async () => {
    if (!isMonitoringAvailable()) return;
    
    setLoading(prev => ({ ...prev, config: true }));
    setErrors(prev => ({ ...prev, config: null }));
    
    try {
      const configData = await apiService.getMonitoringConfig();
      setConfig(configData);
    } catch (error) {
      setErrors(prev => ({ 
        ...prev, 
        config: error instanceof Error ? error.message : 'Error loading config' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, config: false }));
    }
  }, []);
  
  // Mark alert as read
  const markAlertAsRead = useCallback(async (alert_id: string) => {
    try {
      await apiService.markAlertAsRead(alert_id);
      setAlerts(prev => 
        prev.map(alert => 
          alert.id === alert_id ? { ...alert, is_read: true } : alert
        )
      );
    } catch (error) {
      console.error('Error marking alert as read:', error);
    }
  }, []);
  
  // Start monitoring
  const startMonitoring = useCallback(async (pair: string, brain_type?: string) => {
    setLoading(prev => ({ ...prev, startMonitoring: true }));
    setErrors(prev => ({ ...prev, startMonitoring: null }));
    
    try {
      await apiService.startMonitoring(pair, brain_type);
      await loadSystemStatus(); // Refresh status
    } catch (error) {
      setErrors(prev => ({ 
        ...prev, 
        startMonitoring: error instanceof Error ? error.message : 'Error starting monitoring' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, startMonitoring: false }));
    }
  }, [loadSystemStatus]);
  
  // Stop monitoring
  const stopMonitoring = useCallback(async (pair: string) => {
    setLoading(prev => ({ ...prev, stopMonitoring: true }));
    setErrors(prev => ({ ...prev, stopMonitoring: null }));
    
    try {
      await apiService.stopMonitoring(pair);
      await loadSystemStatus(); // Refresh status
    } catch (error) {
      setErrors(prev => ({ 
        ...prev, 
        stopMonitoring: error instanceof Error ? error.message : 'Error stopping monitoring' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, stopMonitoring: false }));
    }
  }, [loadSystemStatus]);
  
  // Update config
  const updateConfig = useCallback(async (newConfig: Partial<MonitoringConfig>) => {
    if (!config) return;
    
    setLoading(prev => ({ ...prev, config: true }));
    setErrors(prev => ({ ...prev, config: null }));
    
    try {
      const updatedConfig = await apiService.updateMonitoringConfig(newConfig);
      setConfig(updatedConfig);
    } catch (error) {
      setErrors(prev => ({ 
        ...prev, 
        config: error instanceof Error ? error.message : 'Error updating config' 
      }));
    } finally {
      setLoading(prev => ({ ...prev, config: false }));
    }
  }, [config]);
  
  // Clear errors
  const clearErrors = useCallback(() => {
    setErrors({
      alerts: null,
      systemStatus: null,
      config: null,
      startMonitoring: null,
      stopMonitoring: null,
    });
  }, []);
  
  // Refresh all data
  const refreshAll = useCallback(async () => {
    await Promise.all([
      loadAlerts(),
      loadSystemStatus(),
      loadConfig()
    ]);
  }, [loadAlerts, loadSystemStatus, loadConfig]);
  
  // Utility functions
  const getAlertsBySeverity = useCallback((severity: string) => {
    return alerts.filter(alert => alert.severity === severity);
  }, [alerts]);
  
  const getAlertsByAgent = useCallback((agent_type: string) => {
    return alerts.filter(alert => alert.agent_type === agent_type);
  }, [alerts]);
  
  const getUnreadAlertsCount = useCallback(() => {
    return alerts.filter(alert => !alert.is_read).length;
  }, [alerts]);
  
  const getCriticalAlertsCount = useCallback(() => {
    return alerts.filter(alert => alert.severity === 'critical').length;
  }, [alerts]);
  
  const isMonitoringAvailable = useCallback(() => {
    return checkFeature('monitoring_agents');
  }, [checkFeature]);
  
  const getSubscriptionLimits = useCallback(() => {
    if (!config || !subscription) return null;
    
    const planType = subscription.planType;
    return config.subscription_limits[planType as keyof typeof config.subscription_limits] || null;
  }, [config, subscription]);
  
  // Initial load
  useEffect(() => {
    if (isMonitoringAvailable()) {
      refreshAll();
    }
  }, [isMonitoringAvailable, refreshAll]);
  
  return {
    // Data states
    alerts,
    systemStatus,
    config,
    agentsStatus,
    
    // Loading states
    loading,
    
    // Error states
    errors,
    
    // API functions
    loadAlerts,
    loadSystemStatus,
    loadConfig,
    markAlertAsRead,
    startMonitoring,
    stopMonitoring,
    updateConfig,
    
    // Utility functions
    clearErrors,
    refreshAll,
    getAlertsBySeverity,
    getAlertsByAgent,
    getUnreadAlertsCount,
    getCriticalAlertsCount,
    isMonitoringAvailable,
    getSubscriptionLimits,
  };
}; 