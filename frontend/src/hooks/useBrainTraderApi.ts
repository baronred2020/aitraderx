import { useState, useEffect, useCallback } from 'react';
import { apiService, BrainTraderPrediction, BrainTraderSignal, BrainTraderTrend, MegaMindPrediction, MegaMindCollaboration, MegaMindArena, MegaMindPerformance } from '../services/api';

export interface UseBrainTraderApiReturn {
  // Brain Trader Data
  predictions: BrainTraderPrediction[];
  signals: BrainTraderSignal[];
  trends: BrainTraderTrend[];
  
  // Mega Mind Data
  megaMindPredictions: MegaMindPrediction[];
  megaMindCollaboration: MegaMindCollaboration | null;
  megaMindArena: MegaMindArena | null;
  megaMindPerformance: MegaMindPerformance | null;
  
  // Available brains
  availableBrains: string[];
  defaultBrain: string;
  
  // Loading states
  loading: {
    predictions: boolean;
    signals: boolean;
    trends: boolean;
    megaMind: boolean;
    collaboration: boolean;
    arena: boolean;
    performance: boolean;
    brains: boolean;
  };
  
  // Error states
  errors: {
    predictions: string | null;
    signals: string | null;
    trends: string | null;
    megaMind: string | null;
    collaboration: string | null;
    arena: string | null;
    performance: string | null;
    brains: string | null;
  };
  
  // API functions
  loadPredictions: (brainType: string, pair?: string, style?: string, limit?: number) => Promise<void>;
  loadSignals: (brainType: string, pair?: string, limit?: number) => Promise<void>;
  loadTrends: (brainType: string, pair?: string, limit?: number) => Promise<void>;
  loadMegaMindPredictions: (pair?: string, style?: string, limit?: number) => Promise<void>;
  loadMegaMindCollaboration: (pair?: string) => Promise<void>;
  loadMegaMindArena: (pair?: string) => Promise<void>;
  loadMegaMindPerformance: () => Promise<void>;
  loadAvailableBrains: () => Promise<void>;
  
  // Utility functions
  clearErrors: () => void;
  refreshAll: (brainType: string, pair?: string, style?: string) => Promise<void>;
}

export const useBrainTraderApi = (): UseBrainTraderApiReturn => {
  // Data states
  const [predictions, setPredictions] = useState<BrainTraderPrediction[]>([]);
  const [signals, setSignals] = useState<BrainTraderSignal[]>([]);
  const [trends, setTrends] = useState<BrainTraderTrend[]>([]);
  const [megaMindPredictions, setMegaMindPredictions] = useState<MegaMindPrediction[]>([]);
  const [megaMindCollaboration, setMegaMindCollaboration] = useState<MegaMindCollaboration | null>(null);
  const [megaMindArena, setMegaMindArena] = useState<MegaMindArena | null>(null);
  const [megaMindPerformance, setMegaMindPerformance] = useState<MegaMindPerformance | null>(null);
  const [availableBrains, setAvailableBrains] = useState<string[]>([]);
  const [defaultBrain, setDefaultBrain] = useState<string>('brain_max');

  // Loading states
  const [loading, setLoading] = useState({
    predictions: false,
    signals: false,
    trends: false,
    megaMind: false,
    collaboration: false,
    arena: false,
    performance: false,
    brains: false,
  });

  // Error states
  const [errors, setErrors] = useState<{
    predictions: string | null;
    signals: string | null;
    trends: string | null;
    megaMind: string | null;
    collaboration: string | null;
    arena: string | null;
    performance: string | null;
    brains: string | null;
  }>({
    predictions: null,
    signals: null,
    trends: null,
    megaMind: null,
    collaboration: null,
    arena: null,
    performance: null,
    brains: null,
  });

  // Clear all errors
  const clearErrors = useCallback(() => {
    setErrors({
      predictions: null,
      signals: null,
      trends: null,
      megaMind: null,
      collaboration: null,
      arena: null,
      performance: null,
      brains: null,
    });
  }, []);

  // Load available brains
  const loadAvailableBrains = useCallback(async () => {
    setLoading(prev => ({ ...prev, brains: true }));
    setErrors(prev => ({ ...prev, brains: null }));
    
    try {
      const result = await apiService.getAvailableBrains();
      setAvailableBrains(result.available_brains);
      setDefaultBrain(result.default_brain);
    } catch (error) {
      setErrors(prev => ({ ...prev, brains: error instanceof Error ? error.message : 'Error loading brains' }));
    } finally {
      setLoading(prev => ({ ...prev, brains: false }));
    }
  }, []);

  // Load predictions
  const loadPredictions = useCallback(async (
    brainType: string,
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    limit: number = 5
  ) => {
    setLoading(prev => ({ ...prev, predictions: true }));
    setErrors(prev => ({ ...prev, predictions: null }));
    
    try {
      const result = await apiService.getPredictions(brainType, pair, style, limit);
      setPredictions(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, predictions: error instanceof Error ? error.message : 'Error loading predictions' }));
    } finally {
      setLoading(prev => ({ ...prev, predictions: false }));
    }
  }, []);

  // Load signals
  const loadSignals = useCallback(async (
    brainType: string,
    pair: string = 'EURUSD',
    limit: number = 5
  ) => {
    setLoading(prev => ({ ...prev, signals: true }));
    setErrors(prev => ({ ...prev, signals: null }));
    
    try {
      const result = await apiService.getSignals(brainType, pair, limit);
      setSignals(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, signals: error instanceof Error ? error.message : 'Error loading signals' }));
    } finally {
      setLoading(prev => ({ ...prev, signals: false }));
    }
  }, []);

  // Load trends
  const loadTrends = useCallback(async (
    brainType: string,
    pair: string = 'EURUSD',
    limit: number = 3
  ) => {
    setLoading(prev => ({ ...prev, trends: true }));
    setErrors(prev => ({ ...prev, trends: null }));
    
    try {
      const result = await apiService.getTrends(brainType, pair, limit);
      setTrends(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, trends: error instanceof Error ? error.message : 'Error loading trends' }));
    } finally {
      setLoading(prev => ({ ...prev, trends: false }));
    }
  }, []);

  // Load Mega Mind predictions
  const loadMegaMindPredictions = useCallback(async (
    pair: string = 'EURUSD',
    style: string = 'day_trading',
    limit: number = 5
  ) => {
    setLoading(prev => ({ ...prev, megaMind: true }));
    setErrors(prev => ({ ...prev, megaMind: null }));
    
    try {
      const result = await apiService.getMegaMindPredictions(pair, style, limit);
      setMegaMindPredictions(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, megaMind: error instanceof Error ? error.message : 'Error loading Mega Mind predictions' }));
    } finally {
      setLoading(prev => ({ ...prev, megaMind: false }));
    }
  }, []);

  // Load Mega Mind collaboration
  const loadMegaMindCollaboration = useCallback(async (pair: string = 'EURUSD') => {
    setLoading(prev => ({ ...prev, collaboration: true }));
    setErrors(prev => ({ ...prev, collaboration: null }));
    
    try {
      const result = await apiService.getMegaMindCollaboration(pair);
      setMegaMindCollaboration(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, collaboration: error instanceof Error ? error.message : 'Error loading collaboration' }));
    } finally {
      setLoading(prev => ({ ...prev, collaboration: false }));
    }
  }, []);

  // Load Mega Mind arena
  const loadMegaMindArena = useCallback(async (pair: string = 'EURUSD') => {
    setLoading(prev => ({ ...prev, arena: true }));
    setErrors(prev => ({ ...prev, arena: null }));
    
    try {
      const result = await apiService.getMegaMindArena(pair);
      setMegaMindArena(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, arena: error instanceof Error ? error.message : 'Error loading arena' }));
    } finally {
      setLoading(prev => ({ ...prev, arena: false }));
    }
  }, []);

  // Load Mega Mind performance
  const loadMegaMindPerformance = useCallback(async () => {
    setLoading(prev => ({ ...prev, performance: true }));
    setErrors(prev => ({ ...prev, performance: null }));
    
    try {
      const result = await apiService.getMegaMindPerformance();
      setMegaMindPerformance(result);
    } catch (error) {
      setErrors(prev => ({ ...prev, performance: error instanceof Error ? error.message : 'Error loading performance' }));
    } finally {
      setLoading(prev => ({ ...prev, performance: false }));
    }
  }, []);

  // Refresh all data for a specific brain
  const refreshAll = useCallback(async (
    brainType: string,
    pair: string = 'EURUSD',
    style: string = 'day_trading'
  ) => {
    clearErrors();
    
    if (brainType === 'mega_mind') {
      await Promise.all([
        loadMegaMindPredictions(pair, style),
        loadMegaMindCollaboration(pair),
        loadMegaMindArena(pair),
        loadMegaMindPerformance(),
      ]);
    } else {
      await Promise.all([
        loadPredictions(brainType, pair, style),
        loadSignals(brainType, pair),
        loadTrends(brainType, pair),
      ]);
    }
  }, [clearErrors, loadPredictions, loadSignals, loadTrends, loadMegaMindPredictions, loadMegaMindCollaboration, loadMegaMindArena, loadMegaMindPerformance]);

  // Load available brains on mount
  useEffect(() => {
    loadAvailableBrains();
  }, [loadAvailableBrains]);

  return {
    // Data
    predictions,
    signals,
    trends,
    megaMindPredictions,
    megaMindCollaboration,
    megaMindArena,
    megaMindPerformance,
    availableBrains,
    defaultBrain,
    
    // Loading states
    loading,
    
    // Error states
    errors,
    
    // API functions
    loadPredictions,
    loadSignals,
    loadTrends,
    loadMegaMindPredictions,
    loadMegaMindCollaboration,
    loadMegaMindArena,
    loadMegaMindPerformance,
    loadAvailableBrains,
    
    // Utility functions
    clearErrors,
    refreshAll,
  };
}; 