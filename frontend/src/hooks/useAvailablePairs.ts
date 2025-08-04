import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { apiService } from '../services/api';

export interface AvailablePair {
  symbol: string;
  name: string;
  category: string;
  spread: number;
  description: string;
}

export interface AvailablePairsResponse {
  plan_type: string;
  available_pairs: AvailablePair[];
  total_pairs: number;
  categories: {
    Major: number;
    Minor: number;
  };
}

export const useAvailablePairs = () => {
  const { subscription } = useAuth();
  const [pairs, setPairs] = useState<AvailablePair[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [categories, setCategories] = useState<{ Major: number; Minor: number }>({ Major: 0, Minor: 0 });

  const loadAvailablePairs = async (planType?: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const plan = planType || subscription?.planType || 'starter';
      const response = await apiService.getAvailablePairs(plan);
      
      setPairs(response.available_pairs);
      setCategories(response.categories);
    } catch (err) {
      console.error('Error loading available pairs:', err);
      setError('Error al cargar los pares disponibles');
      
      // Fallback con pares básicos
      setPairs([
        {
          symbol: 'EURUSD',
          name: 'Euro/Dólar',
          category: 'Major',
          spread: 1.5,
          description: 'Par más líquido del mercado'
        }
      ]);
      setCategories({ Major: 1, Minor: 0 });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAvailablePairs();
  }, [subscription?.planType]);

  const getPairsByCategory = (category: string) => {
    return pairs.filter(pair => pair.category === category);
  };

  const getPairBySymbol = (symbol: string) => {
    return pairs.find(pair => pair.symbol === symbol);
  };

  const getDefaultPair = () => {
    return pairs[0] || {
      symbol: 'EURUSD',
      name: 'Euro/Dólar',
      category: 'Major',
      spread: 1.5,
      description: 'Par más líquido del mercado'
    };
  };

  return {
    pairs,
    loading,
    error,
    categories,
    loadAvailablePairs,
    getPairsByCategory,
    getPairBySymbol,
    getDefaultPair
  };
}; 