import React, { createContext, useContext, useState, useEffect, ReactNode, useMemo, useCallback } from 'react';

export interface User {
  id: string;
  username: string;
  email: string;
  role: 'admin' | 'user';
  isActive: boolean;
}

export interface Subscription {
  id: string;
  planType: 'freemium' | 'basic' | 'pro' | 'elite';
  status: 'active' | 'expired' | 'cancelled' | 'trial';
  startDate: string;
  endDate: string;
  isTrial: boolean;
}

export interface AuthContextType {
  user: User | null;
  subscription: Subscription | null;
  isLoading: boolean;
  login: (username: string, password: string) => Promise<boolean>;
  logout: () => void;
  checkSubscription: () => Promise<void>;
  hasFeature: (feature: string) => boolean;
  canAccess: (section: string) => boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  // Mapeo de características por plan - mover fuera del componente o memoizar
  const planFeatures = useMemo(() => ({
    freemium: {
      features: ['basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis'],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis']
    },
    basic: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts']
    },
    pro: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts',
        'ai_monitor', 'reinforcement_learning', 'reports', 'mt4_integration'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts', 'ai-monitor', 'rl', 'reports']
    },
    elite: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts',
        'ai_monitor', 'reinforcement_learning', 'reports', 'mt4_integration',
        'api_access', 'custom_models', 'priority_support'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts', 'ai-monitor', 'rl', 'reports', 'community']
    }
  }), []);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    try {
      setIsLoading(true);
      
      // Intentar llamada a API
      try {
        const response = await fetch('http://localhost:8000/api/auth/login', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ username, password }),
        });

        if (response.ok) {
          const data = await response.json();
          setUser(data.user);
          setSubscription(data.subscription);
          localStorage.setItem('auth_token', data.token);
          return true;
        }
      } catch (apiError) {
        console.warn('API no disponible, usando modo desarrollo:', apiError);
      }
      
      // Fallback para desarrollo - usar credenciales por defecto
      if (username === 'admin' && password === 'admin123') {
        const defaultUser: User = {
          id: '1',
          username: 'admin',
          email: 'admin@aitraderx.com',
          role: 'admin',
          isActive: true
        };
        
        const defaultSubscription: Subscription = {
          id: '1',
          planType: 'elite',
          status: 'active',
          startDate: new Date().toISOString(),
          endDate: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString(),
          isTrial: false
        };
        
        setUser(defaultUser);
        setSubscription(defaultSubscription);
        localStorage.setItem('auth_token', 'dev-token');
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setSubscription(null);
    localStorage.removeItem('auth_token');
  }, []);

  const checkSubscription = useCallback(async () => {
    try {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        setIsLoading(false);
        return;
      }

      // Si es token de desarrollo, no hacer llamada al backend
      if (token === 'dev-token') {
        console.log('Usando modo desarrollo, saltando verificación de suscripción');
        setIsLoading(false);
        return;
      }

      const response = await fetch('http://localhost:8000/api/subscriptions/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setSubscription(data.subscription);
      } else if (response.status === 401) {
        // Token inválido o expirado, limpiar y continuar
        console.warn('Token inválido, limpiando sesión');
        localStorage.removeItem('auth_token');
        setUser(null);
        setSubscription(null);
      } else {
        console.error('Error checking subscription:', response.status, response.statusText);
      }
    } catch (error) {
      console.error('Error checking subscription:', error);
      // En caso de error de red, mantener el estado actual pero no bloquear
    } finally {
      setIsLoading(false);
    }
  }, []);

  const hasFeature = useCallback((feature: string): boolean => {
    if (!subscription || subscription.status !== 'active') {
      return false;
    }

    const plan = planFeatures[subscription.planType];
    return plan?.features.includes(feature) || false;
  }, [subscription, planFeatures]);

  const canAccess = useCallback((section: string): boolean => {
    if (!subscription || subscription.status !== 'active') {
      return section === 'dashboard'; // Solo dashboard para usuarios sin suscripción
    }

    const plan = planFeatures[subscription.planType];
    return plan?.sections.includes(section) || false;
  }, [subscription, planFeatures]);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      checkSubscription();
    } else {
      setIsLoading(false);
    }
  }, [checkSubscription]);

  // Memoizar el valor del contexto para evitar re-renders innecesarios
  const value = useMemo<AuthContextType>(() => ({
    user,
    subscription,
    isLoading,
    login,
    logout,
    checkSubscription,
    hasFeature,
    canAccess,
  }), [user, subscription, isLoading, login, logout, checkSubscription, hasFeature, canAccess]);

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 