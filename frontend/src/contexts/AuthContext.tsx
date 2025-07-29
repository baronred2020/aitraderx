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
  planType: 'starter' | 'trader' | 'expert' | 'premium' | 'institutional';
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
    starter: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis', 
        'brain_trader_basic', 'subscription_management', 'help_support'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'brain-trader', 'subscriptions', 'help']
    },
    trader: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts',
        'brain_trader_advanced', 'monitoring_agents', 'monitoring_alerts',
        'subscription_management', 'help_support', 'community_access'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts', 'brain-trader', 'subscriptions', 'help', 'community']
    },
    expert: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts',
        'ai_monitor', 'reinforcement_learning', 'reports', 'mt4_integration',
        'brain_trader_pro', 'monitoring_agents', 'monitoring_alerts', 'monitoring_config',
        'subscription_management', 'help_support', 'community_access'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts', 'ai-monitor', 'rl', 'reports', 'brain-trader', 'subscriptions', 'help', 'community']
    },
    premium: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts',
        'ai_monitor', 'reinforcement_learning', 'reports', 'mt4_integration',
        'api_access', 'custom_models', 'priority_support', 'brain_trader_premium',
        'monitoring_agents', 'monitoring_alerts', 'monitoring_config',
        'subscription_management', 'help_support', 'community_access', 'mega_mind_institutional'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts', 'ai-monitor', 'rl', 'reports', 'community', 'brain-trader', 'mega-mind', 'subscriptions', 'help']
    },
    institutional: {
      features: [
        'basic_dashboard', 'basic_trading', 'basic_portfolio', 'basic_analysis',
        'advanced_trading', 'advanced_portfolio', 'advanced_analysis', 'alerts',
        'ai_monitor', 'reinforcement_learning', 'reports', 'mt4_integration',
        'api_access', 'custom_models', 'priority_support', 'brain_trader_premium',
        'mega_mind', 'institutional_features', 'dedicated_support',
        'monitoring_agents', 'monitoring_alerts', 'monitoring_config',
        'subscription_management', 'help_support', 'community_access'
      ],
      sections: ['dashboard', 'trading', 'portfolio', 'analysis', 'alerts', 'ai-monitor', 'rl', 'reports', 'community', 'brain-trader', 'mega-mind', 'subscriptions', 'help']
    }
  }), []);

  const login = useCallback(async (username: string, password: string): Promise<boolean> => {
    try {
      console.log('🔐 Iniciando login con backend real...');
      
      const response = await fetch('http://localhost:8000/api/auth/login', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Login exitoso:', data);
        console.log('🔍 Datos de suscripción recibidos:', data.subscription);
        
        setUser(data.user);
        setSubscription(data.subscription);
        localStorage.setItem('auth_token', data.token);
        
        // Verificar que la suscripción se estableció correctamente
        console.log('🔍 Estado después del login - User:', data.user);
        console.log('🔍 Estado después del login - Subscription:', data.subscription);
        
        return true;
      } else {
        console.error('❌ Error en login:', response.status, response.statusText);
        const errorData = await response.json().catch(() => ({}));
        console.error('📄 Detalles del error:', errorData);
        return false;
      }
    } catch (error) {
      console.error('❌ Error de conexión:', error);
      return false;
    }
  }, []);

  const logout = useCallback(() => {
    setUser(null);
    setSubscription(null);
    localStorage.removeItem('auth_token');
  }, []);

  const checkSubscription = useCallback(async () => {
    const token = localStorage.getItem('auth_token');
    
    if (!token) {
      setIsLoading(false);
      return;
    }

    try {
      console.log('🔍 Verificando suscripción con backend real...');
      
      const response = await fetch('http://localhost:8000/api/subscriptions/me', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Suscripción obtenida:', data);
        setSubscription(data.subscription);
        setUser(data.user);
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
      console.error('Error de conexión al verificar suscripción:', error);
      // En caso de error de conexión, limpiar sesión
      localStorage.removeItem('auth_token');
      setUser(null);
      setSubscription(null);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const hasFeature = useCallback((feature: string): boolean => {
    // El admin tiene acceso a todas las características
    if (user?.role === 'admin') {
      return true;
    }
    
    if (!subscription || subscription.status !== 'active') {
      return false;
    }

    const plan = planFeatures[subscription.planType];
    return plan?.features.includes(feature) || false;
  }, [subscription, planFeatures, user]);

  const canAccess = useCallback((section: string): boolean => {
    console.log(`🔍 canAccess llamado para sección: ${section}`);
    console.log(`🔍 Estado actual - User:`, user);
    console.log(`🔍 Estado actual - Subscription:`, subscription);
    
    // El admin tiene acceso a todas las secciones
    if (user?.role === 'admin') {
      console.log(`✅ canAccess: Admin tiene acceso a ${section}`);
      return true;
    }
    
    // Si no hay suscripción, solo permitir dashboard
    if (!subscription || subscription.status !== 'active') {
      console.log(`🚫 canAccess: No hay suscripción activa para ${section}`);
      return section === 'dashboard';
    }

    // Plan starter tiene acceso a todas las secciones básicas
    if (subscription.planType === 'starter') {
      const allowedSections = ['dashboard', 'trading', 'portfolio', 'analysis', 'brain-trader', 'subscriptions', 'help'];
      const hasAccess = allowedSections.includes(section);
      console.log(`🔍 canAccess: Plan starter - ${section} ${hasAccess ? 'permitido' : 'denegado'}`);
      return hasAccess;
    }

    const plan = planFeatures[subscription.planType];
    const hasAccess = plan?.sections.includes(section) || false;
    console.log(`🔍 canAccess: Plan ${subscription.planType} - ${section} ${hasAccess ? 'permitido' : 'denegado'}`);
    return hasAccess;
  }, [subscription, planFeatures, user]);

  useEffect(() => {
    const token = localStorage.getItem('auth_token');
    if (token) {
      checkSubscription();
    } else {
      setIsLoading(false);
    }
  }, [checkSubscription]);

  // Debug: Monitorear cambios en el estado
  useEffect(() => {
    console.log('🔍 AuthContext: Estado actualizado - User:', user);
    console.log('🔍 AuthContext: Estado actualizado - Subscription:', subscription);
    console.log('🔍 AuthContext: Estado actualizado - isLoading:', isLoading);
  }, [user, subscription, isLoading]);

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