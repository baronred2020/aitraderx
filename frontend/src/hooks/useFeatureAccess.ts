import { useState, useCallback, useMemo } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { UpgradeModal } from '../components/Common/UpgradeModal';

interface FeatureAccessConfig {
  [key: string]: {
    requiredPlan: 'starter' | 'trader' | 'expert' | 'premium';
    feature: string;
  };
}

// Configuración de características por sección
const featureConfig: FeatureAccessConfig = {
  'dashboard': {
    requiredPlan: 'starter',
    feature: 'basic_dashboard'
  },
  'trading': {
    requiredPlan: 'starter',
    feature: 'basic_trading'
  },
  'portfolio': {
    requiredPlan: 'starter',
    feature: 'basic_portfolio'
  },
  'analysis': {
    requiredPlan: 'starter',
    feature: 'basic_analysis'
  },
  'ai-monitor': {
    requiredPlan: 'expert',
    feature: 'ai-monitor'
  },
  'rl': {
    requiredPlan: 'expert',
    feature: 'rl'
  },
  'reports': {
    requiredPlan: 'expert',
    feature: 'reports'
  },
  'alerts': {
    requiredPlan: 'trader',
    feature: 'alerts'
  },
  'mt4_integration': {
    requiredPlan: 'expert',
    feature: 'mt4_integration'
  },
  'api_access': {
    requiredPlan: 'premium',
    feature: 'api_access'
  },
  'custom_models': {
    requiredPlan: 'premium',
    feature: 'custom_models'
  },
  'brain-trader': {
    requiredPlan: 'starter',
    feature: 'brain_trader_basic'
  },
  'mega-mind': {
    requiredPlan: 'premium',
    feature: 'mega_mind_institutional'
  },
  'subscriptions': {
    requiredPlan: 'starter',
    feature: 'subscription_management'
  },
  'community': {
    requiredPlan: 'trader',
    feature: 'community_access'
  },
  'help': {
    requiredPlan: 'starter',
    feature: 'help_support'
  },
  'monitoring_agents': {
    requiredPlan: 'trader',
    feature: 'monitoring_agents'
  },
  'monitoring_alerts': {
    requiredPlan: 'trader',
    feature: 'monitoring_alerts'
  },
  'monitoring_config': {
    requiredPlan: 'expert',
    feature: 'monitoring_config'
  }
};

export const useFeatureAccess = () => {
  const { subscription, canAccess, hasFeature, user } = useAuth();
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [upgradeInfo, setUpgradeInfo] = useState<{
    currentPlan: string;
    requiredPlan: string;
    feature: string;
  } | null>(null);

  // Memoizar las funciones de verificación para evitar re-renders
  const checkAccess = useMemo(() => {
    return (section: string): boolean => {
      // El admin tiene acceso a todas las secciones
      if (user?.role === 'admin') {
        return true;
      }
      
      if (!subscription || subscription.status !== 'active') {
        return section === 'dashboard';
      }
      return canAccess(section);
    };
  }, [subscription, canAccess, user]);

  const checkFeature = useMemo(() => {
    return (feature: string): boolean => {
      // El admin tiene acceso a todas las características
      if (user?.role === 'admin') {
        return true;
      }
      
      if (!subscription || subscription.status !== 'active') {
        return false;
      }
      return hasFeature(feature);
    };
  }, [subscription, hasFeature, user]);

  const requireAccess = useCallback((section: string): boolean => {
    console.log(`🔍 useFeatureAccess: requireAccess llamado con section: ${section}`);
    
    const hasAccess = checkAccess(section);
    
    // Debug logging
    console.log(`🔍 Verificando acceso a ${section}:`, {
      hasAccess,
      userRole: user?.role,
      subscriptionStatus: subscription?.status,
      subscriptionPlan: subscription?.planType,
      canAccessResult: canAccess(section)
    });
    
    // No mostrar modal de upgrade al admin
    if (!hasAccess && user?.role !== 'admin') {
      console.log(`🚫 useFeatureAccess: Acceso denegado a ${section}, buscando configuración...`);
      const config = featureConfig[section];
      if (config) {
        console.log(`🚫 Acceso denegado a ${section}, mostrando modal de upgrade:`, config);
        setUpgradeInfo({
          currentPlan: subscription?.planType || 'starter',
          requiredPlan: config.requiredPlan,
          feature: config.feature
        });
        setShowUpgradeModal(true);
        console.log(`✅ useFeatureAccess: Modal configurado para ${section}`);
      } else {
        console.log(`❌ useFeatureAccess: No se encontró configuración para ${section}`);
      }
    } else {
      console.log(`✅ useFeatureAccess: Acceso permitido a ${section} o es admin`);
    }
    
    return hasAccess;
  }, [checkAccess, subscription, user, canAccess]);

  const requireFeature = useCallback((feature: string): boolean => {
    const hasFeatureAccess = checkFeature(feature);
    
    // No mostrar modal de upgrade al admin
    if (!hasFeatureAccess && user?.role !== 'admin') {
      // Encontrar la sección que requiere esta característica
      const section = Object.keys(featureConfig).find(
        key => featureConfig[key].feature === feature
      );
      
      if (section) {
        const config = featureConfig[section];
        setUpgradeInfo({
          currentPlan: subscription?.planType || 'freemium',
          requiredPlan: config.requiredPlan,
          feature: config.feature
        });
        setShowUpgradeModal(true);
      }
    }
    
    return hasFeatureAccess;
  }, [checkFeature, subscription, user]);

  const closeUpgradeModal = useCallback(() => {
    setShowUpgradeModal(false);
    setUpgradeInfo(null);
  }, []);

  return {
    checkAccess,
    checkFeature,
    requireAccess,
    requireFeature,
    subscription,
    showUpgradeModal,
    upgradeInfo,
    closeUpgradeModal
  };
}; 