import { useState } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { UpgradeModal } from '../components/Common/UpgradeModal';

interface FeatureAccessConfig {
  [key: string]: {
    requiredPlan: 'freemium' | 'basic' | 'pro' | 'elite';
    feature: string;
  };
}

// Configuración de características por sección
const featureConfig: FeatureAccessConfig = {
  'ai-monitor': {
    requiredPlan: 'pro',
    feature: 'ai-monitor'
  },
  'rl': {
    requiredPlan: 'pro',
    feature: 'rl'
  },
  'reports': {
    requiredPlan: 'pro',
    feature: 'reports'
  },
  'alerts': {
    requiredPlan: 'basic',
    feature: 'alerts'
  },
  'mt4_integration': {
    requiredPlan: 'pro',
    feature: 'mt4_integration'
  },
  'api_access': {
    requiredPlan: 'elite',
    feature: 'api_access'
  },
  'custom_models': {
    requiredPlan: 'elite',
    feature: 'custom_models'
  }
};

export const useFeatureAccess = () => {
  const { subscription, canAccess, hasFeature } = useAuth();
  const [showUpgradeModal, setShowUpgradeModal] = useState(false);
  const [upgradeInfo, setUpgradeInfo] = useState<{
    currentPlan: string;
    requiredPlan: string;
    feature: string;
  } | null>(null);

  const checkAccess = (section: string): boolean => {
    if (!subscription || subscription.status !== 'active') {
      return section === 'dashboard';
    }

    return canAccess(section);
  };

  const checkFeature = (feature: string): boolean => {
    if (!subscription || subscription.status !== 'active') {
      return false;
    }

    return hasFeature(feature);
  };

  const requireAccess = (section: string): boolean => {
    const hasAccess = checkAccess(section);
    
    if (!hasAccess) {
      const config = featureConfig[section];
      if (config) {
        setUpgradeInfo({
          currentPlan: subscription?.planType || 'freemium',
          requiredPlan: config.requiredPlan,
          feature: config.feature
        });
        setShowUpgradeModal(true);
      }
    }
    
    return hasAccess;
  };

  const requireFeature = (feature: string): boolean => {
    const hasFeature = checkFeature(feature);
    
    if (!hasFeature) {
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
    
    return hasFeature;
  };

  const closeUpgradeModal = () => {
    setShowUpgradeModal(false);
    setUpgradeInfo(null);
  };

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