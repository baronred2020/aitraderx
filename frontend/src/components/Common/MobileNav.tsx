import React from 'react';
import { Brain, BarChart3, Zap, Settings, Home } from 'lucide-react';

interface MobileNavProps {
  activeTab: string;
  onTabChange: (tabId: string) => void;
}

export const MobileNav: React.FC<MobileNavProps> = ({ activeTab, onTabChange }) => {
  const tabs = [
    { id: 'dashboard', name: 'Dashboard', icon: Home },
    { id: 'rl', name: 'RL', icon: Brain },
    { id: 'analysis', name: 'Análisis', icon: BarChart3 },
    { id: 'alerts', name: 'Alertas', icon: Zap },
    { id: 'settings', name: 'Config', icon: Settings },
  ];

  return (
    <div className="md:hidden fixed bottom-0 left-0 right-0 bg-white border-t border-gray-200 z-50">
      <div className="flex justify-around">
        {tabs.map(tab => {
          const Icon = tab.icon;
          const isActive = activeTab === tab.id;
          
          return (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`flex flex-col items-center py-2 px-3 flex-1 ${
                isActive 
                  ? 'text-blue-600 bg-blue-50' 
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              <Icon className={`w-5 h-5 mb-1 ${isActive ? 'text-blue-600' : 'text-gray-500'}`} />
              <span className={`text-xs font-medium ${isActive ? 'text-blue-600' : 'text-gray-500'}`}>
                {tab.name}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}; 