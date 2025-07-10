import React, { useState, useEffect } from 'react';
import { Activity, Clock, AlertCircle, CheckCircle } from 'lucide-react';

interface APIMonitorProps {
  className?: string;
}

export const APIMonitor: React.FC<APIMonitorProps> = ({ className = '' }) => {
  const [dailyCalls, setDailyCalls] = useState(0);
  const [lastCallTime, setLastCallTime] = useState<Date | null>(null);
  const [timeUntilReset, setTimeUntilReset] = useState<string>('');

  useEffect(() => {
    const updateStats = () => {
      // Obtener estadísticas del localStorage o de un estado global
      const stored = localStorage.getItem('api_stats');
      if (stored) {
        const stats = JSON.parse(stored);
        setDailyCalls(stats.dailyCalls || 0);
        setLastCallTime(stats.lastCallTime ? new Date(stats.lastCallTime) : null);
      }
    };

    const updateTimeUntilReset = () => {
      const now = new Date();
      const tomorrow = new Date(now);
      tomorrow.setDate(tomorrow.getDate() + 1);
      tomorrow.setHours(0, 0, 0, 0);
      
      const timeLeft = tomorrow.getTime() - now.getTime();
      const hours = Math.floor(timeLeft / (1000 * 60 * 60));
      const minutes = Math.floor((timeLeft % (1000 * 60 * 60)) / (1000 * 60));
      
      setTimeUntilReset(`${hours}h ${minutes}m`);
    };

    updateStats();
    updateTimeUntilReset();

    const interval = setInterval(() => {
      updateStats();
      updateTimeUntilReset();
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const usagePercentage = (dailyCalls / 800) * 100;
  const isNearLimit = usagePercentage > 80;
  const isAtLimit = usagePercentage >= 100;

  return (
    <div className={`bg-gray-800/50 rounded-lg p-3 border border-gray-700/50 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-white flex items-center">
          <Activity className="w-4 h-4 mr-2" />
          API Usage Monitor
        </h3>
        <div className="flex items-center space-x-2">
          {isAtLimit ? (
            <AlertCircle className="w-4 h-4 text-red-400" />
          ) : isNearLimit ? (
            <AlertCircle className="w-4 h-4 text-yellow-400" />
          ) : (
            <CheckCircle className="w-4 h-4 text-green-400" />
          )}
        </div>
      </div>

      <div className="space-y-2">
        {/* Progress Bar */}
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div 
            className={`h-2 rounded-full transition-all duration-300 ${
              isAtLimit ? 'bg-red-500' : isNearLimit ? 'bg-yellow-500' : 'bg-green-500'
            }`}
            style={{ width: `${Math.min(usagePercentage, 100)}%` }}
          />
        </div>

        {/* Stats */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Calls Today:</span>
            <span className={`font-semibold ${
              isAtLimit ? 'text-red-400' : isNearLimit ? 'text-yellow-400' : 'text-green-400'
            }`}>
              {dailyCalls} / 800
            </span>
          </div>
          
          <div className="flex items-center justify-between">
            <span className="text-gray-400">Usage:</span>
            <span className="font-semibold text-white">
              {usagePercentage.toFixed(1)}%
            </span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-gray-400">Next Reset:</span>
            <span className="font-semibold text-white flex items-center">
              <Clock className="w-3 h-3 mr-1" />
              {timeUntilReset}
            </span>
          </div>

          <div className="flex items-center justify-between">
            <span className="text-gray-400">Last Call:</span>
            <span className="font-semibold text-white">
              {lastCallTime ? lastCallTime.toLocaleTimeString() : 'Never'}
            </span>
          </div>
        </div>

        {/* Status Message */}
        <div className="text-xs text-center mt-2">
          {isAtLimit ? (
            <span className="text-red-400">API limit reached. Using cached data.</span>
          ) : isNearLimit ? (
            <span className="text-yellow-400">Approaching API limit. Updates may be delayed.</span>
          ) : (
            <span className="text-green-400">API usage normal. Real-time updates active.</span>
          )}
        </div>
      </div>
    </div>
  );
}; 