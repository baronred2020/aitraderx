import React, { useState, useEffect } from 'react';
import { 
  CheckCircle, 
  AlertTriangle, 
  XCircle, 
  Info, 
  X,
  TrendingUp,
  TrendingDown,
  Bell,
  Settings
} from 'lucide-react';

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info' | 'trade';
  title: string;
  message: string;
  timestamp: Date;
  read?: boolean;
  action?: {
    label: string;
    onClick: () => void;
  };
}

interface NotificationProps {
  notification: Notification;
  onClose: (id: string) => void;
  onRead: (id: string) => void;
}

const NotificationItem: React.FC<NotificationProps> = ({ notification, onClose, onRead }) => {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  const getIcon = () => {
    switch (notification.type) {
      case 'success':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'error':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="w-5 h-5 text-yellow-400" />;
      case 'info':
        return <Info className="w-5 h-5 text-blue-400" />;
      case 'trade':
        return notification.message.includes('BUY') ? 
          <TrendingUp className="w-5 h-5 text-green-400" /> : 
          <TrendingDown className="w-5 h-5 text-red-400" />;
      default:
        return <Bell className="w-5 h-5 text-gray-400" />;
    }
  };

  const getBorderColor = () => {
    switch (notification.type) {
      case 'success':
        return 'border-green-500/30';
      case 'error':
        return 'border-red-500/30';
      case 'warning':
        return 'border-yellow-500/30';
      case 'info':
        return 'border-blue-500/30';
      case 'trade':
        return notification.message.includes('BUY') ? 
          'border-green-500/30' : 'border-red-500/30';
      default:
        return 'border-gray-500/30';
    }
  };

  const getBgColor = () => {
    switch (notification.type) {
      case 'success':
        return 'bg-green-500/10';
      case 'error':
        return 'bg-red-500/10';
      case 'warning':
        return 'bg-yellow-500/10';
      case 'info':
        return 'bg-blue-500/10';
      case 'trade':
        return notification.message.includes('BUY') ? 
          'bg-green-500/10' : 'bg-red-500/10';
      default:
        return 'bg-gray-500/10';
    }
  };

  const formatTime = (date: Date) => {
    const now = new Date();
    const diff = now.getTime() - date.getTime();
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);

    if (minutes < 1) return 'Ahora';
    if (minutes < 60) return `${minutes}m`;
    if (hours < 24) return `${hours}h`;
    return `${days}d`;
  };

  return (
    <div
      className={`notification ${getBgColor()} ${getBorderColor()} p-4 rounded-lg border-l-4 transition-all duration-300 ${
        isVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full'
      }`}
      onClick={() => onRead(notification.id)}
    >
      <div className="flex items-start space-x-3">
        <div className="flex-shrink-0 mt-0.5">
          {getIcon()}
        </div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <p className="text-sm font-semibold text-white">
              {notification.title}
            </p>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-400">
                {formatTime(notification.timestamp)}
              </span>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onClose(notification.id);
                }}
                className="text-gray-400 hover:text-white transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>
          <p className="text-sm text-gray-300 mt-1">
            {notification.message}
          </p>
          {notification.action && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                notification.action!.onClick();
              }}
              className="mt-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              {notification.action.label}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

interface NotificationCenterProps {
  notifications: Notification[];
  onClose: (id: string) => void;
  onRead: (id: string) => void;
  onClearAll: () => void;
  onMarkAllRead: () => void;
}

export const NotificationCenter: React.FC<NotificationCenterProps> = ({
  notifications,
  onClose,
  onRead,
  onClearAll,
  onMarkAllRead
}) => {
  const unreadCount = notifications.filter(n => !n.read).length;

  const groupedNotifications = {
    unread: notifications.filter(n => !n.read),
    read: notifications.filter(n => n.read)
  };

  return (
    <div className="trading-card p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <div className="relative">
            <Bell className="w-6 h-6 text-white" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-xs rounded-full flex items-center justify-center text-white font-semibold">
                {unreadCount}
              </span>
            )}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">Notificaciones</h3>
            <p className="text-sm text-gray-400">
              {unreadCount} sin leer • {notifications.length} total
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={onMarkAllRead}
            className="text-sm text-blue-400 hover:text-blue-300 transition-colors"
          >
            Marcar como leídas
          </button>
          <button
            onClick={onClearAll}
            className="text-sm text-red-400 hover:text-red-300 transition-colors"
          >
            Limpiar todas
          </button>
        </div>
      </div>

      <div className="space-y-4">
        {/* Notificaciones sin leer */}
        {groupedNotifications.unread.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-white mb-3">Sin leer</h4>
            <div className="space-y-3">
              {groupedNotifications.unread.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onClose={onClose}
                  onRead={onRead}
                />
              ))}
            </div>
          </div>
        )}

        {/* Notificaciones leídas */}
        {groupedNotifications.read.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-400 mb-3">Leídas</h4>
            <div className="space-y-3">
              {groupedNotifications.read.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onClose={onClose}
                  onRead={onRead}
                />
              ))}
            </div>
          </div>
        )}

        {/* Estado vacío */}
        {notifications.length === 0 && (
          <div className="text-center py-8">
            <Bell className="w-12 h-12 text-gray-500 mx-auto mb-4" />
            <p className="text-gray-400">No hay notificaciones</p>
            <p className="text-sm text-gray-500 mt-1">
              Las notificaciones aparecerán aquí cuando haya actividad
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

// Hook para gestionar notificaciones
export const useNotifications = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const addNotification = (notification: Omit<Notification, 'id' | 'timestamp'>) => {
    const newNotification: Notification = {
      ...notification,
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date(),
      read: false
    };
    setNotifications(prev => [newNotification, ...prev]);
  };

  const removeNotification = (id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  };

  const markAsRead = (id: string) => {
    setNotifications(prev => 
      prev.map(n => n.id === id ? { ...n, read: true } : n)
    );
  };

  const clearAll = () => {
    setNotifications([]);
  };

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  return {
    notifications,
    addNotification,
    removeNotification,
    markAsRead,
    clearAll,
    markAllAsRead
  };
}; 