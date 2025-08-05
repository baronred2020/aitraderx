import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';

interface AdminPermissions {
  hasFullAccess: boolean;
  bypassSubscriptionLimits: boolean;
  bypassUsageLimits: boolean;
  accessAllFeatures: boolean;
  accessAllSections: boolean;
}

interface AdminStatusData {
  user: {
    id: string;
    username: string;
    email: string;
    role: string;
    isAdmin: boolean;
  };
  subscription: any;
  adminPermissions: AdminPermissions;
}

const AdminStatus: React.FC = () => {
  const { user } = useAuth();
  const [adminStatus, setAdminStatus] = useState<AdminStatusData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkAdminStatus = async () => {
    if (!user || user.role !== 'admin') return;

    setLoading(true);
    setError(null);

    try {
      const token = localStorage.getItem('auth_token');
      if (!token) {
        setError('No hay token de autenticación');
        return;
      }

      const response = await fetch('http://localhost:8000/api/auth/admin/status', {
        headers: {
          'Authorization': `Bearer ${token}`,
        },
      });

      if (response.ok) {
        const data = await response.json();
        setAdminStatus(data);
      } else {
        setError('Error al obtener el estado del admin');
      }
    } catch (err) {
      setError('Error de conexión');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === 'admin') {
      checkAdminStatus();
    }
  }, [user]);

  if (!user || user.role !== 'admin') {
    return null;
  }

  return (
    <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold flex items-center">
          <span className="mr-2">👑</span>
          Panel de Administrador
        </h2>
        <button
          onClick={checkAdminStatus}
          disabled={loading}
          className="bg-white text-purple-600 px-4 py-2 rounded-lg font-semibold hover:bg-gray-100 disabled:opacity-50"
        >
          {loading ? 'Verificando...' : 'Actualizar Estado'}
        </button>
      </div>

      {error && (
        <div className="bg-red-500 text-white p-3 rounded-lg mb-4">
          ❌ {error}
        </div>
      )}

      {adminStatus && (
        <div className="space-y-4">
          {/* Información del Usuario */}
          <div className="bg-white/10 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">👤 Información del Usuario</h3>
            <div className="grid grid-cols-2 gap-2 text-sm">
              <div><strong>Usuario:</strong> {adminStatus.user.username}</div>
              <div><strong>Email:</strong> {adminStatus.user.email}</div>
              <div><strong>Rol:</strong> {adminStatus.user.role}</div>
              <div><strong>ID:</strong> {adminStatus.user.id}</div>
            </div>
          </div>

          {/* Permisos de Admin */}
          <div className="bg-white/10 p-4 rounded-lg">
            <h3 className="text-lg font-semibold mb-2">🔐 Permisos de Administrador</h3>
            <div className="grid grid-cols-1 gap-2 text-sm">
              {Object.entries(adminStatus.adminPermissions).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="capitalize">
                    {key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}:
                  </span>
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    value ? 'bg-green-500 text-white' : 'bg-red-500 text-white'
                  }`}>
                    {value ? '✅ Permitido' : '❌ Denegado'}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Suscripción */}
          {adminStatus.subscription ? (
            <div className="bg-white/10 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2">📋 Información de Suscripción</h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><strong>Plan:</strong> {adminStatus.subscription.planType}</div>
                <div><strong>Estado:</strong> {adminStatus.subscription.status}</div>
                <div><strong>Inicio:</strong> {new Date(adminStatus.subscription.startDate).toLocaleDateString()}</div>
                <div><strong>Fin:</strong> {new Date(adminStatus.subscription.endDate).toLocaleDateString()}</div>
              </div>
            </div>
          ) : (
            <div className="bg-green-500/20 border border-green-500/30 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-2 text-green-300">👑 Estado de Propietario</h3>
              <div className="text-sm text-green-100">
                <p className="mb-2">
                  <strong>No tienes suscripción</strong> porque eres el <strong>propietario del sistema</strong>.
                </p>
                <p>
                  Como administrador, tienes acceso ilimitado a todas las funciones sin necesidad de planes de suscripción.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {loading && (
        <div className="text-center py-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mx-auto"></div>
          <p className="mt-2">Verificando estado del administrador...</p>
        </div>
      )}
    </div>
  );
};

export default AdminStatus; 