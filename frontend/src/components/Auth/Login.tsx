import React, { useState } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { Brain, Eye, EyeOff, Loader2 } from 'lucide-react';

export const Login: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  
  const { login } = useAuth();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const success = await login(username, password);
      if (!success) {
        setError('Credenciales inválidas. Intenta con admin/admin123');
      }
    } catch (error) {
      setError('Error de conexión. Verifica que el backend esté corriendo.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo y título */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-teal-500 rounded-2xl flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-teal-400 bg-clip-text text-transparent mb-2">
            AITRADERX
          </h1>
          <p className="text-gray-400">AI Trading Platform</p>
        </div>

        {/* Formulario de login */}
        <div className="glass-effect rounded-2xl p-8 border border-gray-700/50">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">Iniciar Sesión</h2>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Usuario */}
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-gray-300 mb-2">
                Usuario
              </label>
              <input
                id="username"
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                placeholder="Ingresa tu usuario"
                required
              />
            </div>

            {/* Contraseña */}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
                Contraseña
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="w-full px-4 py-3 bg-gray-800/50 border border-gray-600/50 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all pr-12"
                  placeholder="Ingresa tu contraseña"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-white transition-colors"
                >
                  {showPassword ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
            </div>

            {/* Error */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Botón de login */}
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-blue-600 to-teal-600 hover:from-blue-700 hover:to-teal-700 disabled:opacity-50 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-all duration-200 flex items-center justify-center space-x-2"
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Iniciando sesión...</span>
                </>
              ) : (
                <span>Iniciar Sesión</span>
              )}
            </button>
          </form>

          {/* Información de desarrollo */}
          <div className="mt-6 p-4 bg-blue-500/10 border border-blue-500/20 rounded-lg">
            <h3 className="text-sm font-semibold text-blue-400 mb-2">Credenciales de Desarrollo</h3>
            <div className="text-xs text-gray-400 space-y-1">
              <p><strong>Usuario:</strong> admin</p>
              <p><strong>Contraseña:</strong> admin123</p>
              <p><strong>Plan:</strong> Elite (acceso completo)</p>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="text-center mt-8">
          <p className="text-gray-400 text-sm">
            ¿No tienes cuenta?{' '}
            <button 
              onClick={() => window.location.href = '/register'}
              className="text-blue-400 hover:text-blue-300 transition-colors"
            >
              Regístrate aquí
            </button>
          </p>
        </div>
      </div>
    </div>
  );
}; 