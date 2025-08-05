import React, { useState, useEffect } from 'react';
import { useAuth } from '../../contexts/AuthContext';
import { useAutomatedTrading } from '../../hooks/useAutomatedTrading';

interface MT4InstallerProps {
  onInstallationComplete: () => void;
}

interface InstallationStep {
  id: number;
  title: string;
  description: string;
  status: 'pending' | 'in-progress' | 'completed' | 'error';
  action?: () => void;
}

const MT4Installer: React.FC<MT4InstallerProps> = ({ onInstallationComplete }) => {
  const { user, subscription } = useAuth();
  const { downloadEA } = useAutomatedTrading();
  const [currentStep, setCurrentStep] = useState(1);
  const [steps, setSteps] = useState<InstallationStep[]>([
    {
      id: 1,
      title: 'Verificar Membresía Premium',
      description: 'Confirmando acceso a trading automático',
      status: 'pending'
    },
    {
      id: 2,
      title: 'Descargar Expert Advisor',
      description: 'Descargar el EA para MetaTrader 4/5',
      status: 'pending'
    },
    {
      id: 3,
      title: 'Instalar en MetaTrader',
      description: 'Instalar y configurar el EA en tu terminal',
      status: 'pending'
    },
    {
      id: 4,
      title: 'Activar Expert Advisor',
      description: 'Activar el EA en un gráfico de trading',
      status: 'pending'
    },
    {
      id: 5,
      title: 'Verificar Conexión',
      description: 'Confirmar que la conexión funciona correctamente',
      status: 'pending'
    }
  ]);

  const [downloadProgress, setDownloadProgress] = useState(0);
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected' | 'error'>('disconnected');

  useEffect(() => {
    // Verificar membresía premium al cargar
    if (subscription?.planType === 'premium' || subscription?.planType === 'institutional') {
      updateStepStatus(1, 'completed');
      setCurrentStep(2);
    } else {
      updateStepStatus(1, 'error');
    }
  }, [user]);

  const updateStepStatus = (stepId: number, status: InstallationStep['status']) => {
    setSteps(prev => prev.map(step => 
      step.id === stepId ? { ...step, status } : step
    ));
  };

  const handleDownloadEA = async () => {
    try {
      updateStepStatus(2, 'in-progress');
      setDownloadProgress(0);

      // Simular progreso de descarga
      const progressInterval = setInterval(() => {
        setDownloadProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      // Descargar el EA real
      await downloadEA();
      
      clearInterval(progressInterval);
      setDownloadProgress(100);
      
      setTimeout(() => {
        updateStepStatus(2, 'completed');
        setCurrentStep(3);
      }, 500);

    } catch (error) {
      console.error('Error descargando EA:', error);
      updateStepStatus(2, 'error');
    }
  };

  const verifyConnection = async () => {
    try {
      updateStepStatus(5, 'in-progress');
      setConnectionStatus('connecting');

      // Verificar conexión con el backend
      const response = await fetch('http://localhost:8000/api/v1/trading/mt4/status');
      const data = await response.json();

      if (data.connected) {
        setConnectionStatus('connected');
        updateStepStatus(5, 'completed');
        onInstallationComplete();
      } else {
        setConnectionStatus('error');
        updateStepStatus(5, 'error');
      }
    } catch (error) {
      console.error('Error verificando conexión:', error);
      setConnectionStatus('error');
      updateStepStatus(5, 'error');
    }
  };

  const getStepIcon = (status: InstallationStep['status']) => {
    switch (status) {
      case 'completed':
        return '✅';
      case 'in-progress':
        return '🔄';
      case 'error':
        return '❌';
      default:
        return '⭕';
    }
  };

  const getStepClass = (status: InstallationStep['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-50 border-green-200';
      case 'in-progress':
        return 'bg-blue-50 border-blue-200';
      case 'error':
        return 'bg-red-50 border-red-200';
      default:
        return 'bg-gray-50 border-gray-200';
    }
  };

  if (subscription?.planType !== 'premium' && subscription?.planType !== 'institutional') {
    return (
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="text-center">
          <div className="text-6xl mb-4">🔒</div>
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            Trading Automático - Solo Premium
          </h2>
          <p className="text-gray-600 mb-6">
            El trading automático está disponible exclusivamente para usuarios con membresía Premium o Pro.
          </p>
          <div className="bg-gradient-to-r from-purple-500 to-blue-500 text-white p-4 rounded-lg">
            <h3 className="font-semibold mb-2">Actualiza tu Plan</h3>
            <p className="text-sm opacity-90">
              Obtén acceso a trading automático, señales premium y más funcionalidades avanzadas.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="text-center mb-8">
        <div className="text-6xl mb-4">🤖</div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">
          Instalador de Trading Automático
        </h2>
        <p className="text-gray-600">
          Configura tu MetaTrader 4/5 para trading automático en 5 pasos simples
        </p>
      </div>

      {/* Pasos de instalación */}
      <div className="space-y-4 mb-8">
        {steps.map((step) => (
          <div
            key={step.id}
            className={`p-4 rounded-lg border-2 transition-all duration-300 ${getStepClass(step.status)} ${
              step.id === currentStep ? 'ring-2 ring-blue-500' : ''
            }`}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <span className="text-2xl">{getStepIcon(step.status)}</span>
                <div>
                  <h3 className="font-semibold text-gray-800">{step.title}</h3>
                  <p className="text-sm text-gray-600">{step.description}</p>
                </div>
              </div>
              
              {step.id === 2 && step.status === 'in-progress' && (
                <div className="flex items-center space-x-2">
                  <div className="w-24 bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${downloadProgress}%` }}
                    ></div>
                  </div>
                  <span className="text-sm text-gray-600">{downloadProgress}%</span>
                </div>
              )}

              {step.id === 5 && step.status === 'in-progress' && (
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                  <span className="text-sm text-gray-600">Verificando...</span>
                </div>
              )}
            </div>

            {/* Acciones específicas por paso */}
            {step.id === 2 && step.status === 'pending' && (
              <div className="mt-4">
                <button
                  onClick={handleDownloadEA}
                  className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  📥 Descargar Expert Advisor
                </button>
              </div>
            )}

            {step.id === 3 && step.status === 'pending' && (
              <div className="mt-4 p-4 bg-blue-50 rounded-lg">
                <h4 className="font-semibold text-blue-800 mb-2">Instrucciones de Instalación:</h4>
                <ol className="text-sm text-blue-700 space-y-1">
                  <li>1. Abre MetaTrader 4/5</li>
                  <li>2. Ve a <strong>Archivo → Abrir Carpeta de Datos</strong></li>
                  <li>3. Navega a <strong>MQL4 → Experts</strong></li>
                  <li>4. Copia el archivo <strong>AITRADERX_EA.mq4</strong> descargado</li>
                  <li>5. Reinicia MetaTrader</li>
                </ol>
                <button
                  onClick={() => {
                    updateStepStatus(3, 'completed');
                    setCurrentStep(4);
                  }}
                  className="mt-3 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  ✅ Confirmar Instalación
                </button>
              </div>
            )}

            {step.id === 4 && step.status === 'pending' && (
              <div className="mt-4 p-4 bg-yellow-50 rounded-lg">
                <h4 className="font-semibold text-yellow-800 mb-2">Activar Expert Advisor:</h4>
                <ol className="text-sm text-yellow-700 space-y-1">
                  <li>1. Abre cualquier gráfico en MetaTrader</li>
                  <li>2. Arrastra <strong>AITRADERX_EA</strong> desde el panel Navegador</li>
                  <li>3. Marca <strong>"Permitir trading automático"</strong></li>
                  <li>4. Marca <strong>"Permitir importación de DLL"</strong></li>
                  <li>5. Haz clic en <strong>OK</strong></li>
                </ol>
                <button
                  onClick={() => {
                    updateStepStatus(4, 'completed');
                    setCurrentStep(5);
                  }}
                  className="mt-3 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  ✅ Confirmar Activación
                </button>
              </div>
            )}

            {step.id === 5 && step.status === 'pending' && (
              <div className="mt-4">
                <button
                  onClick={verifyConnection}
                  className="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  🔗 Verificar Conexión
                </button>
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Estado de conexión */}
      {connectionStatus !== 'disconnected' && (
        <div className="mt-6 p-4 rounded-lg border">
          <h3 className="font-semibold text-gray-800 mb-2">Estado de Conexión:</h3>
          <div className="flex items-center space-x-2">
            {connectionStatus === 'connecting' && (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                <span className="text-blue-600">Conectando con MetaTrader...</span>
              </>
            )}
            {connectionStatus === 'connected' && (
              <>
                <span className="text-green-500">✅</span>
                <span className="text-green-600">Conectado exitosamente</span>
              </>
            )}
            {connectionStatus === 'error' && (
              <>
                <span className="text-red-500">❌</span>
                <span className="text-red-600">Error de conexión</span>
              </>
            )}
          </div>
        </div>
      )}

      {/* Información adicional */}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h3 className="font-semibold text-gray-800 mb-2">💡 Información Importante:</h3>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• El EA debe estar activo en al menos un gráfico para funcionar</li>
          <li>• Asegúrate de que MetaTrader tenga permisos de escritura</li>
          <li>• La conexión se mantiene mientras el EA esté activo</li>
          <li>• Puedes desconectar en cualquier momento desde el panel de control</li>
        </ul>
      </div>
    </div>
  );
};

export default MT4Installer; 