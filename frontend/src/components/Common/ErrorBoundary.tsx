import React, { Component, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: React.ErrorInfo;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return {
      hasError: true,
      error
    };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('Error boundary caught an error:', error, errorInfo);
    this.setState({
      error,
      errorInfo
    });
  }

  handleRefresh = () => {
    window.location.reload();
  };

  handleReset = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-red-900 to-gray-900 flex items-center justify-center p-4">
          <div className="text-center max-w-md w-full">
            <div className="flex items-center justify-center mb-8">
              <div className="w-24 h-24 bg-gradient-to-r from-red-500 to-orange-500 rounded-3xl flex items-center justify-center shadow-2xl">
                <AlertTriangle className="w-12 h-12 text-white" />
              </div>
            </div>
            
            <h1 className="text-4xl font-bold text-white mb-4">
              ¡Ups! Algo salió mal
            </h1>
            
            <p className="text-gray-400 text-lg mb-8">
              La aplicación encontró un error inesperado. No te preocupes, estamos trabajando para solucionarlo.
            </p>

            {process.env.NODE_ENV === 'development' && this.state.error && (
              <div className="bg-gray-800/50 border border-gray-700 rounded-lg p-4 mb-6 text-left">
                <h3 className="text-red-400 font-semibold mb-2">Error Details:</h3>
                <p className="text-sm text-gray-300 font-mono">
                  {this.state.error.message}
                </p>
                {this.state.errorInfo && (
                  <details className="mt-2">
                    <summary className="text-sm text-gray-400 cursor-pointer">
                      Stack Trace
                    </summary>
                    <pre className="text-xs text-gray-500 mt-2 overflow-auto">
                      {this.state.errorInfo.componentStack}
                    </pre>
                  </details>
                )}
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button
                onClick={this.handleReset}
                className="px-6 py-3 bg-blue-500 hover:bg-blue-600 text-white font-medium rounded-lg transition-colors"
              >
                Intentar de nuevo
              </button>
              
              <button
                onClick={this.handleRefresh}
                className="px-6 py-3 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors flex items-center justify-center"
              >
                <RefreshCw className="w-4 h-4 mr-2" />
                Recargar página
              </button>
            </div>

            <p className="text-gray-500 text-sm mt-6">
              Si el problema persiste, contacta con soporte técnico.
            </p>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
} 