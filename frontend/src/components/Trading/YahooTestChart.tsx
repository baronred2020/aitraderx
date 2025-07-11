import React, { useState, useEffect } from 'react';
import { useMarketData } from '../../hooks/useMarketData';
import { useCandles } from '../../hooks/useCandles';

const YahooTestChart: React.FC = () => {
  const [symbol, setSymbol] = useState('AAPL');
  const [timeframe, setTimeframe] = useState('15');
  const [lastSymbol, setLastSymbol] = useState(symbol);
  
  // Usar los hooks principales
  const { data: marketData, loading: marketLoading, error: marketError } = useMarketData([symbol]);
  const { data: candleData, loading: candleLoading, error: candleError } = useCandles(symbol, timeframe, 50);

  // Detectar cambio de símbolo
  useEffect(() => {
    if (symbol !== lastSymbol) {
      console.log(`[YahooTestChart] Symbol changed from ${lastSymbol} to ${symbol}`);
      setLastSymbol(symbol);
    }
  }, [symbol, lastSymbol]);

  // Determinar si está cargando
  const isLoading = marketLoading || candleLoading || symbol !== lastSymbol;

  const symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOGL', 'AMZN', 'EURUSD', 'GBPUSD', 'BTCUSD'];

  return (
    <div className="p-6 space-y-6">
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h2 className="text-xl font-bold text-white mb-4">Yahoo Finance Test</h2>
        
        {/* Selector de símbolo */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Símbolo:</label>
          <select 
            value={symbol} 
            onChange={(e) => setSymbol(e.target.value)}
            className="bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600"
            disabled={isLoading}
          >
            {symbols.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>

        {/* Selector de timeframe */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-300 mb-2">Timeframe:</label>
          <select 
            value={timeframe} 
            onChange={(e) => setTimeframe(e.target.value)}
            className="bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600"
            disabled={isLoading}
          >
            <option value="1">1 min</option>
            <option value="5">5 min</option>
            <option value="15">15 min</option>
            <option value="30">30 min</option>
            <option value="60">1 hora</option>
            <option value="D">1 día</option>
          </select>
        </div>

        {/* Estado de carga */}
        <div className="mb-4 p-3 bg-gray-700/50 rounded-lg">
          <div className="flex items-center space-x-2 mb-2">
            <div className={`w-3 h-3 rounded-full ${isLoading ? 'bg-blue-400 animate-pulse' : 'bg-green-400'}`}></div>
            <span className="text-sm font-medium text-white">
              {isLoading ? '🔄 Cargando...' : '✅ Listo'}
            </span>
          </div>
          <div className="text-sm text-gray-400 space-y-1">
            <div>Market Data: {marketLoading ? '🔄 Cargando...' : '✅ Listo'}</div>
            <div>Candles: {candleLoading ? '🔄 Cargando...' : '✅ Listo'}</div>
            {symbol !== lastSymbol && (
              <div className="text-blue-400">🔄 Cambiando a {symbol}...</div>
            )}
          </div>
        </div>

        {/* Errores */}
        {(marketError || candleError) && (
          <div className="mb-4 p-3 bg-red-500/20 border border-red-500/50 rounded-lg">
            <div className="text-red-400 text-sm">
              {marketError && <div>Market Error: {marketError}</div>}
              {candleError && <div>Candle Error: {candleError}</div>}
            </div>
          </div>
        )}

        {/* Datos de mercado */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-white mb-2">Datos de Mercado</h3>
          {isLoading ? (
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-600 rounded mb-2"></div>
                <div className="h-4 bg-gray-600 rounded mb-2"></div>
                <div className="h-4 bg-gray-600 rounded"></div>
              </div>
            </div>
          ) : marketData[symbol] ? (
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div><span className="text-gray-400">Precio:</span> <span className="text-green-400">{marketData[symbol].price}</span></div>
                <div><span className="text-gray-400">Cambio:</span> <span className="text-blue-400">{marketData[symbol].change}</span></div>
                <div><span className="text-gray-400">% Cambio:</span> <span className="text-purple-400">{marketData[symbol].changePercent}%</span></div>
                <div><span className="text-gray-400">Volumen:</span> <span className="text-yellow-400">{marketData[symbol].volume}</span></div>
                <div><span className="text-gray-400">Máximo:</span> <span className="text-green-400">{marketData[symbol].high}</span></div>
                <div><span className="text-gray-400">Mínimo:</span> <span className="text-red-400">{marketData[symbol].low}</span></div>
              </div>
            </div>
          ) : (
            <div className="text-gray-400">No hay datos disponibles</div>
          )}
        </div>

        {/* Datos de velas */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-white mb-2">Datos de Velas</h3>
          {isLoading ? (
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <div className="animate-pulse">
                <div className="h-4 bg-gray-600 rounded mb-2"></div>
                <div className="h-4 bg-gray-600 rounded mb-2"></div>
                <div className="h-4 bg-gray-600 rounded mb-2"></div>
                <div className="h-4 bg-gray-600 rounded"></div>
              </div>
            </div>
          ) : candleData && candleData.values ? (
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <div className="text-sm text-gray-400 mb-2">
                Total de velas: {candleData.values.length}
              </div>
              <div className="max-h-40 overflow-y-auto">
                {candleData.values.slice(0, 5).map((candle, index) => (
                  <div key={index} className="text-xs text-gray-300 mb-1">
                    {candle.datetime} - O:{candle.open} H:{candle.high} L:{candle.low} C:{candle.close} V:{candle.volume}
                  </div>
                ))}
                {candleData.values.length > 5 && (
                  <div className="text-xs text-gray-500">... y {candleData.values.length - 5} más</div>
                )}
              </div>
            </div>
          ) : (
            <div className="text-gray-400">No hay datos de velas disponibles</div>
          )}
        </div>

        {/* Información de debug */}
        <div className="text-xs text-gray-500">
          <div>Última actualización: {new Date().toLocaleTimeString()}</div>
          <div>Símbolo actual: {symbol}</div>
          <div>Timeframe actual: {timeframe}</div>
          <div>Estado: {isLoading ? 'Cargando' : 'Listo'}</div>
        </div>
      </div>
    </div>
  );
};

export default YahooTestChart; 