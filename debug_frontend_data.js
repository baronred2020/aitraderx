// Script de depuración para verificar datos del frontend
// Ejecutar en la consola del navegador

console.log('🔍 Iniciando depuración de datos del frontend...');

// Función para verificar la conexión con el backend
async function checkBackendConnection() {
    console.log('\n=== VERIFICANDO CONEXIÓN CON BACKEND ===');
    try {
        const response = await fetch('http://localhost:8000/api/market-data?symbols=EURUSD,GBPUSD');
        console.log('✅ Backend responde:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('📊 Datos del backend:', data);
            
            // Verificar estructura de datos
            Object.keys(data).forEach(symbol => {
                const symbolData = data[symbol];
                console.log(`${symbol}:`, {
                    price: symbolData.price,
                    change: symbolData.change,
                    changePercent: symbolData.changePercent,
                    marketStatus: symbolData.marketStatus
                });
            });
        } else {
            console.error('❌ Backend error:', response.status, response.statusText);
        }
    } catch (error) {
        console.error('❌ Error de conexión:', error);
    }
}

// Función para simular el hook useYahooMarketData
async function simulateYahooMarketData(symbols) {
    console.log('\n=== SIMULANDO HOOK useYahooMarketData ===');
    console.log('Símbolos solicitados:', symbols);
    
    try {
        const response = await fetch(`http://localhost:8000/api/market-data?symbols=${symbols.join(',')}`);
        
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Datos obtenidos:', data);
            
            // Simular el procesamiento del hook
            const processedData = {};
            symbols.forEach(symbol => {
                if (data[symbol]) {
                    processedData[symbol] = {
                        price: data[symbol].price,
                        change: data[symbol].change,
                        changePercent: data[symbol].changePercent,
                        volume: data[symbol].volume,
                        high: data[symbol].high,
                        low: data[symbol].low,
                        open: data[symbol].open,
                        previousClose: data[symbol].previousClose,
                        marketStatus: data[symbol].marketStatus
                    };
                }
            });
            
            console.log('📊 Datos procesados:', processedData);
            return processedData;
        } else {
            console.error('❌ Error en la respuesta:', response.status);
            return {};
        }
    } catch (error) {
        console.error('❌ Error en la simulación:', error);
        return {};
    }
}

// Función para verificar el estado del mercado
async function checkMarketStatus() {
    console.log('\n=== VERIFICANDO ESTADO DEL MERCADO ===');
    try {
        const response = await fetch('http://localhost:8000/api/market-status?symbol=EURUSD');
        if (response.ok) {
            const data = await response.json();
            console.log('📈 Estado del mercado:', data);
        } else {
            console.error('❌ Error obteniendo estado del mercado');
        }
    } catch (error) {
        console.error('❌ Error verificando estado del mercado:', error);
    }
}

// Función para verificar el componente TradingView
function checkTradingViewComponent() {
    console.log('\n=== VERIFICANDO COMPONENTE TRADINGVIEW ===');
    
    // Buscar el componente en el DOM
    const tradingView = document.querySelector('[data-testid="trading-view"]') || 
                       document.querySelector('.trading-card');
    
    if (tradingView) {
        console.log('✅ Componente TradingView encontrado');
        
        // Buscar el selector de símbolos
        const symbolSelect = tradingView.querySelector('select');
        if (symbolSelect) {
            console.log('✅ Selector de símbolos encontrado');
            console.log('Opciones disponibles:', symbolSelect.options.length);
            
            // Verificar las opciones
            Array.from(symbolSelect.options).forEach((option, index) => {
                console.log(`Opción ${index}:`, option.text);
            });
        } else {
            console.log('❌ Selector de símbolos no encontrado');
        }
    } else {
        console.log('❌ Componente TradingView no encontrado');
    }
}

// Función para verificar el estado de React
function checkReactState() {
    console.log('\n=== VERIFICANDO ESTADO DE REACT ===');
    
    // Intentar acceder al estado de React DevTools
    if (window.__REACT_DEVTOOLS_GLOBAL_HOOK__) {
        console.log('✅ React DevTools disponible');
    } else {
        console.log('⚠️ React DevTools no disponible');
    }
    
    // Verificar si hay errores en la consola
    console.log('📝 Revisar la consola para errores de React');
}

// Función principal de depuración
async function debugFrontendData() {
    console.log('🚀 Iniciando depuración completa...');
    
    // Verificar conexión con backend
    await checkBackendConnection();
    
    // Verificar estado del mercado
    await checkMarketStatus();
    
    // Simular hook de datos
    const testSymbols = ['EURUSD', 'GBPUSD', 'USDJPY'];
    await simulateYahooMarketData(testSymbols);
    
    // Verificar componente
    checkTradingViewComponent();
    
    // Verificar estado de React
    checkReactState();
    
    console.log('\n✅ Depuración completada');
    console.log('💡 Revisa los resultados arriba para identificar el problema');
}

// Ejecutar depuración
debugFrontendData().catch(error => {
    console.error('❌ Error en la depuración:', error);
});

// Función para verificar datos en tiempo real
function monitorDataUpdates() {
    console.log('\n=== MONITOREO DE ACTUALIZACIONES ===');
    
    // Crear un observador para detectar cambios en el DOM
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                const symbolElements = document.querySelectorAll('select option');
                symbolElements.forEach((option, index) => {
                    if (option.text.includes('Loading...') || option.text.includes('EUR/USD')) {
                        console.log(`🔄 Opción ${index} actualizada:`, option.text);
                    }
                });
            }
        });
    });
    
    // Observar cambios en el documento
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    console.log('👀 Monitoreando cambios en tiempo real...');
    console.log('💡 Los cambios se mostrarán automáticamente');
}

// Función para limpiar cache y forzar recarga
function forceDataReload() {
    console.log('\n=== FORZANDO RECARGA DE DATOS ===');
    
    // Limpiar cache del navegador para esta página
    if ('caches' in window) {
        caches.keys().then(names => {
            names.forEach(name => {
                caches.delete(name);
            });
        });
    }
    
    // Recargar la página
    console.log('🔄 Recargando página...');
    window.location.reload();
}

// Agregar funciones al objeto global para uso manual
window.debugFrontendData = debugFrontendData;
window.monitorDataUpdates = monitorDataUpdates;
window.forceDataReload = forceDataReload;
window.simulateYahooMarketData = simulateYahooMarketData;

console.log('🔧 Funciones de depuración disponibles:');
console.log('- debugFrontendData()');
console.log('- monitorDataUpdates()');
console.log('- forceDataReload()');
console.log('- simulateYahooMarketData([symbols])'); 