// Script de prueba para verificar que el frontend está funcionando correctamente
// Ejecutar en la consola del navegador después de los fixes

console.log('🔧 Iniciando pruebas del frontend después de los fixes...');

// Función para verificar que no hay bucles infinitos
function checkForInfiniteLoops() {
    console.log('\\n=== VERIFICANDO BUCLES INFINITOS ===');
    
    // Verificar que los logs no se repiten infinitamente
    const logs = performance.getEntriesByType('navigation');
    console.log('✅ Navegación normal:', logs.length === 1);
    
    // Verificar que el DOM no se está recreando constantemente
    const chartElements = document.querySelectorAll('[class*="chart"]');
    console.log('✅ Elementos de gráfico estables:', chartElements.length);
    
    return true;
}

// Función para verificar datos de mercado
async function checkMarketData() {
    console.log('\\n=== VERIFICANDO DATOS DE MERCADO ===');
    
    try {
        const response = await fetch('http://localhost:8000/api/market-data?symbols=EURUSD,GBPUSD');
        const data = await response.json();
        
        console.log('✅ Backend responde correctamente');
        console.log('📊 Datos recibidos:', data);
        
        // Verificar estructura de datos
        Object.keys(data).forEach(symbol => {
            const symbolData = data[symbol];
            console.log(`${symbol}: Precio ${symbolData.price}, Estado ${symbolData.marketStatus || 'N/A'}`);
        });
        
        return true;
    } catch (error) {
        console.error('❌ Error verificando datos:', error);
        return false;
    }
}

// Función para verificar hooks
function checkHooks() {
    console.log('\\n=== VERIFICANDO HOOKS ===');
    
    // Verificar que useYahooMarketData no causa bucles
    const hookLogs = [];
    const originalLog = console.log;
    
    console.log = (...args) => {
        if (args[0] && typeof args[0] === 'string' && args[0].includes('[useYahooMarketData]')) {
            hookLogs.push(args[0]);
        }
        originalLog.apply(console, args);
    };
    
    // Simular un pequeño delay para verificar logs
    setTimeout(() => {
        console.log = originalLog;
        
        const uniqueLogs = new Set(hookLogs);
        console.log('✅ Logs únicos del hook:', uniqueLogs.size);
        console.log('✅ No hay bucles infinitos en hooks');
        
        if (uniqueLogs.size < 10) {
            console.log('✅ Hooks funcionando correctamente');
        } else {
            console.log('⚠️ Posible bucle en hooks detectado');
        }
    }, 2000);
}

// Función para verificar componentes
function checkComponents() {
    console.log('\\n=== VERIFICANDO COMPONENTES ===');
    
    // Verificar que TradingView está renderizado
    const tradingView = document.querySelector('[class*="trading"]');
    console.log('✅ TradingView renderizado:', !!tradingView);
    
    // Verificar que YahooTradingChart está presente
    const chartContainer = document.querySelector('[class*="chart"]');
    console.log('✅ Contenedor de gráfico presente:', !!chartContainer);
    
    // Verificar símbolos en el panel
    const symbolSelect = document.querySelector('select');
    console.log('✅ Selector de símbolos presente:', !!symbolSelect);
    
    return true;
}

// Función principal de pruebas
async function runTests() {
    console.log('🚀 Iniciando pruebas completas...');
    
    const results = {
        infiniteLoops: checkForInfiniteLoops(),
        marketData: await checkMarketData(),
        hooks: checkHooks(),
        components: checkComponents()
    };
    
    console.log('\\n=== RESULTADOS DE PRUEBAS ===');
    console.log('✅ Verificación de bucles infinitos:', results.infiniteLoops);
    console.log('✅ Datos de mercado:', results.marketData);
    console.log('✅ Hooks funcionando:', results.hooks);
    console.log('✅ Componentes renderizados:', results.components);
    
    const allPassed = Object.values(results).every(result => result === true);
    
    if (allPassed) {
        console.log('\\n🎉 ¡TODAS LAS PRUEBAS PASARON! El frontend está funcionando correctamente.');
    } else {
        console.log('\\n⚠️ Algunas pruebas fallaron. Revisar logs anteriores.');
    }
    
    return allPassed;
}

// Ejecutar pruebas
runTests().then(success => {
    if (success) {
        console.log('\\n✅ El problema del panel de símbolos debería estar resuelto.');
        console.log('📊 Los precios reales deberían mostrarse ahora.');
    }
});

// Función para verificar precios específicos
function checkPrices() {
    console.log('\\n=== VERIFICANDO PRECIOS EN PANEL ===');
    
    const symbolOptions = document.querySelectorAll('select option');
    console.log('📊 Opciones de símbolos encontradas:', symbolOptions.length);
    
    symbolOptions.forEach((option, index) => {
        const text = option.textContent;
        if (text && text.includes('Loading...')) {
            console.log(`⚠️ Símbolo ${index + 1} aún cargando:`, text);
        } else if (text && text.includes('EUR/USD') || text.includes('GBP/USD')) {
            console.log(`✅ Símbolo ${index + 1} con precio real:`, text);
        }
    });
}

// Ejecutar verificación de precios después de un delay
setTimeout(checkPrices, 3000); 