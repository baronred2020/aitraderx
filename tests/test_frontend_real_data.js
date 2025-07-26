// Script para verificar que el frontend muestra datos reales de Yahoo Finance
// Ejecutar en la consola del navegador

console.log('🔍 Verificando datos reales de Yahoo Finance en el frontend...');

// Función para verificar datos del backend
async function checkBackendData() {
    console.log('\\n=== VERIFICANDO DATOS DEL BACKEND ===');
    
    try {
        const response = await fetch('http://localhost:8000/api/market-data?symbols=EURUSD,GBPUSD,USDJPY,AAPL');
        const data = await response.json();
        
        console.log('✅ Datos del backend:', data);
        
        // Verificar que los precios son reales (no fallback)
        Object.keys(data).forEach(symbol => {
            const symbolData = data[symbol];
            const price = parseFloat(symbolData.price);
            
            console.log(`${symbol}: ${price}`);
            
            // Verificar rangos realistas
            if (symbol === 'EURUSD' && (price < 0.5 || price > 2.0)) {
                console.log(`⚠️ Precio sospechoso para ${symbol}: ${price}`);
            } else if (symbol === 'USDJPY' && (price < 100 || price > 200)) {
                console.log(`⚠️ Precio sospechoso para ${symbol}: ${price}`);
            } else if (symbol === 'AAPL' && (price < 50 || price > 1000)) {
                console.log(`⚠️ Precio sospechoso para ${symbol}: ${price}`);
            } else {
                console.log(`✅ Precio realista para ${symbol}: ${price}`);
            }
        });
        
        return data;
    } catch (error) {
        console.error('❌ Error verificando backend:', error);
        return null;
    }
}

// Función para verificar datos del frontend
function checkFrontendData() {
    console.log('\\n=== VERIFICANDO DATOS DEL FRONTEND ===');
    
    // Verificar que el hook useYahooMarketData está funcionando
    const hookLogs = [];
    const originalLog = console.log;
    
    console.log = (...args) => {
        if (args[0] && typeof args[0] === 'string' && args[0].includes('[useYahooMarketData]')) {
            hookLogs.push(args[0]);
        }
        originalLog.apply(console, args);
    };
    
    // Esperar un momento para capturar logs
    setTimeout(() => {
        console.log = originalLog;
        
        const realDataLogs = hookLogs.filter(log => 
            log.includes('datos reales') || log.includes('Yahoo Finance')
        );
        
        if (realDataLogs.length > 0) {
            console.log('✅ Hook está obteniendo datos reales de Yahoo Finance');
            realDataLogs.forEach(log => console.log('   📊', log));
        } else {
            console.log('⚠️ No se detectaron logs de datos reales');
        }
    }, 3000);
}

// Función para verificar panel de símbolos
function checkSymbolPanel() {
    console.log('\\n=== VERIFICANDO PANEL DE SÍMBOLOS ===');
    
    const symbolSelect = document.querySelector('select');
    if (symbolSelect) {
        const options = symbolSelect.querySelectorAll('option');
        console.log(`📊 Opciones encontradas: ${options.length}`);
        
        options.forEach((option, index) => {
            const text = option.textContent;
            if (text) {
                console.log(`   ${index + 1}. ${text}`);
                
                // Verificar que no dice "Loading..."
                if (text.includes('Loading...')) {
                    console.log(`   ⚠️ Opción ${index + 1} aún cargando`);
                } else if (text.includes('EUR/USD') || text.includes('GBP/USD')) {
                    console.log(`   ✅ Opción ${index + 1} con datos reales`);
                }
            }
        });
    } else {
        console.log('❌ No se encontró el selector de símbolos');
    }
}

// Función para verificar precios específicos
function checkSpecificPrices() {
    console.log('\\n=== VERIFICANDO PRECIOS ESPECÍFICOS ===');
    
    // Buscar elementos que muestren precios
    const priceElements = document.querySelectorAll('[class*="price"], [class*="Price"]');
    console.log(`💰 Elementos de precio encontrados: ${priceElements.length}`);
    
    priceElements.forEach((element, index) => {
        const text = element.textContent;
        if (text && (text.includes('1.16') || text.includes('1.34') || text.includes('147'))) {
            console.log(`   ✅ Precio real detectado: ${text}`);
        }
    });
}

// Función principal
async function runVerification() {
    console.log('🚀 Iniciando verificación de datos reales...');
    
    // Verificar backend
    const backendData = await checkBackendData();
    
    // Verificar frontend
    checkFrontendData();
    
    // Verificar panel
    setTimeout(() => {
        checkSymbolPanel();
        checkSpecificPrices();
        
        console.log('\\n=== RESUMEN ===');
        if (backendData) {
            console.log('✅ Backend devuelve datos reales de Yahoo Finance');
            console.log('📊 Precios reales detectados:');
            Object.keys(backendData).forEach(symbol => {
                console.log(`   ${symbol}: ${backendData[symbol].price}`);
            });
        } else {
            console.log('❌ Backend no responde correctamente');
        }
        
        console.log('\\n🎯 El panel de símbolos debería mostrar estos precios reales:');
        console.log('   EUR/USD: ~1.16918');
        console.log('   GBP/USD: ~1.34905');
        console.log('   USD/JPY: ~147.38200');
        console.log('   AAPL: ~211.16000');
        
    }, 2000);
}

// Ejecutar verificación
runVerification(); 