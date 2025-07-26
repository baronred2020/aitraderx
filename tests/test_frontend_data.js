// Script de prueba para verificar que el frontend puede obtener datos del backend
// Ejecutar en el navegador en la consola de desarrollador

async function testFrontendData() {
    console.log('🚀 Iniciando pruebas del frontend...');
    
    // Probar datos de mercado
    console.log('\n=== PRUEBA DE DATOS DE MERCADO ===');
    try {
        const marketResponse = await fetch('http://localhost:8000/api/market-data?symbols=EURUSD,GBPUSD,AAPL');
        const marketData = await marketResponse.json();
        console.log('✅ Datos de mercado obtenidos:', marketData);
        
        // Verificar que cada símbolo tiene datos
        Object.keys(marketData).forEach(symbol => {
            const data = marketData[symbol];
            console.log(`${symbol}: Precio ${data.price}, Estado ${data.marketStatus}`);
        });
    } catch (error) {
        console.error('❌ Error obteniendo datos de mercado:', error);
    }
    
    // Probar datos de velas
    console.log('\n=== PRUEBA DE VELAS ===');
    try {
        const candlesResponse = await fetch('http://localhost:8000/api/candles?symbol=EURUSD&interval=15&count=50');
        const candlesData = await candlesResponse.json();
        console.log('✅ Datos de velas obtenidos:', {
            symbol: candlesData.symbol,
            interval: candlesData.interval,
            candleCount: candlesData.values.length
        });
        
        if (candlesData.values.length > 0) {
            console.log('Primera vela:', candlesData.values[0]);
            console.log('Última vela:', candlesData.values[candlesData.values.length - 1]);
        }
    } catch (error) {
        console.error('❌ Error obteniendo velas:', error);
    }
    
    // Probar estado del mercado
    console.log('\n=== PRUEBA DE ESTADO DEL MERCADO ===');
    try {
        const statusResponse = await fetch('http://localhost:8000/api/market-status?symbol=EURUSD');
        const statusData = await statusResponse.json();
        console.log('✅ Estado del mercado:', statusData);
    } catch (error) {
        console.error('❌ Error obteniendo estado del mercado:', error);
    }
    
    console.log('\n✅ Pruebas del frontend completadas');
}

// Función para probar los hooks del frontend
function testFrontendHooks() {
    console.log('\n=== PRUEBA DE HOOKS DEL FRONTEND ===');
    
    // Simular el hook useMarketData
    const mockMarketData = {
        EURUSD: {
            price: '1.0850',
            change: '0.0010',
            changePercent: '0.09',
            volume: '1000000',
            high: '1.0860',
            low: '1.0840',
            open: '1.0850',
            previousClose: '1.0840',
            marketStatus: 'closed'
        }
    };
    
    console.log('✅ Datos simulados del hook useMarketData:', mockMarketData);
    
    // Simular el hook useCandles
    const mockCandlesData = {
        symbol: 'EURUSD',
        interval: '15',
        values: [
            {
                datetime: '2025-07-11 22:00:00',
                open: '1.0850',
                high: '1.0860',
                low: '1.0840',
                close: '1.0855',
                volume: '1000000'
            },
            {
                datetime: '2025-07-11 22:15:00',
                open: '1.0855',
                high: '1.0865',
                low: '1.0845',
                close: '1.0860',
                volume: '1000000'
            }
        ]
    };
    
    console.log('✅ Datos simulados del hook useCandles:', mockCandlesData);
}

// Ejecutar pruebas
testFrontendData().then(() => {
    testFrontendHooks();
}).catch(error => {
    console.error('❌ Error en las pruebas:', error);
}); 