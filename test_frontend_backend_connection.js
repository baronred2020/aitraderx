// Test script to verify frontend-backend connection
const fetch = require('node-fetch');

const API_BASE_URL = 'http://localhost:8080/api/v1';

async function testApiConnection() {
  console.log('🧪 Probando conexión Frontend-Backend...');
  console.log('=' .repeat(50));

  try {
    // Test 1: Health check
    console.log('1. Probando Health Check...');
    const healthResponse = await fetch(`http://localhost:8080/health`);
    const healthData = await healthResponse.json();
    console.log('✅ Health Check:', healthData.status);
    console.log('');

    // Test 2: Available brains
    console.log('2. Probando Available Brains...');
    const brainsResponse = await fetch(`${API_BASE_URL}/brain-trader/available-brains`);
    const brainsData = await brainsResponse.json();
    console.log('✅ Available Brains:', brainsData.available_brains);
    console.log('');

    // Test 3: Brain Max predictions
    console.log('3. Probando Brain Max Predictions...');
    const predictionsResponse = await fetch(`${API_BASE_URL}/brain-trader/predictions/brain_max?pair=EURUSD&limit=2`);
    const predictionsData = await predictionsResponse.json();
    console.log('✅ Brain Max Predictions:', predictionsData.length, 'predictions');
    console.log('   Sample:', predictionsData[0]);
    console.log('');

    // Test 4: Mega Mind predictions
    console.log('4. Probando Mega Mind Predictions...');
    const megaMindResponse = await fetch(`${API_BASE_URL}/mega-mind/predictions?pair=EURUSD&limit=2`);
    const megaMindData = await megaMindResponse.json();
    console.log('✅ Mega Mind Predictions:', megaMindData.length, 'predictions');
    console.log('   Sample:', megaMindData[0]);
    console.log('');

    // Test 5: Mega Mind collaboration
    console.log('5. Probando Mega Mind Collaboration...');
    const collaborationResponse = await fetch(`${API_BASE_URL}/mega-mind/collaboration?pair=EURUSD`);
    const collaborationData = await collaborationResponse.json();
    console.log('✅ Mega Mind Collaboration:', collaborationData.collaboration_score);
    console.log('');

    console.log('🎉 ¡Todas las pruebas pasaron! El backend está funcionando correctamente.');
    console.log('📱 Ahora puedes abrir http://localhost:3000 en tu navegador para probar el frontend.');

  } catch (error) {
    console.error('❌ Error en las pruebas:', error.message);
    console.log('');
    console.log('🔧 Solución de problemas:');
    console.log('1. Asegúrate de que el servidor backend esté ejecutándose en puerto 8080');
    console.log('2. Verifica que no haya errores en la consola del backend');
    console.log('3. Comprueba que el firewall no esté bloqueando las conexiones');
  }
}

// Run the test
testApiConnection(); 