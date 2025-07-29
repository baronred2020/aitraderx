// Script para verificar el estado de autenticación
console.log('🔍 Verificando estado de autenticación...');

// Verificar token
const token = localStorage.getItem('auth_token');
console.log('Token:', token ? 'Presente' : 'No encontrado');

// Verificar si hay datos de usuario en el contexto
// Esto se ejecuta en la consola del navegador
console.log('Para verificar el estado completo, ejecuta en la consola del navegador:');
console.log('window.authDebug = () => {');
console.log('  const token = localStorage.getItem("auth_token");');
console.log('  console.log("Token:", token);');
console.log('  if (window.authContext) {');
console.log('    console.log("Usuario:", window.authContext.user);');
console.log('    console.log("Suscripción:", window.authContext.subscription);');
console.log('  }');
console.log('};');
console.log('window.authDebug();');