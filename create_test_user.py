#!/usr/bin/env python3
"""
Script para crear un usuario de prueba en el backend
"""

import requests
import json
from datetime import datetime

def create_test_user():
    """Crea un usuario de prueba en el backend"""
    print("🧪 Creando usuario de prueba...")
    
    # URL base
    base_url = "http://localhost:8000"
    
    # Datos del usuario de prueba
    user_data = {
        "username": "demo_user",
        "email": "demo@aitraderx.com",
        "password": "Demo123456",
        "firstName": "Demo",
        "lastName": "User"
    }
    
    try:
        print(f"\n📝 Creando usuario: {user_data['username']}")
        response = requests.post(f"{base_url}/api/auth/register", 
                               json=user_data, 
                               timeout=10)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Usuario creado exitosamente:")
            print(f"   Usuario: {data.get('user', {}).get('username', 'N/A')}")
            print(f"   Email: {data.get('user', {}).get('email', 'N/A')}")
            print(f"   Token: {data.get('token', 'N/A')[:20]}...")
            
            # Probar login
            print(f"\n🔐 Probando login...")
            login_data = {
                "username": user_data["username"],
                "password": user_data["password"]
            }
            
            login_response = requests.post(f"{base_url}/api/auth/login", 
                                         json=login_data, 
                                         timeout=10)
            
            if login_response.status_code == 200:
                login_data = login_response.json()
                print("✅ Login exitoso:")
                print(f"   Token: {login_data.get('token', 'N/A')[:20]}...")
                
                # Probar wallet con token real
                print(f"\n💰 Probando wallet con token real...")
                headers = {
                    'Authorization': f'Bearer {login_data.get("token")}',
                    'Content-Type': 'application/json',
                }
                
                wallet_response = requests.get(f"{base_url}/wallet", 
                                             headers=headers, 
                                             timeout=10)
                
                print(f"Wallet Status: {wallet_response.status_code}")
                if wallet_response.status_code == 200:
                    wallet_data = wallet_response.json()
                    print("✅ Wallet funcionando:")
                    print(f"   Balance: {wallet_data.get('balance', 'N/A')}")
                    print(f"   Transacciones: {len(wallet_data.get('transactions', []))}")
                else:
                    print(f"❌ Error en wallet: {wallet_response.text}")
                    
            else:
                print(f"❌ Error en login: {login_response.text}")
                
        else:
            print(f"❌ Error creando usuario: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def test_backend_health():
    """Prueba la salud del backend"""
    print("\n🏥 Probando salud del backend...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Backend saludable:")
            print(f"   Status: {data.get('status', 'N/A')}")
            print(f"   Timestamp: {data.get('timestamp', 'N/A')}")
            return True
        else:
            print(f"❌ Backend no responde: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error conectando al backend: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Iniciando creación de usuario de prueba...")
    print(f"⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        if test_backend_health():
            create_test_user()
        else:
            print("\n❌ Backend no está disponible. Asegúrate de que esté ejecutándose.")
        
        print("\n✅ Proceso completado")
        
    except Exception as e:
        print(f"\n❌ Error general: {str(e)}") 