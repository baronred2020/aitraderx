#!/usr/bin/env python3
"""
Script para probar login con usuario existente
"""

import requests
import json
from datetime import datetime

def test_login():
    """Prueba el login con usuario existente"""
    print("🧪 Probando login con usuario existente...")
    
    # URL base
    base_url = "http://localhost:8000"
    
    # Datos del usuario existente
    login_data = {
        "username": "demo_user",
        "password": "Demo123456"
    }
    
    try:
        print(f"\n🔐 Probando login con: {login_data['username']}")
        response = requests.post(f"{base_url}/api/auth/login", 
                               json=login_data, 
                               timeout=10)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Login exitoso:")
            print(f"   Usuario: {data.get('user', {}).get('username', 'N/A')}")
            print(f"   Email: {data.get('user', {}).get('email', 'N/A')}")
            print(f"   Token: {data.get('token', 'N/A')[:30]}...")
            
            # Probar wallet con token real
            print(f"\n💰 Probando wallet con token real...")
            headers = {
                'Authorization': f'Bearer {data.get("token")}',
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
                
                # Probar recarga
                print(f"\n💳 Probando recarga de saldo...")
                recharge_response = requests.post(f"{base_url}/wallet/recharge?amount=1000", 
                                                headers=headers, 
                                                timeout=10)
                
                print(f"Recharge Status: {recharge_response.status_code}")
                if recharge_response.status_code == 200:
                    recharge_data = recharge_response.json()
                    print("✅ Recarga exitosa:")
                    print(f"   Nuevo balance: {recharge_data.get('balance', 'N/A')}")
                else:
                    print(f"❌ Error en recarga: {recharge_response.text}")
                    
            else:
                print(f"❌ Error en wallet: {wallet_response.text}")
                
        else:
            print(f"❌ Error en login: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    print("🚀 Iniciando prueba de login...")
    print(f"⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_login()
        print("\n✅ Proceso completado")
        
    except Exception as e:
        print(f"\n❌ Error general: {str(e)}") 