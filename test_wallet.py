#!/usr/bin/env python3
"""
Script de prueba para verificar el funcionamiento de la wallet
"""

import requests
import json
from datetime import datetime

def test_wallet_endpoints():
    """Prueba los endpoints de la wallet"""
    print("🧪 Probando endpoints de wallet...")
    
    # URL base
    base_url = "http://localhost:8000"
    
    # Token de prueba (necesitarás un token válido)
    test_token = "test_token_here"  # Reemplazar con token válido
    
    headers = {
        'Authorization': f'Bearer {test_token}',
        'Content-Type': 'application/json',
    }
    
    # Probar endpoint de wallet
    try:
        print("\n📊 Probando GET /wallet...")
        response = requests.get(f"{base_url}/wallet", headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print("✅ Wallet data:")
            print(f"   Balance: {data.get('balance', 'N/A')}")
            print(f"   Transactions: {len(data.get('transactions', []))}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error probando wallet: {str(e)}")
    
    # Probar endpoint de transacciones
    try:
        print("\n📈 Probando GET /wallet/transactions...")
        response = requests.get(f"{base_url}/wallet/transactions", headers=headers, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Transacciones: {len(data)} encontradas")
            for i, tx in enumerate(data[:3]):  # Mostrar primeras 3
                print(f"   {i+1}. {tx.get('type', 'N/A')} - {tx.get('amount', 'N/A')} - {tx.get('description', 'N/A')}")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error probando transacciones: {str(e)}")

def test_wallet_without_auth():
    """Prueba la wallet sin autenticación para ver el error"""
    print("\n🔐 Probando wallet sin autenticación...")
    
    try:
        response = requests.get("http://localhost:8000/wallet", timeout=10)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 401:
            print("✅ Correcto: Endpoint requiere autenticación")
        else:
            print(f"❌ Inesperado: {response.text}")
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
        else:
            print(f"❌ Backend no responde: {response.text}")
    except Exception as e:
        print(f"❌ Error conectando al backend: {str(e)}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas de wallet...")
    print(f"⏰ Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        test_backend_health()
        test_wallet_without_auth()
        test_wallet_endpoints()
        
        print("\n✅ Todas las pruebas completadas")
        
    except Exception as e:
        print(f"\n❌ Error general: {str(e)}") 