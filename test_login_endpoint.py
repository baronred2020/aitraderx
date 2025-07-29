#!/usr/bin/env python3
"""
Script para probar el endpoint de login
"""

import requests
import json

def test_login():
    url = "http://localhost:8000/api/auth/login"
    data = {
        "username": "user",
        "password": "user123"  # Cambiado a user123
    }
    
    try:
        print("🔍 Probando endpoint de login...")
        response = requests.post(url, json=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Login exitoso!")
            print(f"User: {result.get('user', {}).get('username')}")
            print(f"Subscription: {result.get('subscription', {}).get('planType') if result.get('subscription') else 'None'}")
            print(f"Token: {result.get('token', '')[:50]}...")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Error de conexión: No se puede conectar al backend")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_login() 