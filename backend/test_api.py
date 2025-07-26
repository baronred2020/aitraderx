import requests
import json
import time

def test_api():
    base_url = "http://localhost:8080"
    
    print("🧪 Probando APIs de Brain Trader...")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Root endpoint
    print("\n2. Root Endpoint:")
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Test endpoint
    print("\n3. Test Endpoint:")
    try:
        response = requests.get(f"{base_url}/test")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Iniciando pruebas de API...")
    print("Asegúrate de que el servidor esté ejecutándose en http://localhost:8080")
    print("=" * 50)
    
    # Esperar un poco para que el servidor se inicie
    time.sleep(2)
    
    test_api() 