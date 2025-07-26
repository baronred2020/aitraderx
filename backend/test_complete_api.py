import requests
import json
import time

def test_complete_api():
    base_url = "http://localhost:8080"
    
    print("🧪 Probando APIs completas de Brain Trader y Mega Mind...")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 2: Available brains
    print("\n2. Available Brains:")
    try:
        response = requests.get(f"{base_url}/api/v1/brain-trader/available-brains")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 3: Brain Max Predictions
    print("\n3. Brain Max Predictions:")
    try:
        response = requests.get(f"{base_url}/api/v1/brain-trader/predictions/brain_max", 
                              params={"pair": "EURUSD", "style": "day_trading", "limit": 3})
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 4: Brain Ultra Signals
    print("\n4. Brain Ultra Signals:")
    try:
        response = requests.get(f"{base_url}/api/v1/brain-trader/signals/brain_ultra", 
                              params={"pair": "EURUSD", "limit": 2})
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 5: Brain Predictor Trends
    print("\n5. Brain Predictor Trends:")
    try:
        response = requests.get(f"{base_url}/api/v1/brain-trader/trends/brain_predictor", 
                              params={"pair": "EURUSD", "limit": 2})
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 6: Mega Mind Predictions
    print("\n6. Mega Mind Predictions:")
    try:
        response = requests.get(f"{base_url}/api/v1/mega-mind/predictions", 
                              params={"pair": "EURUSD", "style": "day_trading", "limit": 2})
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 7: Mega Mind Collaboration
    print("\n7. Mega Mind Collaboration:")
    try:
        response = requests.get(f"{base_url}/api/v1/mega-mind/collaboration", 
                              params={"pair": "EURUSD"})
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 8: Mega Mind Arena
    print("\n8. Mega Mind Arena:")
    try:
        response = requests.get(f"{base_url}/api/v1/mega-mind/arena", 
                              params={"pair": "EURUSD"})
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 9: Mega Mind Performance
    print("\n9. Mega Mind Performance:")
    try:
        response = requests.get(f"{base_url}/api/v1/mega-mind/performance")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test 10: Documentation
    print("\n10. API Documentation:")
    try:
        response = requests.get(f"{base_url}/docs")
        print(f"Status: {response.status_code}")
        print("Documentación disponible en: http://localhost:8080/docs")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("🚀 Iniciando pruebas completas de API...")
    print("Asegúrate de que el servidor esté ejecutándose en http://localhost:8080")
    print("=" * 60)
    
    # Esperar un poco para que el servidor se inicie
    time.sleep(2)
    
    test_complete_api() 