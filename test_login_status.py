#!/usr/bin/env python3
import requests
import json

def test_login_status():
    """Probar el login y verificar que el plan starter siempre tenga status active"""
    
    login_data = {
        "username": "user",
        "password": "user123"
    }
    
    try:
        print("🔍 Probando login con usuario starter...")
        
        response = requests.post(
            "http://localhost:8000/api/auth/login",
            json=login_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Login exitoso")
            
            # Verificar datos del usuario
            user = data.get('user', {})
            subscription = data.get('subscription', {})
            
            print(f"\n📋 Datos del usuario:")
            print(f"   ID: {user.get('id')}")
            print(f"   Username: {user.get('username')}")
            print(f"   Email: {user.get('email')}")
            
            print(f"\n📋 Datos de la suscripción:")
            print(f"   Plan Type: {subscription.get('planType')}")
            print(f"   Status: {subscription.get('status')}")
            print(f"   Is Trial: {subscription.get('isTrial')}")
            
            # Verificar que el plan starter tenga status "active"
            if subscription.get('planType') == 'starter':
                if subscription.get('status') == 'active':
                    print("✅ Plan starter con status 'active' correctamente")
                    return True
                else:
                    print(f"❌ Error: Plan starter tiene status '{subscription.get('status')}' en lugar de 'active'")
                    return False
            else:
                print(f"⚠️ Usuario tiene plan {subscription.get('planType')}, no starter")
                return True
        else:
            print(f"❌ Error en login: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_new_user_login():
    """Probar login con un usuario recién registrado"""
    
    # Primero registrar un usuario
    register_data = {
        "username": "newuser",
        "email": "newuser@example.com",
        "password": "TestPass123",
        "firstName": "New",
        "lastName": "User"
    }
    
    try:
        print("\n🔍 Registrando nuevo usuario...")
        
        register_response = requests.post(
            "http://localhost:8000/api/auth/register",
            json=register_data,
            headers={"Content-Type": "application/json"}
        )
        
        if register_response.status_code == 200:
            print("✅ Usuario registrado exitosamente")
            
            # Ahora hacer login
            login_data = {
                "username": "newuser",
                "password": "TestPass123"
            }
            
            print("🔍 Probando login con nuevo usuario...")
            
            login_response = requests.post(
                "http://localhost:8000/api/auth/login",
                json=login_data,
                headers={"Content-Type": "application/json"}
            )
            
            if login_response.status_code == 200:
                data = login_response.json()
                subscription = data.get('subscription', {})
                
                print(f"✅ Login exitoso")
                print(f"   Plan Type: {subscription.get('planType')}")
                print(f"   Status: {subscription.get('status')}")
                
                if subscription.get('planType') == 'starter' and subscription.get('status') == 'active':
                    print("✅ Nuevo usuario con plan starter y status 'active' correctamente")
                    return True
                else:
                    print(f"❌ Error: Nuevo usuario no tiene configuración correcta")
                    return False
            else:
                print(f"❌ Error en login: {login_response.status_code}")
                return False
        else:
            print(f"❌ Error en registro: {register_response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Probando que el plan starter siempre tenga status 'active'...")
    
    success1 = test_login_status()
    success2 = test_new_user_login()
    
    if success1 and success2:
        print("\n🎉 ¡Todas las pruebas pasaron!")
        print("✅ El plan starter siempre tiene status 'active' al hacer login")
    else:
        print("\n❌ Algunas pruebas fallaron") 