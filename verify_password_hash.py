#!/usr/bin/env python3
"""
Script para verificar el hash de la contraseña
"""

import hashlib

def verify_password():
    password = "user123"  # Cambiado a user123
    stored_hash = "e606e38b0d8c19b24cf0ee3808183162ea7cd63ff7912dbb22b5e803286b4446"
    
    # Generar hash de la contraseña
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    print(f"🔍 Verificando contraseña: {password}")
    print(f"Hash generado: {password_hash}")
    print(f"Hash almacenado: {stored_hash}")
    print(f"¿Coinciden? {password_hash == stored_hash}")
    
    if password_hash == stored_hash:
        print("✅ La contraseña es correcta")
    else:
        print("❌ La contraseña no coincide")
        
        # Probar con otras variaciones
        variations = [
            "user123",
            "User123",
            "USER123",
            "password123",
            "Password123",
            "PASSWORD123", 
            "password",
            "123",
            "user",
            "admin"
        ]
        
        print("\n🔍 Probando otras variaciones:")
        for var in variations:
            var_hash = hashlib.sha256(var.encode()).hexdigest()
            if var_hash == stored_hash:
                print(f"✅ Encontrada: '{var}' -> {var_hash}")

if __name__ == "__main__":
    verify_password() 