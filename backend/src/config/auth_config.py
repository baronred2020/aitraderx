"""
Configuración de Autenticación
=============================
Configuración para JWT y autenticación
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Configuración de JWT
SECRET_KEY = os.getenv("SECRET_KEY", "tu_clave_secreta_muy_segura_aqui_cambiala_en_produccion")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Configuración de contraseñas
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRE_UPPERCASE = True
PASSWORD_REQUIRE_LOWERCASE = True
PASSWORD_REQUIRE_NUMBERS = True
PASSWORD_REQUIRE_SPECIAL_CHARS = True

# Configuración de nombres de usuario
USERNAME_MIN_LENGTH = 3
USERNAME_MAX_LENGTH = 20

# Security scheme
security = HTTPBearer()

def create_access_token(data: dict):
    """Crea un token JWT"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verifica un token JWT"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """Obtiene el usuario actual basado en el token JWT"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=401,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    username = payload.get("sub")
    if username is None:
        raise HTTPException(
            status_code=401,
            detail="Token inválido",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"username": username} 