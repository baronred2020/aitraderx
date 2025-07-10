"""
API Routes para Autenticación
============================
Endpoints para login, registro y gestión de usuarios
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from typing import Optional
from datetime import datetime, timedelta
import jwt
import hashlib
import logging
import re

from models.auth_models import UserCreate, UserLogin, UserResponse, SubscriptionResponse, AuthResponse
from services.subscription_service import SubscriptionService
from services.user_service import UserService
from config.auth_config import *

logger = logging.getLogger(__name__)

# Router para autenticación
auth_router = APIRouter(prefix="/api/auth", tags=["authentication"])

# Instancias de servicios
subscription_service = SubscriptionService()
user_service = UserService()

# Usar configuración desde auth_config.py

def create_access_token(data: dict):
    """Crea un token JWT"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def validate_password(password: str) -> bool:
    """Valida la contraseña según las reglas de seguridad"""
    if len(password) < PASSWORD_MIN_LENGTH:
        return False
    
    if PASSWORD_REQUIRE_UPPERCASE and not re.search(r'[A-Z]', password):
        return False
    
    if PASSWORD_REQUIRE_LOWERCASE and not re.search(r'[a-z]', password):
        return False
    
    if PASSWORD_REQUIRE_NUMBERS and not re.search(r'\d', password):
        return False
    
    if PASSWORD_REQUIRE_SPECIAL_CHARS and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False
    
    return True

def validate_email(email: str) -> bool:
    """Valida el formato del email"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_username(username: str) -> bool:
    """Valida el nombre de usuario"""
    if len(username) < USERNAME_MIN_LENGTH or len(username) > USERNAME_MAX_LENGTH:
        return False
    
    # Solo permitir letras, números y guiones bajos
    pattern = r'^[a-zA-Z0-9_]+$'
    return re.match(pattern, username) is not None

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifica la contraseña"""
    return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password

def hash_password(password: str) -> str:
    """Hashea la contraseña"""
    return hashlib.sha256(password.encode()).hexdigest()

# ============================================================================
# ENDPOINTS DE AUTENTICACIÓN
# ============================================================================

@auth_router.post("/login", response_model=AuthResponse)
async def login(login_data: UserLogin):
    """Endpoint para login de usuarios"""
    try:
        username = login_data.username
        password = login_data.password
        
        # Verificar credenciales en la base de datos
        if not user_service.verify_password(username, password):
            raise HTTPException(status_code=401, detail="Credenciales inválidas")
        
        # Obtener usuario de la base de datos
        user = user_service.get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=401, detail="Usuario no encontrado")
        
        # Obtener suscripción del usuario
        subscription = user_service.get_user_subscription(user.user_id)
        
        # Actualizar último login
        user_service.update_last_login(user.user_id)
        
        # Crear token de acceso
        token = create_access_token(data={"sub": username})
        
        # Preparar respuesta
        user_response = {
            "id": user.user_id,
            "username": user.username,
            "email": user.email,
            "firstName": user.first_name,
            "lastName": user.last_name,
            "phone": user.phone,
            "role": user.role,
            "isActive": user.is_active,
            "createdAt": user.created_at.isoformat()
        }
        
        subscription_response = None
        if subscription:
            subscription_response = {
                "id": subscription.subscription_id,
                "planType": subscription.plan_type,
                "status": subscription.status,
                "startDate": subscription.start_date.isoformat(),
                "endDate": subscription.end_date.isoformat(),
                "isTrial": subscription.is_trial
            }
        
        return {
            "user": user_response,
            "subscription": subscription_response,
            "token": token,
            "message": "Login exitoso"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en login: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@auth_router.post("/register", response_model=AuthResponse)
async def register(register_data: UserCreate):
    """Endpoint para registro de usuarios"""
    try:
        # Los datos ya están validados por Pydantic
        username = register_data.username
        email = register_data.email
        password = register_data.password
        firstName = register_data.firstName
        lastName = register_data.lastName
        phone = register_data.phone
        plan_type = register_data.plan_type
        payment_method = register_data.payment_method
        
        # Verificar si el usuario ya existe
        if user_service.user_exists(username, email):
            raise HTTPException(status_code=400, detail="El usuario o email ya existe")
        
        # Crear nuevo usuario en la base de datos
        user_data = {
            "username": username,
            "email": email,
            "password": password,
            "firstName": firstName,
            "lastName": lastName,
            "phone": phone
        }
        
        user = user_service.create_user(user_data)
        if not user:
            raise HTTPException(status_code=500, detail="Error al crear usuario")
        
        # Crear suscripción en la base de datos
        subscription = user_service.create_subscription(user.user_id, plan_type, payment_method)
        if not subscription:
            raise HTTPException(status_code=500, detail="Error al crear suscripción")
        
        # Crear token de acceso
        token = create_access_token(data={"sub": username})
        
        # Preparar respuesta
        user_response = {
            "id": user.user_id,
            "username": user.username,
            "email": user.email,
            "firstName": user.first_name,
            "lastName": user.last_name,
            "phone": user.phone,
            "role": user.role,
            "isActive": user.is_active,
            "createdAt": user.created_at.isoformat()
        }
        
        subscription_response = {
            "id": subscription.subscription_id,
            "planType": subscription.plan_type,
            "status": subscription.status,
            "startDate": subscription.start_date.isoformat(),
            "endDate": subscription.end_date.isoformat(),
            "isTrial": subscription.is_trial
        }
        
        return {
            "user": user_response,
            "subscription": subscription_response,
            "token": token,
            "message": "Usuario registrado exitosamente"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en registro: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")

@auth_router.post("/logout")
async def logout():
    """Endpoint para logout (en JWT, el logout se maneja en el cliente)"""
    return {"message": "Logout exitoso"}

@auth_router.get("/me")
async def get_current_user(token: str):
    """Obtiene información del usuario actual"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Token inválido")
        
        # Aquí buscarías el usuario en la base de datos
        # Por ahora, devolver datos simulados
        return {
            "username": username,
            "role": "user" if username != "admin" else "admin"
        }
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expirado")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Token inválido")
    except Exception as e:
        logger.error(f"Error obteniendo usuario: {e}")
        raise HTTPException(status_code=500, detail="Error interno del servidor") 