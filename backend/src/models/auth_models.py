"""
Modelos de Autenticación
========================
Modelos Pydantic para validación de datos de autenticación
"""

from pydantic import BaseModel, EmailStr, validator
from typing import Optional
from datetime import datetime

class UserCreate(BaseModel):
    """Modelo para creación de usuarios"""
    username: str
    email: str
    password: str
    firstName: str
    lastName: str
    phone: Optional[str] = None
    plan_type: str = "freemium"
    payment_method: Optional[str] = None

    @validator('username')
    def validate_username(cls, v):
        if len(v) < 3 or len(v) > 20:
            raise ValueError('El nombre de usuario debe tener entre 3 y 20 caracteres')
        if not v.replace('_', '').isalnum():
            raise ValueError('El nombre de usuario solo puede contener letras, números y guiones bajos')
        return v

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v or '.' not in v:
            raise ValueError('Formato de email inválido')
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('La contraseña debe tener al menos 8 caracteres')
        if not any(c.isupper() for c in v):
            raise ValueError('La contraseña debe contener al menos una mayúscula')
        if not any(c.islower() for c in v):
            raise ValueError('La contraseña debe contener al menos una minúscula')
        if not any(c.isdigit() for c in v):
            raise ValueError('La contraseña debe contener al menos un número')
        return v

class UserLogin(BaseModel):
    """Modelo para login de usuarios"""
    username: str
    password: str

class UserResponse(BaseModel):
    """Modelo para respuesta de usuario"""
    id: str
    username: str
    email: str
    firstName: str
    lastName: str
    phone: Optional[str] = None
    role: str
    isActive: bool
    createdAt: datetime

class SubscriptionResponse(BaseModel):
    """Modelo para respuesta de suscripción"""
    id: str
    planType: str
    status: str
    startDate: datetime
    endDate: datetime
    isTrial: bool

class AuthResponse(BaseModel):
    """Modelo para respuesta de autenticación"""
    user: UserResponse
    subscription: Optional[SubscriptionResponse] = None
    token: str
    message: Optional[str] = None 