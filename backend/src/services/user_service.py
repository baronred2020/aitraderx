"""
Servicio de Usuarios
===================
Servicio para manejar operaciones de usuarios con MySQL
"""

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta
import hashlib
import uuid
from typing import Optional, Dict, Any

from models.database_models import User, UserSubscription, SubscriptionPlan
from config.database_config import get_db

class UserService:
    """Servicio para operaciones de usuarios"""
    
    def __init__(self):
        self.db: Session = next(get_db())
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[User]:
        """Crea un nuevo usuario en la base de datos"""
        try:
            # Hash de la contraseña
            password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
            
            # Crear usuario
            user = User(
                user_id=str(uuid.uuid4()),
                username=user_data['username'],
                email=user_data['email'],
                first_name=user_data['firstName'],
                last_name=user_data['lastName'],
                phone=user_data.get('phone'),
                password_hash=password_hash,
                role='user',
                is_active=True,
                is_verified=False,
                created_at=datetime.utcnow()
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            return user
            
        except IntegrityError:
            self.db.rollback()
            return None
        except Exception as e:
            self.db.rollback()
            print(f"Error creating user: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Obtiene un usuario por nombre de usuario"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Obtiene un usuario por email"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Obtiene un usuario por ID"""
        return self.db.query(User).filter(User.user_id == user_id).first()
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verifica la contraseña de un usuario"""
        user = self.get_user_by_username(username)
        if not user:
            return False
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        return user.password_hash == password_hash
    
    def create_subscription(self, user_id: str, plan_type: str, payment_method: Optional[str] = None) -> Optional[UserSubscription]:
        """Crea una suscripción para un usuario"""
        try:
            # Obtener el plan
            plan = self.db.query(SubscriptionPlan).filter(SubscriptionPlan.plan_type == plan_type).first()
            if not plan:
                return None
            
            # Calcular fechas
            start_date = datetime.utcnow()
            if plan_type == "freemium":
                end_date = start_date + timedelta(days=365)
            else:
                end_date = start_date + timedelta(days=30)
            
            # Crear suscripción
            subscription = UserSubscription(
                subscription_id=str(uuid.uuid4()),
                user_id=user_id,
                plan_id=plan.plan_id,
                plan_type=plan_type,
                start_date=start_date,
                end_date=end_date,
                status='active',
                is_trial=False,
                payment_method=payment_method,
                created_at=datetime.utcnow()
            )
            
            self.db.add(subscription)
            self.db.commit()
            self.db.refresh(subscription)
            
            return subscription
            
        except Exception as e:
            self.db.rollback()
            print(f"Error creating subscription: {e}")
            return None
    
    def get_user_subscription(self, user_id: str) -> Optional[UserSubscription]:
        """Obtiene la suscripción activa de un usuario"""
        return self.db.query(UserSubscription).filter(
            UserSubscription.user_id == user_id,
            UserSubscription.status == 'active'
        ).first()
    
    def update_last_login(self, user_id: str):
        """Actualiza la fecha del último login"""
        user = self.get_user_by_id(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.db.commit()
    
    def user_exists(self, username: str, email: str) -> bool:
        """Verifica si un usuario ya existe"""
        return (
            self.db.query(User).filter(
                (User.username == username) | (User.email == email)
            ).first() is not None
        ) 