"""
User Service for AI Trading System
"""
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import hashlib
import secrets
import string

# Importar configuración de base de datos
from config.database_config import db_config

logger = logging.getLogger(__name__)

class UserService:
    """Servicio para operaciones de usuarios"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_user_by_id(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene un usuario por ID"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                query = "SELECT * FROM users WHERE user_id = %s"
                cursor.execute(query, (user_id,))
                result = cursor.fetchone()
                cursor.close()
                
                if result:
                    # Convertir a diccionario
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting user by ID: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Obtiene un usuario por nombre de usuario"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                query = "SELECT * FROM users WHERE username = %s"
                cursor.execute(query, (username,))
                result = cursor.fetchone()
                cursor.close()
                
                if result:
                    # Convertir a diccionario
                    columns = [desc[0] for desc in cursor.description]
                    return dict(zip(columns, result))
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting user by username: {e}")
            return None
    
    def verify_password(self, username: str, password: str) -> bool:
        """Verifica la contraseña de un usuario"""
        try:
            user = self.get_user_by_username(username)
            if not user:
                return False
            
            # Verificar contraseña (implementación básica)
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            return user.get('password_hash') == password_hash
            
        except Exception as e:
            self.logger.error(f"Error verifying password: {e}")
            return False
    
    def create_user(self, user_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Crea un nuevo usuario en la base de datos"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                # Generar user_id único
                user_id = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(36))
                
                # Hash de la contraseña
                password_hash = hashlib.sha256(user_data['password'].encode()).hexdigest()
                
                # Insertar usuario
                insert_query = """
                    INSERT INTO users (user_id, username, email, first_name, last_name, 
                                     phone, password_hash, is_active, is_verified, role, plan_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    user_id,
                    user_data['username'],
                    user_data['email'],
                    user_data.get('firstName', ''),
                    user_data.get('lastName', ''),
                    user_data.get('phone', ''),
                    password_hash,
                    True,
                    False,
                    'user',
                    'starter'
                ))
                
                connection.commit()
                cursor.close()
                
                # Retornar usuario creado como objeto con atributos
                return {
                    'user_id': user_id,
                    'username': user_data['username'],
                    'email': user_data['email'],
                    'first_name': user_data.get('firstName', ''),
                    'last_name': user_data.get('lastName', ''),
                    'phone': user_data.get('phone', ''),
                    'role': 'user',
                    'is_active': True,
                    'created_at': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"Error creating user: {e}")
            return None
    
    def user_exists(self, username: str, email: str) -> bool:
        """Verifica si un usuario ya existe"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                query = "SELECT COUNT(*) FROM users WHERE username = %s OR email = %s"
                cursor.execute(query, (username, email))
                count = cursor.fetchone()[0]
                cursor.close()
                
                return count > 0
                
        except Exception as e:
            self.logger.error(f"Error checking if user exists: {e}")
            return False
    
    def create_subscription(self, user_id: str, plan_type: str = "starter", payment_method: str = None) -> Optional[Dict[str, Any]]:
        """Crea una suscripción para un usuario"""
        try:
            with db_config.get_connection() as connection:
                cursor = connection.cursor()
                
                # Generar subscription_id único
                subscription_id = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(36))
                
                # Obtener el plan de la base de datos
                cursor.execute("SELECT plan_id FROM subscription_plans WHERE plan_type = %s", (plan_type,))
                plan_result = cursor.fetchone()
                
                if not plan_result:
                    self.logger.error(f"Plan type {plan_type} not found")
                    return None
                
                plan_id = plan_result[0]
                
                # Fechas de suscripción
                start_date = datetime.now()
                end_date = start_date + timedelta(days=30)  # 30 días por defecto
                
                # Insertar suscripción usando la estructura real de la tabla
                insert_query = """
                    INSERT INTO user_subscriptions (
                        subscription_id, user_id, plan_id, plan_type, start_date, 
                        end_date, status, is_trial, payment_method, auto_renew
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                cursor.execute(insert_query, (
                    subscription_id,
                    user_id,
                    plan_id,
                    plan_type,
                    start_date,
                    end_date,
                    'active',
                    plan_type == 'starter',  # Es trial si es starter
                    payment_method,
                    True  # Auto renew por defecto
                ))
                
                connection.commit()
                cursor.close()
                
                # Retornar suscripción creada
                return {
                    'subscription_id': subscription_id,
                    'user_id': user_id,
                    'plan_id': plan_id,
                    'plan_type': plan_type,
                    'status': 'active',
                    'start_date': start_date,
                    'end_date': end_date,
                    'is_trial': plan_type == 'starter'
                }
                
        except Exception as e:
            self.logger.error(f"Error creating subscription: {e}")
            return None 