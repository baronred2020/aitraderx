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
                
                # Retornar usuario creado
                return self.get_user_by_id(user_id)
                
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