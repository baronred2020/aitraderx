#!/usr/bin/env python3
"""
Servicio de caché optimizado para reducir el consumo de recursos de la API
"""
import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

class CacheItem:
    """Elemento individual del caché con TTL"""
    
    def __init__(self, key: str, value: Any, ttl: int = 300):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = datetime.now()
    
    def is_expired(self) -> bool:
        """Verifica si el elemento ha expirado"""
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl)
    
    def access(self):
        """Registra un acceso al elemento"""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def get_age(self) -> int:
        """Retorna la edad del elemento en segundos"""
        return int((datetime.now() - self.created_at).total_seconds())

class OptimizedCache:
    """Sistema de caché optimizado con LRU y TTL"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheItem] = OrderedDict()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0
        }
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Genera una clave única para los argumentos"""
        key_data = {
            "args": args,
            "kwargs": sorted(kwargs.items())
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor del caché"""
        if key in self.cache:
            item = self.cache[key]
            
            if item.is_expired():
                # Elemento expirado
                del self.cache[key]
                self.stats["expired"] += 1
                self.stats["misses"] += 1
                return None
            
            # Elemento válido
            item.access()
            self.cache.move_to_end(key)  # Mover al final (LRU)
            self.stats["hits"] += 1
            return item.value
        
        self.stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Almacena un valor en el caché"""
        if ttl is None:
            ttl = self.default_ttl
        
        # Limpiar elementos expirados
        self._cleanup_expired()
        
        # Si la clave ya existe, actualizarla
        if key in self.cache:
            del self.cache[key]
        
        # Si el caché está lleno, evictar el elemento más antiguo
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            self.stats["evictions"] += 1
        
        # Agregar nuevo elemento
        self.cache[key] = CacheItem(key, value, ttl)
        self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Elimina un elemento del caché"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Limpia todo el caché"""
        self.cache.clear()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "expired": 0}
    
    def _cleanup_expired(self) -> None:
        """Limpia elementos expirados"""
        expired_keys = [
            key for key, item in self.cache.items() 
            if item.is_expired()
        ]
        for key in expired_keys:
            del self.cache[key]
            self.stats["expired"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        self._cleanup_expired()
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Obtiene información detallada del caché"""
        self._cleanup_expired()
        
        # Agrupar elementos por edad
        age_groups = {"<1min": 0, "1-5min": 0, "5-15min": 0, ">15min": 0}
        
        for item in self.cache.values():
            age = item.get_age()
            if age < 60:
                age_groups["<1min"] += 1
            elif age < 300:
                age_groups["1-5min"] += 1
            elif age < 900:
                age_groups["5-15min"] += 1
            else:
                age_groups[">15min"] += 1
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "age_distribution": age_groups,
            "stats": self.get_stats()
        }

# Instancia global del caché
cache_service = OptimizedCache(max_size=1000, default_ttl=300)

class CachedAPIClient:
    """Cliente de API con caché integrado"""
    
    def __init__(self, cache: OptimizedCache = None):
        self.cache = cache or cache_service
    
    async def cached_request(self, 
                           endpoint: str, 
                           params: Dict[str, Any] = None,
                           ttl: int = 300,
                           force_refresh: bool = False) -> Any:
        """
        Realiza una petición con caché
        
        Args:
            endpoint: Endpoint de la API
            params: Parámetros de la petición
            ttl: Tiempo de vida del caché en segundos
            force_refresh: Forzar refresco ignorando caché
        """
        # Generar clave única para la petición
        cache_key = self.cache._generate_key(endpoint, params or {})
        
        # Si no se fuerza refresco, intentar obtener del caché
        if not force_refresh:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT for {endpoint}")
                return cached_result
        
        # Realizar petición real
        logger.debug(f"Cache MISS for {endpoint}, making API call")
        try:
            result = await self._make_api_request(endpoint, params)
            
            # Almacenar en caché
            self.cache.set(cache_key, result, ttl)
            
            return result
            
        except Exception as e:
            logger.error(f"API request failed for {endpoint}: {e}")
            # En caso de error, intentar devolver caché expirado como fallback
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.warning(f"Using expired cache as fallback for {endpoint}")
                return cached_result
            raise
    
    async def _make_api_request(self, endpoint: str, params: Dict[str, Any] = None) -> Any:
        """
        Realiza la petición real a la API
        Esta función debe ser implementada según el cliente HTTP que uses
        """
        # Placeholder - implementar según tu cliente HTTP
        raise NotImplementedError("Implement _make_api_request according to your HTTP client")
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalida elementos del caché que coincidan con un patrón
        """
        count = 0
        keys_to_delete = []
        
        for key in self.cache.cache.keys():
            if pattern in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            self.cache.delete(key)
            count += 1
        
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas del caché"""
        return self.cache.get_stats()

# Función de utilidad para decoradores
def cached(ttl: int = 300, key_prefix: str = ""):
    """
    Decorador para cachear resultados de funciones
    
    Args:
        ttl: Tiempo de vida del caché en segundos
        key_prefix: Prefijo para la clave del caché
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generar clave única
            cache_key = f"{key_prefix}:{cache_service._generate_key(func.__name__, *args, **kwargs)}"
            
            # Intentar obtener del caché
            cached_result = cache_service.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Ejecutar función
            result = await func(*args, **kwargs)
            
            # Almacenar en caché
            cache_service.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator 