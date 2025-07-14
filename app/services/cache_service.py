"""
Cache Service
Handles caching, performance optimization, and memory management
"""

import asyncio
import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import psutil
import gc

logger = logging.getLogger(__name__)

class CacheEntry:
    """Cache entry with metadata"""
    
    def __init__(self, key: str, value: Any, ttl: int = 3600):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = datetime.now() + timedelta(seconds=ttl)
        self.access_count = 0
        self.last_accessed = datetime.now()
        self.size_bytes = self._calculate_size(value)
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of cached value"""
        try:
            if isinstance(value, pd.DataFrame):
                return value.memory_usage(deep=True).sum()
            elif isinstance(value, (dict, list)):
                return len(pickle.dumps(value))
            else:
                return len(str(value).encode('utf-8'))
        except:
            return 1024  # Default size estimate
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() > self.expires_at
    
    def access(self) -> Any:
        """Access cached value and update statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        return self.value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for monitoring"""
        return {
            "key": self.key,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "access_count": self.access_count,
            "last_accessed": self.last_accessed.isoformat(),
            "size_bytes": self.size_bytes,
            "is_expired": self.is_expired()
        }

class PerformanceMonitor:
    """Monitor system performance and resource usage"""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.alert_thresholds = {
            "memory_percent": 85.0,
            "cpu_percent": 80.0,
            "disk_percent": 90.0
        }
        self.alerts: List[Dict[str, Any]] = []
    
    def collect_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available
            memory_used = memory.used
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            process_cpu = process.cpu_percent()
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "process_percent": process_cpu
                },
                "memory": {
                    "percent": memory_percent,
                    "available_bytes": memory_available,
                    "used_bytes": memory_used,
                    "total_bytes": memory.total,
                    "process_rss": process_memory.rss,
                    "process_vms": process_memory.vms
                },
                "disk": {
                    "percent": disk_percent,
                    "free_bytes": disk.free,
                    "used_bytes": disk.used,
                    "total_bytes": disk.total
                }
            }
            
            # Check for alerts
            self._check_alerts(metrics)
            
            # Store metrics (keep last 1000 entries)
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}
    
    def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds"""
        alerts = []
        
        # Memory alert
        memory_percent = metrics.get("memory", {}).get("percent", 0)
        if memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append({
                "type": "memory_high",
                "message": f"Memory usage is high: {memory_percent:.1f}%",
                "severity": "warning" if memory_percent < 95 else "critical",
                "timestamp": datetime.now().isoformat(),
                "value": memory_percent
            })
        
        # CPU alert
        cpu_percent = metrics.get("cpu", {}).get("percent", 0)
        if cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append({
                "type": "cpu_high",
                "message": f"CPU usage is high: {cpu_percent:.1f}%",
                "severity": "warning" if cpu_percent < 95 else "critical",
                "timestamp": datetime.now().isoformat(),
                "value": cpu_percent
            })
        
        # Disk alert
        disk_percent = metrics.get("disk", {}).get("percent", 0)
        if disk_percent > self.alert_thresholds["disk_percent"]:
            alerts.append({
                "type": "disk_high",
                "message": f"Disk usage is high: {disk_percent:.1f}%",
                "severity": "warning" if disk_percent < 98 else "critical",
                "timestamp": datetime.now().isoformat(),
                "value": disk_percent
            })
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m["timestamp"]) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        # Calculate averages
        cpu_values = [m["cpu"]["percent"] for m in recent_metrics]
        memory_values = [m["memory"]["percent"] for m in recent_metrics]
        disk_values = [m["disk"]["percent"] for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu": {
                "avg": np.mean(cpu_values),
                "max": np.max(cpu_values),
                "min": np.min(cpu_values)
            },
            "memory": {
                "avg": np.mean(memory_values),
                "max": np.max(memory_values),
                "min": np.min(memory_values)
            },
            "disk": {
                "avg": np.mean(disk_values),
                "max": np.max(disk_values),
                "min": np.min(disk_values)
            }
        }

class CacheService:
    """High-performance caching service with automatic optimization"""
    
    def __init__(self, max_size_mb: int = 500, default_ttl: int = 3600):
        self.cache: Dict[str, CacheEntry] = {}
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert MB to bytes
        self.default_ttl = default_ttl
        self.performance_monitor = PerformanceMonitor()
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_requests": 0
        }
        
        # Background tasks will be started when needed
        self._tasks_started = False
    
    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = {
            "prefix": prefix,
            "args": args,
            "kwargs": kwargs
        }
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        self.stats["total_requests"] += 1
        
        if key in self.cache:
            entry = self.cache[key]
            
            if entry.is_expired():
                del self.cache[key]
                self.stats["misses"] += 1
                return None
            
            self.stats["hits"] += 1
            return entry.access()
        
        self.stats["misses"] += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if ttl is None:
            ttl = self.default_ttl
        
        # Create cache entry
        entry = CacheEntry(key, value, ttl)
        
        # Check if we need to make space
        await self._ensure_space(entry.size_bytes)
        
        self.cache[key] = entry
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        gc.collect()  # Force garbage collection
    
    async def _ensure_space(self, required_bytes: int):
        """Ensure there's enough space in cache"""
        current_size = self._get_total_size()
        
        if current_size + required_bytes <= self.max_size_bytes:
            return
        
        # Need to evict entries
        entries_by_score = []
        
        for entry in self.cache.values():
            # Calculate eviction score (lower = more likely to evict)
            age_factor = (datetime.now() - entry.created_at).total_seconds() / 3600  # Hours
            access_factor = 1 / (entry.access_count + 1)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            
            score = access_factor * age_factor * size_factor
            entries_by_score.append((score, entry.key))
        
        # Sort by score (lowest first)
        entries_by_score.sort()
        
        # Evict entries until we have enough space
        space_needed = current_size + required_bytes - self.max_size_bytes
        space_freed = 0
        
        for score, key in entries_by_score:
            if space_freed >= space_needed:
                break
            
            if key in self.cache:
                space_freed += self.cache[key].size_bytes
                del self.cache[key]
                self.stats["evictions"] += 1
        
        # Force garbage collection after eviction
        gc.collect()
    
    def _get_total_size(self) -> int:
        """Get total cache size in bytes"""
        return sum(entry.size_bytes for entry in self.cache.values())
    
    async def _cleanup_loop(self):
        """Background task to clean up expired entries"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                expired_keys = [
                    key for key, entry in self.cache.items()
                    if entry.is_expired()
                ]
                
                for key in expired_keys:
                    del self.cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
                    gc.collect()
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    async def _monitoring_loop(self):
        """Background task to collect performance metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                self.performance_monitor.collect_metrics()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    # Decorator for caching function results
    def cached(self, prefix: str = "func", ttl: int = None):
        """Decorator to cache function results"""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(prefix, func.__name__, *args, **kwargs)
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                await self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    # Cache management methods
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats["total_requests"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "total_entries": len(self.cache),
            "total_size_bytes": self._get_total_size(),
            "total_size_mb": self._get_total_size() / (1024 * 1024),
            "max_size_mb": self.max_size_bytes / (1024 * 1024),
            "utilization_percent": (self._get_total_size() / self.max_size_bytes) * 100,
            "hit_rate_percent": hit_rate,
            "stats": self.stats
        }
    
    def get_cache_entries(self) -> List[Dict[str, Any]]:
        """Get information about cache entries"""
        return [entry.to_dict() for entry in self.cache.values()]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.performance_monitor.collect_metrics()
    
    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get performance summary"""
        return self.performance_monitor.get_metrics_summary(hours)
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get performance alerts"""
        return self.performance_monitor.alerts
    
    # Optimization methods
    async def optimize_cache(self):
        """Optimize cache performance"""
        # Remove expired entries
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry.is_expired()
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        # If still over capacity, remove least used entries
        if self._get_total_size() > self.max_size_bytes * 0.8:  # 80% threshold
            await self._ensure_space(0)
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cache optimization completed")
    
    async def preload_common_data(self, datasets: List[str]):
        """Preload commonly used datasets into cache"""
        from app.services.file_service import FileService
        
        file_service = FileService()
        
        for dataset_id in datasets:
            try:
                df = file_service.get_dataframe(dataset_id)
                if df is not None:
                    cache_key = self._generate_key("dataset", dataset_id)
                    await self.set(cache_key, df, ttl=7200)  # 2 hours
                    
            except Exception as e:
                logger.error(f"Error preloading dataset {dataset_id}: {e}")
        
        logger.info(f"Preloaded {len(datasets)} datasets into cache")
