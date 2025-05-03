import time
import logging
import functools
import gc
import psutil
import threading
from typing import Callable, Any, Optional

logger = logging.getLogger(__name__)

def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log the execution time of functions.
    
    Args:
        func: The function to be timed
        
    Returns:
        The wrapped function with timing functionality
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        logger.debug(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        
        return result
    
    return wrapper

def memory_usage_decorator(func: Callable) -> Callable:
    """
    Decorator to measure and log the memory usage of functions.
    
    Args:
        func: The function to monitor memory usage
        
    Returns:
        The wrapped function with memory monitoring
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get current process
        process = psutil.Process()
        
        # Memory before execution
        gc.collect()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Memory after execution
        gc.collect()
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_diff = memory_after - memory_before
        logger.debug(f"Function '{func.__name__}' memory usage: {memory_diff:.2f} MB")
        
        return result
    
    return wrapper

def retry_decorator(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0, 
                   exceptions: tuple = (Exception,)) -> Callable:
    """
    Decorator to retry a function on failure.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Backoff multiplier for delay
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        The wrapped function with retry functionality
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = max_attempts, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    msg = f"{str(e)}, Retrying in {mdelay} seconds..."
                    logger.warning(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def async_execution(func: Callable) -> Callable:
    """
    Decorator to execute a function asynchronously in a separate thread.
    
    Args:
        func: The function to execute asynchronously
        
    Returns:
        The wrapped function that executes asynchronously
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    
    return wrapper

def validate_inputs(validator: Callable) -> Callable:
    """
    Decorator to validate function inputs.
    
    Args:
        validator: Function that validates the inputs
        
    Returns:
        The wrapped function with input validation
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError(f"Invalid inputs to {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def singleton(cls: Any) -> Any:
    """
    Decorator to implement the singleton pattern for classes.
    
    Args:
        cls: The class to be made a singleton
        
    Returns:
        The singleton class
    """
    instances = {}
    
    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

def cache_result(ttl: Optional[float] = None) -> Callable:
    """
    Decorator to cache function results for a specified time-to-live.
    
    Args:
        ttl: Time-to-live for cached results in seconds (None for indefinite)
        
    Returns:
        The wrapped function with caching
    """
    def decorator(func: Callable) -> Callable:
        cache = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create a cache key from the function arguments
            key = str(args) + str(kwargs)
            
            # Check if result is in cache and not expired
            if key in cache:
                result, timestamp = cache[key]
                if ttl is None or time.time() - timestamp < ttl:
                    return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache[key] = (result, time.time())
            
            return result
        
        # Add method to clear cache
        wrapper.clear_cache = lambda: cache.clear()
        
        return wrapper
    
    return decorator
