import logging
import os
import sys
import datetime
import json
from logging.handlers import RotatingFileHandler
from typing import Dict, Any, Optional

def setup_logging(log_dir: str = "logs", level: int = logging.INFO, 
                 max_size_mb: int = 10, backup_count: int = 5) -> None:
    """
    Set up logging with both console and file handlers.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level (default: INFO)
        max_size_mb: Maximum size of log files in MB before rotation
        backup_count: Number of backup files to keep
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dvc_text_classification_{timestamp}.log")
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler with rotation
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_size_mb * 1024 * 1024,
        backupCount=backup_count
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    logging.info(f"Logging initialized. Log file: {log_file}")

class JsonFormatter(logging.Formatter):
    """
    Format logs as JSON for better parsing by log analysis tools.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON."""
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'name': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def add_json_handler(log_dir: str = "logs", level: int = logging.INFO) -> None:
    """
    Add a JSON handler to the root logger.
    
    Args:
        log_dir: Directory to store log files
        level: Logging level for the JSON handler
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate JSON log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    json_log_file = os.path.join(log_dir, f"dvc_text_classification_{timestamp}.json")
    
    # Create JSON handler
    json_handler = RotatingFileHandler(
        json_log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5
    )
    json_handler.setLevel(level)
    json_handler.setFormatter(JsonFormatter())
    
    # Add handler to root logger
    logging.getLogger().addHandler(json_handler)
    logging.info(f"JSON logging initialized. Log file: {json_log_file}")

class LogContextManager:
    """
    Context manager for adding context to log messages.
    """
    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize the context manager with a logger and context values.
        
        Args:
            logger: The logger to add context to
            **context: Key-value pairs to add to the log context
        """
        self.logger = logger
        self.context = context
        self.old_factory = logging.getLogRecordFactory()
    
    def __enter__(self):
        """Add context to log records when entering the context."""
        old_factory = self.old_factory
        context = self.context
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log record factory when exiting the context."""
        logging.setLogRecordFactory(self.old_factory)

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Name of the logger
        level: Logging level (if None, uses root logger's level)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger
