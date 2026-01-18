"""
logger.py - Centralized Logging Configuration

Provides consistent logging across all modules with:
- File rotation (prevents huge log files)
- Console and file output
- Configurable log levels
- Structured logging format

Usage:
    from src.logger import setup_logging, get_logger
    
    setup_logging()  # Call once at application start
    logger = get_logger(__name__)
    logger.info("Application started")
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5,
    console_output: bool = True,
    file_output: bool = True
) -> None:
    """
    Configure logging system for the entire application.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        max_bytes: Maximum size of each log file before rotation
        backup_count: Number of backup files to keep
        console_output: Enable console logging
        file_output: Enable file logging
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Convert log level string to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_output:
        log_file = log_path / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Log startup message
    root_logger.info("=" * 80)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log level: {log_level}")
    root_logger.info(f"Log directory: {log_path.absolute()}")
    root_logger.info("=" * 80)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the module)
        
    Returns:
        Logger instance
        
    Example:
        logger = get_logger(__name__)
        logger.info("Processing frame 42")
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, message: str, exc_info: bool = True) -> None:
    """
    Log an exception with full traceback.
    
    Args:
        logger: Logger instance
        message: Error message
        exc_info: Include exception info (default: True)
    """
    logger.error(message, exc_info=exc_info)


class LoggerContext:
    """
    Context manager for temporary log level changes.
    
    Usage:
        with LoggerContext('my_module', logging.DEBUG):
            # Detailed logging for this block
            process_data()
    """
    
    def __init__(self, logger_name: str, level: int):
        """
        Initialize context.
        
        Args:
            logger_name: Name of logger to modify
            level: Temporary log level
        """
        self.logger = logging.getLogger(logger_name)
        self.original_level = self.logger.level
        self.new_level = level
    
    def __enter__(self):
        """Enter context - set new log level."""
        self.logger.setLevel(self.new_level)
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore original log level."""
        self.logger.setLevel(self.original_level)
        return False


# Convenience function for quick debugging
def debug_log(message: str, data: dict = None) -> None:
    """
    Quick debug logging with optional data dump.
    
    Args:
        message: Debug message
        data: Optional dictionary to dump
    """
    logger = get_logger('DEBUG')
    logger.debug(message)
    if data:
        import json
        logger.debug(json.dumps(data, indent=2, default=str))
