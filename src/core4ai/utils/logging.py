"""
Logging utilities for Core4AI.

This module provides consistent logging configuration across all components.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any

# Default log format
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# Default log level
DEFAULT_LEVEL = logging.INFO

# Log directory
LOG_DIR = Path.home() / ".core4ai" / "logs"

def ensure_log_dir():
    """Ensure the log directory exists."""
    if not LOG_DIR.exists():
        LOG_DIR.mkdir(parents=True, exist_ok=True)

def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a logger with the specified name and level.
    
    Args:
        name: Logger name (typically the module name)
        level: Logging level (defaults to DEFAULT_LEVEL)
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level or DEFAULT_LEVEL)
    
    # Avoid adding handlers if they already exist
    if not logger.handlers:
        # Add console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        logger.addHandler(console_handler)
    
    return logger

def setup_file_logging(
    name: str, 
    log_file: Optional[str] = None,
    level: Optional[int] = None,
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging to a file.
    
    Args:
        name: Logger name
        log_file: Log file name (defaults to {name}.log)
        level: Logging level (defaults to DEFAULT_LEVEL)
        format_str: Log format string (defaults to DEFAULT_FORMAT)
        
    Returns:
        Configured logger
    """
    logger = get_logger(name, level)
    
    # Ensure log directory exists
    ensure_log_dir()
    
    # Default log file
    if not log_file:
        log_file = f"{name.split('.')[-1]}.log"
    
    log_path = LOG_DIR / log_file
    
    # Avoid adding handlers if they already exist
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    if not file_handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(format_str or DEFAULT_FORMAT))
        logger.addHandler(file_handler)
    
    return logger

def configure_root_logger(
    level: Optional[int] = None,
    console: bool = True,
    file: bool = False,
    log_file: Optional[str] = "core4ai.log",
    format_str: Optional[str] = None
) -> None:
    """
    Configure the root logger.
    
    Args:
        level: Logging level (defaults to DEFAULT_LEVEL)
        console: Whether to log to console
        file: Whether to log to file
        log_file: Log file name
        format_str: Log format string
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level or DEFAULT_LEVEL)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(format_str or DEFAULT_FORMAT)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # Add file handler
    if file:
        ensure_log_dir()
        file_handler = logging.FileHandler(LOG_DIR / log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def set_log_level(level: Union[int, str], logger_name: Optional[str] = None) -> None:
    """
    Set the logging level for a logger.
    
    Args:
        level: Logging level (int or string name)
        logger_name: Logger name (defaults to root logger)
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(level)

def log_dict(logger: logging.Logger, data: Dict[str, Any], message: str = "Dictionary contents:") -> None:
    """
    Log a dictionary with proper formatting.
    
    Args:
        logger: Logger to use
        data: Dictionary to log
        message: Prefix message
    """
    import json
    
    logger.debug(f"{message}\n{json.dumps(data, indent=2, default=str)}")

# Initialize package logging
def init():
    """Initialize package logging."""
    # Don't add handlers to the root logger by default
    # This allows the application to configure logging as needed
    logging.getLogger("core4ai").setLevel(DEFAULT_LEVEL)
    
    # Configure urllib3 and requests to be less verbose
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    
    # Set lower level for some noisy libraries
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)

# Run initialization when module is imported
init()