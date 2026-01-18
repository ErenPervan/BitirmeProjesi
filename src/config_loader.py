"""
config_loader.py - Configuration Management

Centralized configuration loading from config.yaml file.
Provides default values and validation for all system parameters.

Usage:
    from src.config_loader import load_config, get_config
    
    config = load_config()
    confidence = config['model']['confidence']
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# Global config cache
_config_cache: Dict[str, Any] = None


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with defaults.
    
    Args:
        config_path: Path to config.yaml (default: PROJECT_ROOT/config.yaml)
        
    Returns:
        Dictionary containing all configuration parameters
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    global _config_cache
    
    # Use cached config if available
    if _config_cache is not None:
        return _config_cache
    
    # Determine config path
    if config_path is None:
        project_root = Path(__file__).parent.parent
        config_path = project_root / "config.yaml"
    else:
        config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        _config_cache = get_default_config()
        return _config_cache
    
    # Load YAML
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from: {config_path}")
        _config_cache = config
        return config
        
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        logger.warning("Using default configuration")
        _config_cache = get_default_config()
        return _config_cache


def get_config() -> Dict[str, Any]:
    """
    Get current configuration (loads if not already loaded).
    
    Returns:
        Dictionary containing all configuration parameters
    """
    if _config_cache is None:
        return load_config()
    return _config_cache


def reload_config(config_path: str = None) -> Dict[str, Any]:
    """
    Force reload configuration from file.
    
    Args:
        config_path: Path to config.yaml
        
    Returns:
        Dictionary containing all configuration parameters
    """
    global _config_cache
    _config_cache = None
    return load_config(config_path)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration values.
    Used as fallback if config.yaml is missing.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        'model': {
            'path': 'best1.engine',
            'confidence': 0.5,
            'iou_threshold': 0.5,
            'tracker_config': 'bytetrack.yaml'
        },
        'roi': {
            'top_width': 40.0,
            'bottom_width': 90.0,
            'horizon': 60.0,
            'bottom_height': 90.0,
            'exit_line_y_ratio': 85.0
        },
        'processing': {
            'min_track_frames': 5,
            'proximity_threshold': 0.15,
            'best_frame_pad': 50,
            'ui_update_interval': 200
        },
        'ipm': {
            'output_width': 600,
            'output_height': 400
        },
        'severity': {
            'circularity_weight': 0.4,
            'area_weight': 0.6
        },
        'risk': {
            'low_threshold': 40.0,
            'medium_threshold': 65.0
        },
        'paths': {
            'output_base': 'runs/streamlit',
            'snapshot_dir': 'data/snapshots',
            'logs_dir': 'logs',
            'temp_dir': 'temp'
        },
        'logging': {
            'level': 'INFO',
            'max_file_size_mb': 10,
            'backup_count': 5,
            'console_logging': True,
            'file_logging': True
        },
        'video': {
            'codec': 'mp4v',
            'quality': 95,
            'draw_roi': True,
            'draw_exit_line': True,
            'draw_tracks': True
        },
        'database': {
            'auto_backup': False,
            'backup_on_startup': False
        },
        'gps': {
            'enabled': True,
            'required': False
        },
        'performance': {
            'max_memory_mb': 2048,
            'enable_tensorrt': True,
            'batch_processing': False
        },
        'ui': {
            'theme': 'dark',
            'show_debug_info': False,
            'auto_refresh_interval': 1.0
        }
    }


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid, False otherwise
        
    Logs warnings for invalid values.
    """
    valid = True
    
    # Validate model confidence
    if not 0.0 <= config['model']['confidence'] <= 1.0:
        logger.warning("Invalid confidence threshold. Must be 0.0-1.0")
        valid = False
    
    # Validate ROI percentages
    roi = config['roi']
    for key in ['top_width', 'bottom_width', 'horizon', 'bottom_height', 'exit_line_y_ratio']:
        if not 0.0 <= roi[key] <= 100.0:
            logger.warning(f"Invalid ROI parameter '{key}'. Must be 0-100")
            valid = False
    
    # Validate severity weights
    weights_sum = config['severity']['circularity_weight'] + config['severity']['area_weight']
    if abs(weights_sum - 1.0) > 0.01:
        logger.warning(f"Severity weights sum to {weights_sum}, expected 1.0")
        valid = False
    
    return valid
