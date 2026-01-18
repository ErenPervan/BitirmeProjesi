"""
exceptions.py - Custom Exception Classes

Defines application-specific exceptions for better error handling and logging.

Usage:
    from src.exceptions import VideoProcessingError
    
    raise VideoProcessingError("Failed to open video file")
"""


class PotholeSystemError(Exception):
    """Base exception for all pothole detection system errors."""
    pass


class VideoProcessingError(PotholeSystemError):
    """Raised when video processing fails."""
    pass


class GPSDataError(PotholeSystemError):
    """Raised when GPS data is invalid or corrupted."""
    pass


class ROICalibrationError(PotholeSystemError):
    """Raised when ROI calibration parameters are invalid."""
    pass


class DatabaseError(PotholeSystemError):
    """Raised when database operations fail."""
    pass


class ModelLoadError(PotholeSystemError):
    """Raised when model file cannot be loaded."""
    pass


class ConfigurationError(PotholeSystemError):
    """Raised when configuration is invalid."""
    pass
