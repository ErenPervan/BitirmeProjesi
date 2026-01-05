"""
src - Autonomous Road Damage Assessment Module

Core components:
- metrics.py: Mathematical functions for severity calculation
- detector.py: YOLO model wrapper for inference and tracking
- video_processor.py: Video processing pipeline
- main.py: Entry point for offline video processing
"""

from .metrics import (
    PriorityLevel,
    calculate_circularity,
    calculate_circularity_from_contour,
    calculate_severity_score,
    get_priority_level,
    get_priority_color_bgr,
    analyze_detection
)

from .detector import (
    PotholeDetector,
    ModelLoadError
)

from .video_processor import (
    VideoProcessor,
    VideoProcessingError,
    TrackData,
    FrameResult,
    ProcessingStatus,
    ProcessingResult,
    generate_csv_report
)

from .database_manager import (
    DatabaseManager,
    DatabaseError
)

from .gps_manager import (
    GPSManager,
    GPSPoint,
    GPSDataError
)

__all__ = [
    # Metrics
    'PriorityLevel',
    'calculate_circularity',
    'calculate_circularity_from_contour',
    'calculate_severity_score',
    'get_priority_level',
    'get_priority_color_bgr',
    'analyze_detection',
    # Detector
    'PotholeDetector',
    'ModelLoadError',
    # Video Processor
    'VideoProcessor',
    'VideoProcessingError',
    'TrackData',
    'FrameResult',
    'ProcessingStatus',
    'ProcessingResult',
    'generate_csv_report',
    # Database Manager
    'DatabaseManager',
    'DatabaseError',
    # GPS Manager
    'GPSManager',
    'GPSPoint',
    'GPSDataError'
]

