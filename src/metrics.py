"""
metrics.py - Advanced Severity Metrics for Pothole Detection

This module contains mathematical functions for calculating:
1. Shape Irregularity Index (Circularity)
2. Severity Score (0-100 scale)
3. Priority Level Classification

Mathematical Foundation:
------------------------
Circularity Formula: C = (4 * π * A) / P²
Where:
    A = Area of the polygon (pixel count)
    P = Perimeter of the polygon
    
    C = 1.0 → Perfect circle
    C < 0.6 → Jagged/irregular shape (indicates severe damage)
"""

import math
from typing import Tuple, List, Optional
from enum import Enum
import numpy as np

try:
    from shapely.geometry import Polygon
    from shapely.validation import make_valid
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False


class PriorityLevel(Enum):
    """Priority classification for detected potholes."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def calculate_circularity(area: float, perimeter: float) -> float:
    """
    Calculate the circularity (shape irregularity index) of a polygon.
    
    Formula: Circularity = (4 * π * Area) / (Perimeter²)
    
    Args:
        area: Area of the polygon in pixels
        perimeter: Perimeter of the polygon in pixels
        
    Returns:
        Circularity value between 0 and 1.
        - 1.0 = Perfect circle
        - < 0.6 = Irregular/jagged shape (severe damage indicator)
        - 0.0 = Degenerate case (zero perimeter)
        
    Note:
        Values slightly > 1.0 may occur due to numerical precision;
        these are clamped to 1.0.
    """
    if perimeter <= 0:
        return 0.0
    
    circularity = (4 * math.pi * area) / (perimeter ** 2)
    
    # Clamp to [0, 1] range for numerical stability
    return min(max(circularity, 0.0), 1.0)


def calculate_circularity_from_polygon(polygon: "Polygon") -> float:
    """
    Calculate circularity directly from a Shapely Polygon object.
    
    Args:
        polygon: Shapely Polygon object representing the detected region
        
    Returns:
        Circularity value between 0 and 1
        
    Raises:
        ImportError: If Shapely is not installed
    """
    if not SHAPELY_AVAILABLE:
        raise ImportError("Shapely is required for polygon analysis. "
                          "Install with: pip install shapely")
    
    if polygon.is_empty:
        return 0.0
    
    # Ensure polygon is valid
    if not polygon.is_valid:
        polygon = make_valid(polygon)
    
    area = polygon.area
    perimeter = polygon.length  # In Shapely, 'length' gives the perimeter
    
    return calculate_circularity(area, perimeter)


def calculate_circularity_from_contour(contour: np.ndarray) -> float:
    """
    Calculate circularity from an OpenCV contour array.
    
    This is useful when working directly with cv2.findContours output
    without converting to Shapely.
    
    Args:
        contour: OpenCV contour array of shape (N, 1, 2) or (N, 2)
        
    Returns:
        Circularity value between 0 and 1
    """
    import cv2
    
    # Ensure contour has correct shape
    if contour.ndim == 3:
        contour = contour.squeeze()
    
    if len(contour) < 3:
        return 0.0
    
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    
    return calculate_circularity(area, perimeter)


def calculate_severity_score(
    area: float,
    frame_area: float,
    circularity: float,
    area_weight: float = 0.6,
    irregularity_weight: float = 0.4
) -> float:
    """
    Calculate a composite severity score (0-100) for a detected pothole.
    
    The score is based on two factors:
    1. Relative Area: How much of the frame the pothole occupies
    2. Irregularity: How jagged/irregular the shape is (inverse of circularity)
    
    Scoring Logic:
    --------------
    - Large area + irregular shape = HIGH severity (score closer to 100)
    - Small area + circular shape = LOW severity (score closer to 0)
    
    Args:
        area: Area of the detected pothole in pixels
        frame_area: Total area of the frame in pixels (width * height)
        circularity: Circularity value (0-1) from calculate_circularity()
        area_weight: Weight for area component (default: 0.6)
        irregularity_weight: Weight for irregularity component (default: 0.4)
        
    Returns:
        Severity score from 0 to 100
        
    Formula:
        area_ratio = (area / frame_area) * normalization_factor
        irregularity = 1 - circularity
        severity = (area_weight * area_score + irregularity_weight * irregularity_score) * 100
    """
    if frame_area <= 0:
        return 0.0
    
    # Calculate relative area ratio
    # Normalize: assume a pothole covering 5% of frame is "maximum severity" for area
    # This can be adjusted based on real-world calibration
    MAX_AREA_RATIO = 0.05  # 5% of frame = max area score
    area_ratio = area / frame_area
    area_score = min(area_ratio / MAX_AREA_RATIO, 1.0)
    
    # Calculate irregularity score (inverse of circularity)
    # Lower circularity = more jagged = higher irregularity
    irregularity_score = 1.0 - circularity
    
    # Combine scores with weights
    combined_score = (
        area_weight * area_score +
        irregularity_weight * irregularity_score
    )
    
    # Scale to 0-100 and clamp
    severity = combined_score * 100
    return min(max(severity, 0.0), 100.0)


def get_priority_level(severity_score: float) -> PriorityLevel:
    """
    Classify severity score into priority levels (risk labels).
    
    Thresholds (Updated for Graduation Project):
    ---------------------------------------------
    - LOW (Green): 0-30 (minor damage, can be scheduled for later repair)
    - MEDIUM (Yellow): 30-70 (moderate damage, should be addressed soon)
    - HIGH (Red): 70-100 (severe damage, requires immediate attention)
    
    Args:
        severity_score: Score from 0 to 100
        
    Returns:
        PriorityLevel enum value (LOW, MEDIUM, or HIGH)
    """
    if severity_score < 30:
        return PriorityLevel.LOW
    elif severity_score < 70:
        return PriorityLevel.MEDIUM
    else:
        return PriorityLevel.HIGH


def get_risk_label(severity_score: float) -> str:
    """
    Get human-readable risk label for a severity score (Turkish).
    
    Convenience function that returns the Turkish string representation.
    
    Args:
        severity_score: Score from 0 to 100
        
    Returns:
        Risk label string in Turkish: 'Düşük', 'Orta', or 'Yüksek'
    """
    priority = get_priority_level(severity_score)
    # Turkish labels for presentation
    turkish_labels = {
        PriorityLevel.LOW: "Düşük",
        PriorityLevel.MEDIUM: "Orta",
        PriorityLevel.HIGH: "Yüksek"
    }
    return turkish_labels.get(priority, "Bilinmiyor")


def get_priority_color_bgr(priority: PriorityLevel) -> Tuple[int, int, int]:
    """
    Get BGR color tuple for visualization based on priority level.
    
    Color Scheme (OpenCV uses BGR format) - Plan 5.1 Color Coding:
    - LOW: Green (0, 255, 0) - Score 0-30
    - MEDIUM: Yellow (0, 255, 255) - Score 30-70 (changed from Orange)
    - HIGH: Red (0, 0, 255) - Score 70-100
    
    Args:
        priority: PriorityLevel enum value
        
    Returns:
        Tuple of (Blue, Green, Red) values for OpenCV
    """
    color_map = {
        PriorityLevel.LOW: (0, 255, 0),       # Green
        PriorityLevel.MEDIUM: (0, 255, 255),  # Yellow (changed for visibility)
        PriorityLevel.HIGH: (0, 0, 255),      # Red
    }
    return color_map.get(priority, (255, 255, 255))


def calculate_severity_score_ipm(
    relative_area: float,
    circularity: float,
    area_weight: float = 0.6,
    irregularity_weight: float = 0.4,
    max_relative_area: float = 0.05
) -> float:
    """
    Calculate severity score using IPM-normalized relative area.
    
    This version uses the Bird's Eye View relative area for accurate,
    perspective-corrected measurements.
    
    Args:
        relative_area: Pothole area / Total ROI area (from IPM transformation)
        circularity: Circularity value (0-1) from calculate_circularity()
        area_weight: Weight for area component (default: 0.6)
        irregularity_weight: Weight for irregularity component (default: 0.4)
        max_relative_area: Maximum expected relative area (default: 0.05 = 5%)
        
    Returns:
        Severity score from 0 to 100
    """
    # Normalize relative area to 0-1 score
    area_score = min(relative_area / max_relative_area, 1.0)
    
    # Calculate irregularity score (inverse of circularity)
    irregularity_score = 1.0 - circularity
    
    # Combine scores with weights
    combined_score = (
        area_weight * area_score +
        irregularity_weight * irregularity_score
    )
    
    # Scale to 0-100 and clamp
    severity = combined_score * 100
    return min(max(severity, 0.0), 100.0)


def analyze_detection(
    contour: np.ndarray,
    frame_width: int,
    frame_height: int
) -> dict:
    """
    Perform complete analysis on a detected pothole contour.
    
    This is a convenience function that calculates all metrics at once.
    
    Args:
        contour: OpenCV contour array
        frame_width: Width of the video frame in pixels
        frame_height: Height of the video frame in pixels
        
    Returns:
        Dictionary containing:
        - 'area': Pixel area of the detection
        - 'perimeter': Perimeter in pixels
        - 'circularity': Shape irregularity index (0-1)
        - 'severity_score': Composite severity (0-100)
        - 'priority': PriorityLevel enum
        - 'priority_str': Priority as string ("LOW", "MEDIUM", "HIGH")
        - 'color_bgr': BGR color tuple for visualization
    """
    import cv2
    
    # Ensure contour has correct shape
    if contour.ndim == 3:
        contour_2d = contour.squeeze()
    else:
        contour_2d = contour
    
    if len(contour_2d) < 3:
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'circularity': 0.0,
            'severity_score': 0.0,
            'priority': PriorityLevel.LOW,
            'priority_str': 'LOW',
            'color_bgr': (0, 255, 0)
        }
    
    # Calculate basic metrics
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    frame_area = frame_width * frame_height
    
    # Calculate advanced metrics
    circularity = calculate_circularity(area, perimeter)
    severity_score = calculate_severity_score(area, frame_area, circularity)
    priority = get_priority_level(severity_score)
    
    return {
        'area': area,
        'perimeter': perimeter,
        'circularity': circularity,
        'severity_score': severity_score,
        'priority': priority,
        'priority_str': priority.value,
        'color_bgr': get_priority_color_bgr(priority)
    }


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("metrics.py - Unit Test")
    print("=" * 60)
    
    # Test 1: Circularity of a perfect circle
    # A circle with radius r has: Area = πr², Perimeter = 2πr
    # Circularity = (4π * πr²) / (2πr)² = (4π²r²) / (4π²r²) = 1.0
    r = 100
    circle_area = math.pi * r * r
    circle_perimeter = 2 * math.pi * r
    circ = calculate_circularity(circle_area, circle_perimeter)
    print(f"\n[Test 1] Perfect Circle (r={r}):")
    print(f"  Area: {circle_area:.2f}, Perimeter: {circle_perimeter:.2f}")
    print(f"  Circularity: {circ:.4f} (Expected: 1.0)")
    assert abs(circ - 1.0) < 0.001, "Circle circularity should be ~1.0"
    
    # Test 2: Square (less circular than a circle)
    # Square with side s: Area = s², Perimeter = 4s
    # Circularity = (4π * s²) / (4s)² = (4πs²) / (16s²) = π/4 ≈ 0.785
    s = 100
    square_area = s * s
    square_perimeter = 4 * s
    circ_sq = calculate_circularity(square_area, square_perimeter)
    print(f"\n[Test 2] Square (s={s}):")
    print(f"  Area: {square_area:.2f}, Perimeter: {square_perimeter:.2f}")
    print(f"  Circularity: {circ_sq:.4f} (Expected: ~0.785)")
    assert 0.78 < circ_sq < 0.79, "Square circularity should be ~0.785"
    
    # Test 3: Severity Score
    print(f"\n[Test 3] Severity Score:")
    frame_area = 1920 * 1080  # Full HD frame
    
    # Small circular pothole (low severity)
    small_area = 5000  # ~0.24% of frame
    high_circ = 0.9
    score_low = calculate_severity_score(small_area, frame_area, high_circ)
    print(f"  Small+Circular: Area={small_area}, Circ={high_circ}")
    print(f"  Severity: {score_low:.2f}, Priority: {get_priority_level(score_low).value}")
    
    # Large irregular pothole (high severity)
    large_area = 80000  # ~3.8% of frame
    low_circ = 0.3
    score_high = calculate_severity_score(large_area, frame_area, low_circ)
    print(f"  Large+Irregular: Area={large_area}, Circ={low_circ}")
    print(f"  Severity: {score_high:.2f}, Priority: {get_priority_level(score_high).value}")
    
    # Test 4: Priority Colors
    print(f"\n[Test 4] Priority Colors (BGR):")
    for level in PriorityLevel:
        color = get_priority_color_bgr(level)
        print(f"  {level.value}: {color}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

