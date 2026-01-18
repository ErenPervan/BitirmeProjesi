"""
geometry_utils.py - ROI & Inverse Perspective Mapping (IPM) Utilities

This module provides geometric transformation functions for:
1. Defining trapezoid ROI (Region of Interest) for road lane
2. Inverse Perspective Mapping (IPM) to Bird's Eye View
3. Mask transformation for accurate area measurement

Mathematical Foundation:
------------------------
IPM transforms the perspective view of the road into a top-down view,
eliminating perspective distortion for accurate area measurements.

The ROI is defined as a trapezoid:
- Top edge: Narrow (further from camera)
- Bottom edge: Wide (closer to camera)

Using cv2.getPerspectiveTransform to create the transformation matrix.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class ROIConfig:
    """
    Configuration for trapezoid ROI using relative coordinates.
    
    All values are percentages (0.0 to 1.0) of frame dimensions.
    
    Default values create a typical dashcam road view ROI:
    - Top edge at 55% height, spanning 30%-70% width
    - Bottom edge at 95% height, spanning 5%-95% width
    """
    # Top edge (far from camera - narrow)
    top_left_x: float = 0.30      # 30% from left
    top_right_x: float = 0.70     # 70% from left
    top_y: float = 0.55           # 55% from top
    
    # Bottom edge (close to camera - wide)
    bottom_left_x: float = 0.05   # 5% from left
    bottom_right_x: float = 0.95  # 95% from left
    bottom_y: float = 0.95        # 95% from top
    
    @classmethod
    def from_percentages(cls, top_width_pct: float, bottom_width_pct: float, horizon_pct: float, bottom_height_pct: float = 95.0, horizontal_offset_pct: float = 0.0) -> 'ROIConfig':
        """
        Create ROIConfig from simplified percentage-based sliders.
        
        Args:
            top_width_pct: Width of top edge as percentage (10-100), e.g., 40 = 40%
            bottom_width_pct: Width of bottom edge as percentage (10-100), e.g., 90 = 90%
            horizon_pct: Horizon line position as percentage (0-100), e.g., 60 = 60% from top
            bottom_height_pct: Bottom edge Y position as percentage (50-100), e.g., 90 = 90% from top
            horizontal_offset_pct: Horizontal shift for off-center camera (-20 to +20), e.g., 5 = 5% right shift
            
        Returns:
            ROIConfig with calculated coordinates
        """
        # Convert to 0-1 range
        top_w = top_width_pct / 100.0
        bottom_w = bottom_width_pct / 100.0
        horizon = horizon_pct / 100.0
        bottom_height = bottom_height_pct / 100.0
        h_offset = horizontal_offset_pct / 100.0  # Convert to 0-1 range
        
        # Calculate centered trapezoid coordinates
        top_margin = (1.0 - top_w) / 2.0
        bottom_margin = (1.0 - bottom_w) / 2.0
        
        # Apply horizontal offset (shift entire trapezoid left/right)
        return cls(
            top_left_x=max(0.0, min(1.0, top_margin + h_offset)),
            top_right_x=max(0.0, min(1.0, 1.0 - top_margin + h_offset)),
            top_y=horizon,
            bottom_left_x=max(0.0, min(1.0, bottom_margin + h_offset)),
            bottom_right_x=max(0.0, min(1.0, 1.0 - bottom_margin + h_offset)),
            bottom_y=bottom_height  # Adjustable bottom position
        )


@dataclass  
class IPMConfig:
    """
    Configuration for Bird's Eye View output dimensions.
    
    The output BEV image will have these dimensions.
    Using a fixed width and height ensures consistent area calculations.
    """
    output_width: int = 400       # BEV output width in pixels
    output_height: int = 600      # BEV output height in pixels


class GeometryProcessor:
    """
    Handles ROI definition and IPM transformations.
    
    This class provides:
    1. ROI polygon generation from relative coordinates
    2. Perspective transformation matrix computation
    3. Mask warping to Bird's Eye View
    4. Area calculations in normalized space
    
    Usage:
        geo = GeometryProcessor(frame_width=1920, frame_height=1080)
        
        # Get transformation matrix
        M = geo.perspective_matrix
        
        # Transform a mask to BEV
        bev_mask = geo.transform_mask_to_birdseye(mask)
        
        # Calculate relative area
        relative_area = geo.calculate_relative_area(pothole_mask)
    """
    
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        roi_config: Optional[ROIConfig] = None,
        ipm_config: Optional[IPMConfig] = None,
        exit_line_y_ratio: float = 0.85
    ):
        """
        Initialize the GeometryProcessor.
        
        Args:
            frame_width: Width of video frame in pixels
            frame_height: Height of video frame in pixels
            roi_config: ROI configuration (uses defaults if None)
            ipm_config: IPM configuration (uses defaults if None)
            exit_line_y_ratio: Exit line position as percentage (0.5-0.99, default 0.85)
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi_config = roi_config or ROIConfig()
        self.ipm_config = ipm_config or IPMConfig()
        
        # Calculate ROI polygon points (source trapezoid)
        self.roi_polygon = self._calculate_roi_polygon()
        
        # Calculate destination rectangle for BEV
        self.bev_rect = self._calculate_bev_rectangle()
        
        # Compute perspective transformation matrices
        self.perspective_matrix = self._compute_perspective_matrix()
        self.inverse_matrix = self._compute_inverse_matrix()
        
        # Calculate total ROI area in BEV space (for normalization)
        self.total_bev_roi_area = self._calculate_total_bev_area()
        
        # Exit line configuration (y position as percentage of frame height)
        self.exit_line_y_ratio = exit_line_y_ratio
        self.exit_line_y = int(frame_height * self.exit_line_y_ratio)
    
    def _calculate_roi_polygon(self) -> np.ndarray:
        """
        Calculate ROI polygon points from relative coordinates.
        
        Returns:
            np.ndarray: 4x2 array of (x, y) points in order:
                        [top_left, top_right, bottom_right, bottom_left]
        """
        cfg = self.roi_config
        w, h = self.frame_width, self.frame_height
        
        # Calculate absolute pixel coordinates
        top_left = (int(w * cfg.top_left_x), int(h * cfg.top_y))
        top_right = (int(w * cfg.top_right_x), int(h * cfg.top_y))
        bottom_right = (int(w * cfg.bottom_right_x), int(h * cfg.bottom_y))
        bottom_left = (int(w * cfg.bottom_left_x), int(h * cfg.bottom_y))
        
        return np.array([
            top_left,
            top_right,
            bottom_right,
            bottom_left
        ], dtype=np.float32)
    
    def _calculate_bev_rectangle(self) -> np.ndarray:
        """
        Calculate destination rectangle for Bird's Eye View.
        
        Returns:
            np.ndarray: 4x2 array of destination points forming a rectangle
        """
        w, h = self.ipm_config.output_width, self.ipm_config.output_height
        
        # Map trapezoid to rectangle
        return np.array([
            [0, 0],           # top_left
            [w, 0],           # top_right
            [w, h],           # bottom_right
            [0, h]            # bottom_left
        ], dtype=np.float32)
    
    def _compute_perspective_matrix(self) -> np.ndarray:
        """
        Compute perspective transformation matrix (Frame -> BEV).
        
        Returns:
            3x3 perspective transformation matrix
        """
        return cv2.getPerspectiveTransform(self.roi_polygon, self.bev_rect)
    
    def _compute_inverse_matrix(self) -> np.ndarray:
        """
        Compute inverse perspective transformation matrix (BEV -> Frame).
        
        Returns:
            3x3 inverse perspective transformation matrix
        """
        return cv2.getPerspectiveTransform(self.bev_rect, self.roi_polygon)
    
    def _calculate_total_bev_area(self) -> float:
        """
        Calculate total area of ROI in Bird's Eye View space.
        
        This is simply width * height of the BEV output.
        
        Returns:
            Total ROI area in BEV pixels
        """
        return float(self.ipm_config.output_width * self.ipm_config.output_height)
    
    def transform_mask_to_birdseye(
        self,
        mask: np.ndarray,
        interpolation: int = cv2.INTER_NEAREST
    ) -> np.ndarray:
        """
        Transform a binary mask to Bird's Eye View using IPM.
        
        Uses cv2.INTER_NEAREST to preserve binary nature and area accuracy.
        
        Args:
            mask: Binary mask (H x W) where 255 = pothole, 0 = background
            interpolation: Interpolation method (default: INTER_NEAREST)
            
        Returns:
            Transformed mask in BEV space (ipm_height x ipm_width)
        """
        if mask is None:
            return None
        
        bev_size = (self.ipm_config.output_width, self.ipm_config.output_height)
        
        bev_mask = cv2.warpPerspective(
            mask,
            self.perspective_matrix,
            bev_size,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return bev_mask
    
    def transform_point_to_birdseye(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point from frame coordinates to BEV coordinates.
        
        Args:
            point: (x, y) coordinates in frame space
            
        Returns:
            (x, y) coordinates in BEV space
        """
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.perspective_matrix)
        return (transformed[0, 0, 0], transformed[0, 0, 1])
    
    def transform_point_to_frame(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """
        Transform a single point from BEV coordinates to frame coordinates.
        
        Args:
            point: (x, y) coordinates in BEV space
            
        Returns:
            (x, y) coordinates in frame space
        """
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.inverse_matrix)
        return (transformed[0, 0, 0], transformed[0, 0, 1])
    
    def create_roi_mask(self) -> np.ndarray:
        """
        Create a binary mask for the ROI region.
        
        Returns:
            Binary mask (frame_height x frame_width) where ROI = 255
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        pts = self.roi_polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def mask_to_roi_only(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply ROI mask to keep only detections within ROI.
        
        Args:
            mask: Input binary mask
            
        Returns:
            Mask with only ROI region preserved
        """
        roi_mask = self.create_roi_mask()
        return cv2.bitwise_and(mask, roi_mask)
    
    def calculate_bev_area(self, bev_mask: np.ndarray) -> float:
        """
        Calculate area of pothole in Bird's Eye View mask.
        
        Args:
            bev_mask: Binary mask in BEV space
            
        Returns:
            Pixel area of pothole in BEV space
        """
        if bev_mask is None:
            return 0.0
        return float(np.sum(bev_mask > 0))
    
    def calculate_relative_area(self, bev_mask: np.ndarray) -> float:
        """
        Calculate relative area (pothole area / total ROI area).
        
        This provides a normalized, perspective-corrected area measurement.
        
        Args:
            bev_mask: Binary mask in BEV space
            
        Returns:
            Relative area as a ratio (0.0 to 1.0)
        """
        bev_area = self.calculate_bev_area(bev_mask)
        if self.total_bev_roi_area <= 0:
            return 0.0
        return bev_area / self.total_bev_roi_area
    
    def is_point_in_roi(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside the ROI trapezoid.
        
        Uses cv2.pointPolygonTest for accurate containment check.
        
        Args:
            point: (x, y) coordinates in frame space
            
        Returns:
            True if the point is inside or on the ROI boundary
        """
        result = cv2.pointPolygonTest(
            self.roi_polygon.astype(np.int32),
            point,
            measureDist=False
        )
        # result >= 0 means inside or on the boundary
        return result >= 0
    
    def is_past_exit_line(self, center_y: float) -> bool:
        """
        Check if a point has crossed the exit line.
        
        Args:
            center_y: Y coordinate of pothole center in frame space
            
        Returns:
            True if the point is past (below) the exit line
        """
        return center_y >= self.exit_line_y
    
    def draw_roi_overlay(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 255), alpha: float = 0.2) -> np.ndarray:
        """
        Draw ROI trapezoid overlay on frame for visualization.
        
        Args:
            frame: BGR image
            color: BGR color for ROI overlay
            alpha: Transparency (0.0 = invisible, 1.0 = solid)
            
        Returns:
            Frame with ROI overlay
        """
        overlay = frame.copy()
        pts = self.roi_polygon.astype(np.int32).reshape((-1, 1, 2))
        
        # Fill ROI area
        cv2.fillPoly(overlay, [pts], color)
        
        # Blend with original
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw ROI border
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
        
        return frame
    
    def draw_exit_line(self, frame: np.ndarray, color: Tuple[int, int, int] = (255, 0, 255), thickness: int = 2) -> np.ndarray:
        """
        Draw the exit line on frame for visualization.
        
        Args:
            frame: BGR image
            color: BGR color for exit line
            thickness: Line thickness in pixels
            
        Returns:
            Frame with exit line drawn
        """
        y = self.exit_line_y
        cv2.line(frame, (0, y), (self.frame_width, y), color, thickness)
        
        # Add label
        cv2.putText(
            frame, "EXIT LINE",
            (10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, color, 1
        )
        
        return frame
    
    def create_debug_view(
        self,
        frame: np.ndarray,
        bev_frame: Optional[np.ndarray] = None,
        show_roi: bool = True,
        show_exit_line: bool = True
    ) -> np.ndarray:
        """
        Create a side-by-side debug view (Original + Bird's Eye View).
        
        Args:
            frame: Original BGR frame
            bev_frame: Optional pre-computed BEV frame. If None, will compute.
            show_roi: Whether to draw ROI overlay on original
            show_exit_line: Whether to draw exit line on original
            
        Returns:
            Combined debug view image (wider than original)
        """
        # Annotate original frame
        annotated = frame.copy()
        if show_roi:
            annotated = self.draw_roi_overlay(annotated)
        if show_exit_line:
            annotated = self.draw_exit_line(annotated)
        
        # Compute BEV if not provided
        if bev_frame is None:
            bev_frame = cv2.warpPerspective(
                frame,
                self.perspective_matrix,
                (self.ipm_config.output_width, self.ipm_config.output_height)
            )
        
        # Resize BEV to match original height for side-by-side
        scale = self.frame_height / bev_frame.shape[0]
        bev_resized = cv2.resize(
            bev_frame,
            (int(bev_frame.shape[1] * scale), self.frame_height)
        )
        
        # Add labels
        cv2.putText(annotated, "ORIGINAL VIEW", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(bev_resized, "BIRD'S EYE VIEW (IPM)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Combine side-by-side
        debug_view = np.hstack([annotated, bev_resized])
        
        return debug_view
    
    def polygon_to_mask(
        self,
        polygon: np.ndarray,
        frame_shape: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Convert a polygon to a binary mask.
        
        Args:
            polygon: Nx2 array of polygon points
            frame_shape: (height, width) of output mask. Uses frame dimensions if None.
            
        Returns:
            Binary mask with polygon filled
        """
        if frame_shape is None:
            frame_shape = (self.frame_height, self.frame_width)
        
        mask = np.zeros(frame_shape, dtype=np.uint8)
        
        if polygon is None or len(polygon) < 3:
            return mask
        
        pts = polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        
        return mask
    
    def get_polygon_center(self, polygon: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the center (centroid) of a polygon.
        
        Args:
            polygon: Nx2 array of polygon points
            
        Returns:
            (cx, cy) center coordinates
        """
        if polygon is None or len(polygon) < 3:
            return (0.0, 0.0)
        
        M = cv2.moments(polygon.astype(np.int32))
        
        if M["m00"] == 0:
            # Fallback to mean
            return (float(np.mean(polygon[:, 0])), float(np.mean(polygon[:, 1])))
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        return (cx, cy)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_default_geometry(frame_width: int, frame_height: int) -> GeometryProcessor:
    """
    Create a GeometryProcessor with default ROI and IPM settings.
    
    Args:
        frame_width: Video frame width
        frame_height: Video frame height
        
    Returns:
        Configured GeometryProcessor instance
    """
    return GeometryProcessor(frame_width, frame_height)


def transform_mask_to_birdseye(
    mask: np.ndarray,
    matrix: np.ndarray,
    output_size: Tuple[int, int] = (400, 600)
) -> np.ndarray:
    """
    Standalone function to transform a mask to Bird's Eye View.
    
    Uses cv2.INTER_NEAREST to preserve area accuracy.
    
    Args:
        mask: Binary mask (H x W)
        matrix: 3x3 perspective transformation matrix
        output_size: (width, height) of output BEV image
        
    Returns:
        Transformed mask in BEV space
    """
    if mask is None:
        return None
    
    return cv2.warpPerspective(
        mask,
        matrix,
        output_size,
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )


def create_calibration_geometry(
    frame_width: int, 
    frame_height: int,
    top_width_pct: float = 40.0,
    bottom_width_pct: float = 90.0,
    horizon_pct: float = 60.0,
    exit_line_y_ratio: float = 0.85,
    bottom_height_pct: float = 95.0,
    horizontal_offset_pct: float = 0.0
) -> GeometryProcessor:
    """
    Create a GeometryProcessor with dynamic ROI settings for calibration.
    
    Args:
        frame_width: Video frame width
        frame_height: Video frame height
        top_width_pct: Width of top edge as percentage (10-100)
        bottom_width_pct: Width of bottom edge as percentage (10-100)
        horizon_pct: Horizon line position as percentage (0-100)
        exit_line_y_ratio: Exit line position as percentage ratio (0.5-0.99, default 0.85)
        bottom_height_pct: Bottom edge Y position as percentage (50-100, default 95)
        horizontal_offset_pct: Horizontal shift for off-center camera (-20 to +20)
        
    Returns:
        Configured GeometryProcessor instance with dynamic ROI
    """
    roi_config = ROIConfig.from_percentages(top_width_pct, bottom_width_pct, horizon_pct, bottom_height_pct, horizontal_offset_pct)
    return GeometryProcessor(frame_width, frame_height, roi_config=roi_config, exit_line_y_ratio=exit_line_y_ratio)


def get_calibration_debug_frame(
    frame: np.ndarray,
    top_width_pct: float = 40.0,
    bottom_width_pct: float = 90.0,
    horizon_pct: float = 60.0,
    exit_line_y_ratio: float = 85.0,
    bottom_height_pct: float = 90.0,
    horizontal_offset_pct: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate calibration debug frames with dynamic ROI.
    
    This function is used by the Streamlit calibration mode to show
    the user how the ROI and Bird's Eye View look with their settings.
    
    Args:
        frame: BGR input frame
        top_width_pct: Width of top edge as percentage (10-100)
        bottom_width_pct: Width of bottom edge as percentage (10-100)
        horizon_pct: Horizon line position as percentage (0-100)
        exit_line_y_ratio: Exit line position as percentage (50-99, default 85)
        bottom_height_pct: Bottom edge Y position as percentage (50-100, default 90)
        horizontal_offset_pct: Horizontal shift for off-center camera (-20 to +20)
        
    Returns:
        Tuple of (original_with_roi, birds_eye_view)
    """
    h, w = frame.shape[:2]
    
    # Create geometry processor with dynamic settings
    geo = create_calibration_geometry(w, h, top_width_pct, bottom_width_pct, horizon_pct, exit_line_y_ratio / 100.0, bottom_height_pct, horizontal_offset_pct)
    
    # Draw ROI on original frame (Blue color for calibration)
    original_view = frame.copy()
    pts = geo.roi_polygon.astype(np.int32).reshape((-1, 1, 2))
    
    # Semi-transparent fill
    overlay = original_view.copy()
    cv2.fillPoly(overlay, [pts], (255, 100, 0))  # Blue-ish
    original_view = cv2.addWeighted(overlay, 0.3, original_view, 0.7, 0)
    
    # Draw border (thick blue line)
    cv2.polylines(original_view, [pts], isClosed=True, color=(255, 0, 0), thickness=3)
    
    # Draw corner points
    for i, pt in enumerate(geo.roi_polygon):
        x, y = int(pt[0]), int(pt[1])
        cv2.circle(original_view, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(original_view, (x, y), 8, (0, 0, 0), 2)
    
    # Draw exit line (RED for visibility in calibration)
    original_view = geo.draw_exit_line(original_view, color=(0, 0, 255), thickness=3)
    
    # Add labels
    cv2.putText(original_view, "Orijinal Goruntu + ROI", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(original_view, f"Ust: {top_width_pct:.0f}% | Alt: {bottom_width_pct:.0f}% | Ufuk: {horizon_pct:.0f}%", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(original_view, f"Cikis Cizgisi: {exit_line_y_ratio:.0f}% (KIRMIZI)", 
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Create Bird's Eye View
    bev_size = (geo.ipm_config.output_width, geo.ipm_config.output_height)
    bev_frame = cv2.warpPerspective(frame, geo.perspective_matrix, bev_size)
    
    # Draw grid on BEV for visualization
    bev_h, bev_w = bev_frame.shape[:2]
    grid_color = (0, 255, 0)
    
    # Vertical grid lines
    for x in range(0, bev_w, bev_w // 4):
        cv2.line(bev_frame, (x, 0), (x, bev_h), grid_color, 1)
    
    # Horizontal grid lines
    for y in range(0, bev_h, bev_h // 6):
        cv2.line(bev_frame, (0, y), (bev_w, y), grid_color, 1)
    
    # Add BEV label
    cv2.putText(bev_frame, "Kus Bakisi (BEV)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return original_view, bev_frame


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("geometry_utils.py - Module Test")
    print("=" * 60)
    
    # Test with standard 1920x1080 frame
    frame_width, frame_height = 1920, 1080
    
    print(f"\n[Test 1] Initialize GeometryProcessor ({frame_width}x{frame_height})")
    geo = GeometryProcessor(frame_width, frame_height)
    
    print(f"  ROI Polygon Points:")
    for i, pt in enumerate(geo.roi_polygon):
        print(f"    Point {i}: ({pt[0]:.0f}, {pt[1]:.0f})")
    
    print(f"\n  BEV Output Size: {geo.ipm_config.output_width}x{geo.ipm_config.output_height}")
    print(f"  Total BEV ROI Area: {geo.total_bev_roi_area:.0f} pixels")
    print(f"  Exit Line Y: {geo.exit_line_y} ({geo.exit_line_y_ratio*100:.0f}% of height)")
    
    # Test perspective matrix
    print(f"\n[Test 2] Perspective Transform Matrix:")
    print(f"  Shape: {geo.perspective_matrix.shape}")
    print(f"  Matrix:\n{geo.perspective_matrix}")
    
    # Test point transformation
    print(f"\n[Test 3] Point Transformation:")
    test_point = (960, 900)  # Center-bottom of frame
    bev_point = geo.transform_point_to_birdseye(test_point)
    back_point = geo.transform_point_to_frame(bev_point)
    print(f"  Frame Point: {test_point}")
    print(f"  -> BEV Point: ({bev_point[0]:.1f}, {bev_point[1]:.1f})")
    print(f"  -> Back to Frame: ({back_point[0]:.1f}, {back_point[1]:.1f})")
    
    # Test exit line detection
    print(f"\n[Test 4] Exit Line Detection:")
    test_y_values = [500, 800, 918, 950]
    for y in test_y_values:
        crossed = geo.is_past_exit_line(y)
        print(f"  y={y}: {'PAST EXIT LINE' if crossed else 'before exit line'}")
    
    # Test mask transformation
    print(f"\n[Test 5] Mask Transformation:")
    # Create a test polygon (simulated pothole)
    test_polygon = np.array([[800, 700], [1120, 700], [1100, 800], [820, 800]])
    mask = geo.polygon_to_mask(test_polygon)
    original_area = np.sum(mask > 0)
    print(f"  Test polygon area in frame: {original_area} pixels")
    
    bev_mask = geo.transform_mask_to_birdseye(mask)
    bev_area = geo.calculate_bev_area(bev_mask)
    relative_area = geo.calculate_relative_area(bev_mask)
    print(f"  BEV mask area: {bev_area:.0f} pixels")
    print(f"  Relative area: {relative_area:.6f} ({relative_area*100:.4f}%)")
    
    # Test polygon center
    print(f"\n[Test 6] Polygon Center:")
    center = geo.get_polygon_center(test_polygon)
    print(f"  Center of test polygon: ({center[0]:.1f}, {center[1]:.1f})")
    crossed = geo.is_past_exit_line(center[1])
    print(f"  Past exit line: {crossed}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
