"""
video_processor.py - Core Video Processing Pipeline (IPM Enhanced)

This module orchestrates the entire pothole detection workflow:
1. Video input/output handling
2. Frame-by-frame processing with tracking
3. IPM (Inverse Perspective Mapping) for accurate area measurement
4. Exit Line based commit logic for optimal detection capture
5. Metrics calculation and visualization
6. Data aggregation for final report

Architecture: Headless (NO cv2.imshow) - Batch processing only.

Key Features (Graduation Project Architecture):
- ROI trapezoid for road lane definition
- Bird's Eye View transformation for area normalization
- Exit Line at 85% frame height for commit trigger
- Best Frame capture and snapshot saving
- Risk-based color coding (Green/Yellow/Red)
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from tqdm import tqdm
import logging

from .detector import PotholeDetector
from .metrics import (
    calculate_circularity,
    calculate_severity_score,
    calculate_severity_score_ipm,
    get_priority_level,
    get_priority_color_bgr,
    get_risk_label,
    PriorityLevel
)

logger = logging.getLogger(__name__)
from .gps_manager import GPSManager
from .database_manager import DatabaseManager
from .geometry_utils import GeometryProcessor
from .depth_utils import DepthValidator


@dataclass
class TrackData:
    """Accumulated data for a single tracked object across frames.
    
    Enhanced with IPM-based metrics and exit line logic.
    """
    track_id: int
    frame_appearances: int = 0
    severity_scores: List[float] = field(default_factory=list)
    circularity_values: List[float] = field(default_factory=list)
    areas: List[float] = field(default_factory=list)  # Raw pixel areas
    relative_areas: List[float] = field(default_factory=list)  # IPM normalized areas
    
    # GPS data (can be None if GPS unavailable)
    last_latitude: Optional[float] = None
    last_longitude: Optional[float] = None
    
    # Exit line tracking - only commit when crossed
    has_crossed_exit: bool = False
    committed_to_db: bool = False
    
    # Best frame capture (at exit line crossing)
    best_frame: Optional[np.ndarray] = field(default=None, repr=False)
    best_bbox: Optional[Tuple[float, float, float, float]] = None
    best_mask_polygon: Optional[np.ndarray] = field(default=None, repr=False)
    best_severity: float = 0.0
    best_relative_area: float = 0.0
    best_circularity: float = 0.0
    
    # Center point for exit line check
    last_center_y: float = 0.0
    
    @property
    def avg_severity(self) -> float:
        """Average severity score across all appearances."""
        return np.mean(self.severity_scores) if self.severity_scores else 0.0
    
    @property
    def avg_circularity(self) -> float:
        """Average circularity (irregularity index) across all appearances."""
        return np.mean(self.circularity_values) if self.circularity_values else 0.0
    
    @property
    def avg_relative_area(self) -> float:
        """Average IPM-normalized relative area."""
        return np.mean(self.relative_areas) if self.relative_areas else 0.0
    
    @property
    def max_area(self) -> float:
        """Maximum area detected for this object."""
        return max(self.areas) if self.areas else 0.0
    
    @property
    def max_relative_area(self) -> float:
        """Maximum IPM-normalized relative area."""
        return max(self.relative_areas) if self.relative_areas else 0.0
    
    @property
    def priority_level(self) -> str:
        """Priority level based on average severity."""
        return get_priority_level(self.avg_severity).value
    
    @property
    def risk_label(self) -> str:
        """Human-readable risk label (Low/Medium/High)."""
        return get_risk_label(self.avg_severity)


@dataclass
class FrameResult:
    """Result data yielded for each processed frame (for Streamlit UI)."""
    frame_idx: int
    frame_rgb: np.ndarray  # RGB image for display
    total_frames: int
    progress_percent: float
    current_detections: int
    total_detections: int
    unique_tracks: int
    avg_severity: float
    high_count: int
    medium_count: int
    low_count: int


@dataclass
class ProcessingStatus:
    """
    Lightweight status update for optimized Streamlit processing.
    No image data - only progress and statistics for maximum performance.
    """
    frame_idx: int
    total_frames: int
    progress_percent: float
    total_detections: int
    unique_tracks: int
    status_message: str


@dataclass 
class ProcessingResult:
    """Final result after processing completes."""
    track_data: Dict
    stats: Dict
    output_video_path: str
    processing_time_seconds: float
    average_fps: float
    total_frames: int


class VideoProcessingError(Exception):
    """Raised when video processing fails."""
    pass


class VideoProcessor:
    """
    Main video processing pipeline for pothole detection (IPM Enhanced).
    
    This class handles:
    - Video file I/O (read input, write annotated output)
    - Frame-by-frame detection and tracking
    - IPM transformation for accurate area measurement
    - Exit Line based commit logic
    - Metrics calculation (circularity, severity)
    - Visualization (mask overlay, bounding boxes, labels)
    - GPS integration (optional - strict mode, no fake data)
    - Database logging (SQLite with nullable GPS fields)
    - Best frame snapshot capture
    - Data aggregation for CSV report generation
    
    Usage:
        detector = PotholeDetector("best.engine")
        processor = VideoProcessor(
            input_path="input.mp4",
            output_path="output/annotated.mp4",
            detector=detector,
            gps_file_path="gps_data.csv",  # Optional
            db_path="detections.db"
        )
        track_data = processor.process()
        # track_data contains aggregated metrics per object ID
    """
    
    # Visualization constants
    MASK_ALPHA = 0.4  # Transparency for mask overlay
    TEXT_SCALE = 0.6
    TEXT_THICKNESS = 2
    BOX_THICKNESS = 2
    GPS_TEXT_SCALE = 0.5
    GPS_TEXT_COLOR = (255, 255, 255)  # White
    GPS_BG_COLOR = (50, 50, 50)  # Dark gray background
    
    # IPM and Exit Line settings
    SHOW_DEBUG_VIEW = False  # Set True to enable side-by-side debug view
    SHOW_ROI_OVERLAY = True  # Draw ROI trapezoid on output
    SHOW_EXIT_LINE = True    # Draw exit line on output
    
    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        detector: PotholeDetector,
        gps_file_path: Optional[Union[str, Path]] = None,
        db_path: Optional[Union[str, Path]] = None,
        codec: str = "mp4v",
        enable_debug_view: bool = False,
        roi_top_width: float = 40.0,
        roi_bottom_width: float = 90.0,
        roi_horizon: float = 60.0,
        roi_bottom_height: float = 90.0,
        roi_horizontal_offset: float = 0.0,
        exit_line_y_ratio: float = 85.0
    ):
        """
        Initialize the VideoProcessor.
        
        Args:
            input_path: Path to input video file
            output_path: Path for output annotated video
            detector: Initialized PotholeDetector instance
            gps_file_path: Optional path to GPS data file (CSV/JSON)
                          If None, GPS overlay is disabled (strict mode)
            db_path: Optional path for SQLite database
                    If None, database logging is disabled
            codec: FourCC codec for output video (default: mp4v)
            enable_debug_view: Enable side-by-side Original/BEV debug view
            roi_top_width: Width of ROI top edge as percentage (10-100)
            roi_bottom_width: Width of ROI bottom edge as percentage (10-100)
            roi_horizon: Horizon line position as percentage (0-100)
            roi_bottom_height: Bottom edge Y position as percentage (50-100, default 90)
            roi_horizontal_offset: Horizontal shift for off-center camera (-20 to +20%)
            exit_line_y_ratio: Exit line position as percentage (50-99, default 85)
            
        Raises:
            VideoProcessingError: If input video cannot be opened
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.detector = detector
        self.codec = codec
        self.SHOW_DEBUG_VIEW = enable_debug_view
        
        # Store ROI calibration parameters
        self.roi_top_width = roi_top_width
        self.roi_bottom_width = roi_bottom_width
        self.roi_horizontal_offset = roi_horizontal_offset
        self.roi_horizon = roi_horizon
        self.roi_bottom_height = roi_bottom_height
        self.exit_line_y_ratio = exit_line_y_ratio
        
        # Initialize GPS Manager (strict mode - no fake data)
        self.gps_manager = GPSManager(
            gps_file_path=str(gps_file_path) if gps_file_path else None
        )
        
        # Initialize Database Manager (optional)
        self.db_manager: Optional[DatabaseManager] = None
        if db_path is not None:
            self.db_manager = DatabaseManager(str(db_path))
        
        # Initialize Depth Validator for topographic verification
        self.depth_validator = DepthValidator()
        if self.depth_validator.enabled:
            print("[VideoProcessor] Depth validation enabled (Depth Anything V2)")
        else:
            print("[VideoProcessor] Depth validation disabled (model not available)")
        
        # Video properties (populated on open)
        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_width: int = 0
        self.frame_height: int = 0
        self.fps: float = 0.0
        self.total_frames: int = 0
        
        # Geometry Processor for IPM (initialized after video open)
        self.geometry: Optional[GeometryProcessor] = None
        
        # Active tracks for exit line logic
        self.track_data: Dict[int, TrackData] = {}
        
        # Previous frame track IDs (for lost track detection)
        self.previous_frame_track_ids: set = set()
        
        # Statistics
        self.frames_processed: int = 0
        self.total_detections: int = 0
        self.detections_with_gps: int = 0
        self.tracks_committed: int = 0
        self.tracks_saved_via_proximity: int = 0  # New stat for proximity saves
    
    def _open_video(self) -> None:
        """
        Open input video and initialize output writer.
        
        Raises:
            VideoProcessingError: If video cannot be opened or properties read
        """
        # Validate input file exists
        if not self.input_path.exists():
            raise VideoProcessingError(
                f"Input video not found: {self.input_path.absolute()}"
            )
        
        # Open input video
        self.cap = cv2.VideoCapture(str(self.input_path))
        
        if not self.cap.isOpened():
            raise VideoProcessingError(
                f"Failed to open input video: {self.input_path}"
            )
        
        # Read video properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate properties
        if self.frame_width == 0 or self.frame_height == 0:
            raise VideoProcessingError("Invalid video dimensions")
        
        if self.fps <= 0:
            print("[Warning] Could not read FPS, defaulting to 30.0")
            self.fps = 30.0
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (self.frame_width, self.frame_height)
        )
        
        if not self.writer.isOpened():
            raise VideoProcessingError(
                f"Failed to create output video: {self.output_path}\n"
                f"Codec '{self.codec}' may not be supported."
            )
        
        # Initialize GeometryProcessor for IPM transformations with dynamic ROI
        from .geometry_utils import ROIConfig
        roi_config = ROIConfig.from_percentages(
            self.roi_top_width,
            self.roi_bottom_width,
            self.roi_horizon,
            self.roi_bottom_height,
            self.roi_horizontal_offset
        )
        self.geometry = GeometryProcessor(
            self.frame_width, 
            self.frame_height,
            roi_config=roi_config,
            exit_line_y_ratio=self.exit_line_y_ratio / 100.0
        )
        
        print(f"\n[VideoProcessor] Input: {self.input_path.name}")
        print(f"[VideoProcessor] Output: {self.output_path}")
        print(f"[VideoProcessor] Resolution: {self.frame_width}x{self.frame_height}")
        print(f"[VideoProcessor] FPS: {self.fps:.2f} | Frames: {self.total_frames}")
        print(f"[VideoProcessor] ROI: Ust={self.roi_top_width}%, Alt={self.roi_bottom_width}%, Ufuk={self.roi_horizon}%")
        print(f"[VideoProcessor] IPM Enabled: Exit Line at y={self.geometry.exit_line_y}")
        print(f"[VideoProcessor] BEV Size: {self.geometry.ipm_config.output_width}x{self.geometry.ipm_config.output_height}")
    
    def _close_video(self) -> None:
        """Release video resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.writer is not None:
            self.writer.release()
            self.writer = None
    
    def _draw_mask_overlay(
        self,
        frame: np.ndarray,
        mask_polygon: np.ndarray,
        color_bgr: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw a semi-transparent polygon mask overlay on the frame.
        
        Args:
            frame: BGR image to draw on
            mask_polygon: Array of polygon points (N, 2)
            color_bgr: BGR color tuple
            
        Returns:
            Frame with mask overlay applied
        """
        if mask_polygon is None or len(mask_polygon) < 3:
            return frame
        
        # Create overlay for transparency
        overlay = frame.copy()
        
        # Convert polygon points to integer format for cv2
        pts = mask_polygon.astype(np.int32).reshape((-1, 1, 2))
        
        # Fill the polygon on overlay
        cv2.fillPoly(overlay, [pts], color_bgr)
        
        # Blend overlay with original frame
        frame = cv2.addWeighted(overlay, self.MASK_ALPHA, frame, 1 - self.MASK_ALPHA, 0)
        
        # Draw polygon contour (solid line)
        cv2.polylines(frame, [pts], isClosed=True, color=color_bgr, thickness=2)
        
        return frame
    
    def _draw_bounding_box(
        self,
        frame: np.ndarray,
        bbox: List[float],
        color_bgr: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw bounding box on frame.
        
        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2] coordinates
            color_bgr: BGR color tuple
            
        Returns:
            Frame with bounding box
        """
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, self.BOX_THICKNESS)
        return frame
    
    def _draw_label(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        color_bgr: Tuple[int, int, int]
    ) -> np.ndarray:
        """
        Draw text label with background on frame.
        
        Args:
            frame: BGR image
            text: Label text to display
            position: (x, y) position for text
            color_bgr: BGR color for text and background
            
        Returns:
            Frame with label
        """
        x, y = position
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, self.TEXT_SCALE, self.TEXT_THICKNESS
        )
        
        # Draw background rectangle
        padding = 4
        cv2.rectangle(
            frame,
            (x, y - text_height - padding * 2),
            (x + text_width + padding * 2, y),
            color_bgr,
            -1  # Filled
        )
        
        # Determine text color (white or black for contrast)
        brightness = (color_bgr[0] + color_bgr[1] + color_bgr[2]) / 3
        text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
        
        # Draw text
        cv2.putText(
            frame,
            text,
            (x + padding, y - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.TEXT_SCALE,
            text_color,
            self.TEXT_THICKNESS
        )
        
        return frame
    
    def _draw_gps_overlay(
        self,
        frame: np.ndarray,
        latitude: float,
        longitude: float,
        position: Tuple[int, int]
    ) -> np.ndarray:
        """
        Draw GPS coordinates overlay on frame.
        
        ONLY called when GPS data is available (lat is not None).
        
        Args:
            frame: BGR image
            latitude: GPS latitude
            longitude: GPS longitude
            position: (x, y) position for text
            
        Returns:
            Frame with GPS overlay
        """
        gps_text = f"Lat: {latitude:.6f} | Lon: {longitude:.6f}"
        x, y = position
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(
            gps_text, cv2.FONT_HERSHEY_SIMPLEX, self.GPS_TEXT_SCALE, 1
        )
        
        # Draw semi-transparent background
        padding = 3
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (x - padding, y - text_height - padding),
            (x + text_width + padding, y + padding),
            self.GPS_BG_COLOR,
            -1
        )
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw GPS text
        cv2.putText(
            frame,
            gps_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.GPS_TEXT_SCALE,
            self.GPS_TEXT_COLOR,
            1
        )
        
        return frame
    
    def _process_detection(
        self,
        frame: np.ndarray,
        detection: dict,
        frame_area: int,
        frame_idx: int,
        timestamp: str
    ) -> np.ndarray:
        """
        Process a single detection with IPM-based metrics and Exit Line logic.
        
        Key Workflow (Graduation Project Architecture):
        1. Calculate polygon center to check Exit Line crossing
        2. Transform mask to Bird's Eye View for accurate area
        3. Calculate metrics using IPM-normalized area
        4. Store accumulated data in TrackData
        5. ONLY commit to DB when crossing Exit Line (capture best frame)
        6. Visualize with risk-based color coding
        
        Args:
            frame: BGR image to annotate
            detection: Detection dict from detector.extract_detections()
            frame_area: Total frame area in pixels
            frame_idx: Current frame index (for GPS lookup)
            timestamp: ISO timestamp string for DB logging
            
        Returns:
            Annotated frame
        """
        track_id = detection['track_id']
        bbox = detection['bbox']
        confidence = detection['confidence']
        mask_polygon = detection['mask_polygon']
        
        # Skip if no valid mask or no track ID
        if mask_polygon is None or len(mask_polygon) < 3:
            return frame
        if track_id is None:
            return frame
        
        # === IPM CALCULATIONS ===
        # 1. Get polygon center for exit line check
        center_x, center_y = self.geometry.get_polygon_center(mask_polygon)
        
        # === ROI CHECK - CRITICAL: Only process detections inside ROI ===
        if not self.geometry.is_point_in_roi((center_x, center_y)):
            # Detection is outside ROI - skip processing entirely
            return frame
        
        # 2. Convert polygon to mask and transform to BEV
        polygon_mask = self.geometry.polygon_to_mask(mask_polygon)
        bev_mask = self.geometry.transform_mask_to_birdseye(polygon_mask)
        
        # 3. Calculate IPM-based metrics
        bev_area = self.geometry.calculate_bev_area(bev_mask)
        relative_area = self.geometry.calculate_relative_area(bev_mask)
        
        # 4. Calculate raw metrics (for comparison/fallback)
        mask_pts = mask_polygon.astype(np.int32)
        raw_area = cv2.contourArea(mask_pts)
        perimeter = cv2.arcLength(mask_pts, closed=True)
        circularity = calculate_circularity(raw_area, perimeter)
        
        # 5. Calculate severity using IPM-normalized area
        severity_score = calculate_severity_score_ipm(relative_area, circularity)
        priority = get_priority_level(severity_score)
        color_bgr = get_priority_color_bgr(priority)
        risk_label = get_risk_label(severity_score)
        
        # === GPS LOOKUP (STRICT MODE) ===
        lat, lon = self.gps_manager.get_location(frame_idx)
        
        # === TRACK DATA ACCUMULATION ===
        if track_id not in self.track_data:
            self.track_data[track_id] = TrackData(track_id=track_id)
        
        track = self.track_data[track_id]
        track.frame_appearances += 1
        track.severity_scores.append(severity_score)
        track.circularity_values.append(circularity)
        track.areas.append(raw_area)
        track.relative_areas.append(relative_area)
        track.last_center_y = center_y
        
        # Store GPS if available
        if lat is not None and lon is not None:
            track.last_latitude = lat
            track.last_longitude = lon
            self.detections_with_gps += 1
        
        # === BEST FRAME TRACKING - UPDATE DYNAMICALLY ===
        # Capture the frame with highest severity throughout tracking lifecycle
        # Also ensure first frame is captured even if severity is low
        if severity_score > track.best_severity or track.best_frame is None:
            # Memory optimization: Only save cropped ROI area, not full frame
            x1, y1, x2, y2 = map(int, bbox)
            pad = 50  # Padding around detection
            h, w = frame.shape[:2]
            y1_crop = max(0, y1 - pad)
            y2_crop = min(h, y2 + pad)
            x1_crop = max(0, x1 - pad)
            x2_crop = min(w, x2 + pad)
            
            track.best_frame = frame[y1_crop:y2_crop, x1_crop:x2_crop].copy()
            track.best_bbox = bbox
            track.best_mask_polygon = mask_polygon.copy()
            track.best_severity = severity_score
            track.best_relative_area = relative_area
            track.best_circularity = circularity
        
        # === EXIT LINE LOGIC - COMMIT ON CROSSING ===
        # Check if pothole center has crossed the exit line
        was_above_exit = not track.has_crossed_exit
        is_past_exit = self.geometry.is_past_exit_line(center_y)
        
        if is_past_exit and was_above_exit and not track.committed_to_db:
            # First time crossing exit line - COMMIT!
            track.has_crossed_exit = True
            
            # Best frame already captured above (highest severity frame)
            # Commit to database using helper method
            self._commit_track_to_database(
                track_id=track_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                save_reason="Exit Line Crossed"
            )
        
        # === VISUALIZATION ===
        
        # 1. Draw mask overlay with priority color
        frame = self._draw_mask_overlay(frame, mask_polygon, color_bgr)
        
        # 2. Draw bounding box
        frame = self._draw_bounding_box(frame, bbox, color_bgr)
        
        # 3. Draw detection label with risk info
        id_str = f"ID:{track_id}"
        committed_marker = "✓" if track.committed_to_db else ""
        label_text = f"{id_str} | {risk_label} ({severity_score:.0f}) {committed_marker}"
        
        # Position label above bounding box
        label_x = int(bbox[0])
        label_y = int(bbox[1]) - 5
        if label_y < 20:
            label_y = int(bbox[3]) + 20
        
        frame = self._draw_label(frame, label_text, (label_x, label_y), color_bgr)
        
        # 4. GPS overlay (if available)
        if lat is not None and lon is not None:
            gps_y = label_y + 25
            if gps_y > self.frame_height - 20:
                gps_y = label_y - 25
            frame = self._draw_gps_overlay(frame, lat, lon, (label_x, gps_y))
        
        # Update statistics
        self.total_detections += 1
        
        return frame
    
    def _handle_lost_tracks(
        self,
        current_track_ids: set,
        frame_idx: int,
        timestamp: str
    ) -> None:
        """
        Handle lost/removed tracks with proximity-based save logic.
        
        When a track disappears near the Exit Line, assume it went under
        the car and save it to prevent valid potholes from being discarded.
        
        Args:
            current_track_ids: Set of track IDs detected in current frame
            frame_idx: Current frame index
            timestamp: ISO timestamp for logging
        """
        if self.geometry is None:
            return
        
        # Find tracks that were present in previous frame but not in current
        lost_track_ids = self.previous_frame_track_ids - current_track_ids
        
        if not lost_track_ids:
            return
        
        # Define safety buffer (15% of frame height above exit line)
        safety_buffer = int(self.frame_height * 0.15)
        proximity_threshold = self.geometry.exit_line_y - safety_buffer
        
        for track_id in lost_track_ids:
            # Skip if track doesn't exist
            if track_id not in self.track_data:
                continue
            
            track = self.track_data[track_id]
            
            # Skip if already committed (critical: prevents duplicate saves)
            if track.committed_to_db:
                continue
            
            # Check if last known position was near exit line
            if track.last_center_y >= proximity_threshold:
                # Track disappeared near bottom - save via proximity logic
                print(f"[Proximity Save] Track {track_id} lost at y={track.last_center_y:.0f} "
                      f"(threshold={proximity_threshold}, exit={self.geometry.exit_line_y})")
                
                # Force save to database
                self._commit_track_to_database(
                    track_id=track_id,
                    frame_idx=frame_idx,
                    timestamp=timestamp,
                    save_reason="Proximity Logic"
                )
                
                self.tracks_saved_via_proximity += 1
    
    def _commit_track_to_database(
        self,
        track_id: int,
        frame_idx: int,
        timestamp: str,
        save_reason: str = "Exit Line Crossed"
    ) -> None:
        """
        Commit a track to the database with snapshot saving and depth validation.
        
        DEPTH VALIDATION: Uses Depth Anything V2 to verify if detection is a true pothole.
        Rejects:
        - Bumps (tümsek) - raised surfaces
        - Shadows (gölge) - inconsistent depth
        - Stains/patches (leke/yama) - flat surfaces
        
        Args:
            track_id: Track ID to commit
            frame_idx: Current frame index
            timestamp: ISO timestamp
            save_reason: Reason for saving (for logging)
        """
        if track_id not in self.track_data:
            return
        
        track = self.track_data[track_id]
        
        # Skip if already committed
        if track.committed_to_db:
            return
        
        # === DEPTH VALIDATION - CRITICAL FILTER ===
        # Validate using depth analysis before committing
        if track.best_frame is not None and track.best_bbox is not None:
            # Get the full frame for depth analysis (reconstruct from crop if needed)
            # Since best_frame is cropped, we need the original frame
            # For now, we'll use the best_frame directly with its bbox adjusted
            
            # Run depth validation
            is_valid = self.depth_validator.is_valid_pothole(track.best_frame, (0, 0, track.best_frame.shape[1], track.best_frame.shape[0]))
            
            if not is_valid:
                print(f"[DepthValidator] Track {track_id} REJECTED - Not a valid pothole (bump/shadow/stain)")
                logger.info(f"Track {track_id} rejected by depth validation")
                # Mark as committed to prevent re-processing, but don't save to DB
                track.committed_to_db = True
                return
        
        # Get GPS coordinates
        lat, lon = self.gps_manager.get_location(frame_idx)
        
        # Calculate priority
        priority = get_priority_level(track.avg_severity)
        risk_label = get_risk_label(track.avg_severity)
        
        # Save snapshot if we have best frame
        image_path = None
        heatmap_path = None
        
        if track.best_frame is not None and self.db_manager is not None:
            print(f"[Snapshot] Attempting to save for Track {track_id} (best_frame: {track.best_frame.shape}, best_bbox: {track.best_bbox})")
            image_path = self.db_manager.save_snapshot(
                frame=track.best_frame,
                track_id=track_id,
                bbox=track.best_bbox
            )
            
            # Generate and save depth heatmap
            if track.best_bbox is not None:
                # Use full bbox coordinates relative to cropped best_frame
                heatmap = self.depth_validator.get_heatmap(
                    track.best_frame,
                    (0, 0, track.best_frame.shape[1], track.best_frame.shape[0])
                )
                
                if heatmap is not None:
                    heatmap_path = self.db_manager.save_heatmap(
                        heatmap=heatmap,
                        track_id=track_id
                    )
                    print(f"[Heatmap] Generated and saved for Track {track_id}")
                else:
                    print(f"[Heatmap] WARNING: Could not generate heatmap for Track {track_id}")
        else:
            if track.best_frame is None:
                print(f"[Snapshot] WARNING: Track {track_id} has no best_frame!")
            if self.db_manager is None:
                print(f"[Snapshot] WARNING: db_manager is None!")
        
        # Insert into database
        if self.db_manager is not None:
            self.db_manager.insert_detection(
                track_id=track_id,
                timestamp=timestamp,
                latitude=lat,
                longitude=lon,
                severity_score=track.avg_severity,
                priority_level=priority.value,
                risk_label=risk_label,
                circularity=track.avg_circularity,
                relative_area=track.avg_relative_area,
                image_path=image_path,
                heatmap_path=heatmap_path
            )
            
            print(f"[Database] Track {track_id} committed via {save_reason} "
                  f"(Severity: {track.avg_severity:.1f}, Priority: {priority.value}, Depth: VALIDATED)")
        
        track.committed_to_db = True
        self.tracks_committed += 1
    
    def _process_frame(
        self, 
        frame: np.ndarray, 
        frame_idx: int,
        timestamp: str
    ) -> np.ndarray:
        """
        Process a single video frame with IPM visualization.
        
        Args:
            frame: BGR image from video
            frame_idx: Current frame index (for GPS lookup)
            timestamp: ISO timestamp for DB logging
            
        Returns:
            Annotated frame (with optional ROI and Exit Line overlays)
        """
        frame_area = self.frame_width * self.frame_height
        
        # Run detection with tracking
        results = self.detector.track_frame(frame, persist=True, verbose=False)
        
        # Extract detections
        detections = self.detector.extract_detections(results)
        
        # Collect current frame track IDs
        current_track_ids = set()
        for detection in detections:
            track_id = detection.get('track_id')
            if track_id is not None:
                current_track_ids.add(track_id)
        
        # === LOST TRACK RECOVERY ===
        # Check for tracks that disappeared near exit line
        self._handle_lost_tracks(current_track_ids, frame_idx, timestamp)
        
        # Update previous frame track IDs for next iteration
        self.previous_frame_track_ids = current_track_ids
        
        # Process each detection
        for detection in detections:
            frame = self._process_detection(
                frame, detection, frame_area, frame_idx, timestamp
            )
        
        # Draw ROI overlay (if enabled)
        if self.SHOW_ROI_OVERLAY and self.geometry is not None:
            frame = self.geometry.draw_roi_overlay(frame, color=(0, 255, 255), alpha=0.1)
        
        # Draw Exit Line (if enabled)
        if self.SHOW_EXIT_LINE and self.geometry is not None:
            frame = self.geometry.draw_exit_line(frame, color=(255, 0, 255), thickness=2)
        
        # Add frame counter and committed count
        proximity_info = f" (Proximity: {self.tracks_saved_via_proximity})" if self.tracks_saved_via_proximity > 0 else ""
        info_text = f"Frame: {frame_idx} | Committed: {self.tracks_committed}{proximity_info}"
        cv2.putText(frame, info_text, (10, self.frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _create_debug_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Create debug view with Original and Bird's Eye View side-by-side.
        
        Args:
            frame: Annotated BGR frame
            
        Returns:
            Debug view with both perspectives
        """
        if self.geometry is None:
            return frame
        
        return self.geometry.create_debug_view(
            frame, 
            show_roi=True, 
            show_exit_line=True
        )
    
    def process(self) -> Dict[int, TrackData]:
        """
        Run the full video processing pipeline with IPM and Exit Line logic.
        
        This method:
        1. Opens input/output videos
        2. Initializes GeometryProcessor for IPM
        3. Processes each frame with detection and tracking
        4. Uses IPM for accurate area measurement
        5. Commits to DB only when crossing Exit Line
        6. Saves best frame snapshots
        7. Writes annotated frames to output
        8. Returns aggregated track data for CSV generation
        
        Returns:
            Dictionary mapping track_id -> TrackData with accumulated metrics
            
        Raises:
            VideoProcessingError: If processing fails
        """
        try:
            # Initialize video I/O
            self._open_video()
            
            # Reset tracking data and state
            self.track_data.clear()
            self.previous_frame_track_ids.clear()
            self.frames_processed = 0
            self.total_detections = 0
            self.detections_with_gps = 0
            self.tracks_committed = 0
            self.tracks_saved_via_proximity = 0
            
            print(f"\n[VideoProcessor] Starting processing...")
            print(f"[VideoProcessor] Mode: IPM Enhanced (Exit Line Commit)")
            print(f"[VideoProcessor] GPS Data: {'Available' if self.gps_manager.is_available else 'Not available (strict mode)'}")
            print(f"[VideoProcessor] Database: {'Enabled' if self.db_manager else 'Disabled'}")
            
            # Processing loop with progress bar
            with tqdm(total=self.total_frames, desc="Processing", unit="frame") as pbar:
                frame_idx = 0
                while True:
                    ret, frame = self.cap.read()
                    
                    if not ret:
                        break
                    
                    # Generate timestamp for this frame
                    timestamp = datetime.now().isoformat()
                    
                    # Process frame (detection + IPM + visualization + GPS + DB)
                    annotated_frame = self._process_frame(frame, frame_idx, timestamp)
                    
                    # Write to output
                    self.writer.write(annotated_frame)
                    
                    # Update progress
                    self.frames_processed += 1
                    frame_idx += 1
                    pbar.update(1)
                    
                    # Update postfix with detection and commit count
                    pbar.set_postfix({
                        'tracks': len(self.track_data),
                        'committed': self.tracks_committed
                    })
            
            # Print summary
            print(f"\n[VideoProcessor] Processing complete!")
            print(f"[VideoProcessor] Frames processed: {self.frames_processed}")
            print(f"[VideoProcessor] Total detections (per-frame): {self.total_detections}")
            print(f"[VideoProcessor] Unique tracks: {len(self.track_data)}")
            print(f"[VideoProcessor] Tracks committed (crossed Exit Line): {self.tracks_committed}")
            if self.tracks_saved_via_proximity > 0:
                print(f"[VideoProcessor] Tracks saved via Proximity Logic: {self.tracks_saved_via_proximity}")
            print(f"[VideoProcessor] Detections with GPS: {self.detections_with_gps}")
            print(f"[VideoProcessor] Output saved to: {self.output_path}")
            
            return self.track_data
            
        except Exception as e:
            raise VideoProcessingError(f"Processing failed: {str(e)}") from e
            
        finally:
            # Always release resources
            self._close_video()
    
    def get_summary_stats(self) -> dict:
        """
        Get summary statistics after processing.
        
        Returns:
            Dictionary with processing statistics (IPM enhanced)
        """
        if not self.track_data:
            return {
                'frames_processed': self.frames_processed,
                'total_detections': 0,
                'detections_with_gps': 0,
                'unique_tracks': 0,
                'tracks_committed': 0,
                'high_priority_count': 0,
                'medium_priority_count': 0,
                'low_priority_count': 0,
                'gps_available': self.gps_manager.is_available,
                'db_enabled': self.db_manager is not None,
                'ipm_enabled': self.geometry is not None
            }
        
        # Count by priority level (only committed tracks)
        priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        for track in self.track_data.values():
            if track.committed_to_db:
                priority_counts[track.priority_level] += 1
        
        return {
            'frames_processed': self.frames_processed,
            'total_detections': self.total_detections,
            'detections_with_gps': self.detections_with_gps,
            'unique_tracks': len(self.track_data),
            'tracks_committed': self.tracks_committed,
            'high_priority_count': priority_counts['HIGH'],
            'medium_priority_count': priority_counts['MEDIUM'],
            'low_priority_count': priority_counts['LOW'],
            'gps_available': self.gps_manager.is_available,
            'db_enabled': self.db_manager is not None,
            'ipm_enabled': self.geometry is not None
        }
    
    def process_generator(self, write_output: bool = True):
        """
        Generator-based processing for Streamlit UI.
        
        Yields FrameResult for each processed frame, allowing real-time
        display in web interfaces.
        
        Args:
            write_output: Whether to write annotated video to file
            
        Yields:
            FrameResult: Frame data and statistics for each processed frame
            
        Usage (Streamlit):
            for result in processor.process_generator():
                st.image(result.frame_rgb)
                st.progress(result.progress_percent / 100)
        """
        try:
            # Initialize video I/O
            self._open_video()
            
            # Reset tracking data and state
            self.track_data.clear()
            self.previous_frame_track_ids.clear()
            self.frames_processed = 0
            self.total_detections = 0
            self.detections_with_gps = 0
            self.tracks_committed = 0
            self.tracks_saved_via_proximity = 0
            
            frame_idx = 0
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Generate timestamp for this frame
                timestamp = datetime.now().isoformat()
                
                # Process frame (detection + IPM + visualization + GPS + DB)
                annotated_frame = self._process_frame(frame, frame_idx, timestamp)
                
                # Write to output if enabled
                if write_output and self.writer is not None:
                    self.writer.write(annotated_frame)
                
                # Update progress
                self.frames_processed += 1
                frame_idx += 1
                
                # Calculate current statistics (committed tracks only for priority)
                priority_counts = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
                total_severity = 0.0
                for track in self.track_data.values():
                    if track.committed_to_db:
                        priority_counts[track.priority_level] += 1
                    total_severity += track.avg_severity
                
                avg_severity = total_severity / len(self.track_data) if self.track_data else 0.0
                progress = (frame_idx / self.total_frames) * 100 if self.total_frames > 0 else 0
                
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Yield result
                yield FrameResult(
                    frame_idx=frame_idx,
                    frame_rgb=frame_rgb,
                    total_frames=self.total_frames,
                    progress_percent=progress,
                    current_detections=len(self.track_data),
                    total_detections=self.total_detections,
                    unique_tracks=len(self.track_data),
                    avg_severity=avg_severity,
                    high_count=priority_counts['HIGH'],
                    medium_count=priority_counts['MEDIUM'],
                    low_count=priority_counts['LOW']
                )
            
        finally:
            # Always release resources
            self._close_video()
    
    def get_track_data_as_dataframe(self):
        """
        Convert track data to a pandas DataFrame for display.
        
        Returns:
            pandas.DataFrame with track data (IPM enhanced)
        """
        import pandas as pd
        
        if not self.track_data:
            return pd.DataFrame()
        
        data = []
        for track in self.track_data.values():
            data.append({
                'Cukur ID': track.track_id,
                'Ciddiyet': round(track.avg_severity, 2),
                'Risk Seviyesi': track.risk_label,
                'Duzensizlik': round(track.avg_circularity, 4),
                'Goreceli Alan': round(track.avg_relative_area * 100, 4),  # As percentage
                'Maks Alan': int(track.max_area),
                'Oncelik': track.priority_level,
                'Kaydedildi': 'Evet' if track.committed_to_db else 'Hayir',
                'Kareler': track.frame_appearances,
                'Enlem': track.last_latitude,
                'Boylam': track.last_longitude
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Ciddiyet', ascending=False)
        return df
    
    def process_optimized(self, status_callback=None) -> ProcessingResult:
        """
        Optimized processing for Streamlit - NO live image preview.
        
        This method runs at maximum speed by:
        1. Not yielding image frames (no RGB conversion overhead)
        2. Only reporting progress percentage via callback
        3. Measuring and returning FPS metrics
        
        Args:
            status_callback: Optional callable(ProcessingStatus) for progress updates.
                           Called every N frames for UI updates without blocking.
        
        Returns:
            ProcessingResult with stats, output path, and FPS metrics
            
        Usage (Streamlit):
            def update_progress(status):
                progress_bar.progress(status.progress_percent / 100)
                status_text.text(status.status_message)
            
            result = processor.process_optimized(status_callback=update_progress)
            st.metric("FPS", f"{result.average_fps:.2f}")
        """
        import time
        
        try:
            # Initialize video I/O
            self._open_video()
            
            # Reset tracking data and state
            self.track_data.clear()
            self.previous_frame_track_ids.clear()
            self.frames_processed = 0
            self.total_detections = 0
            self.detections_with_gps = 0
            self.tracks_committed = 0
            self.tracks_saved_via_proximity = 0
            
            # Performance timing
            start_time = time.time()
            
            frame_idx = 0
            update_interval = max(1, self.total_frames // 100)  # Update ~100 times
            
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    break
                
                # Generate timestamp for this frame
                timestamp = datetime.now().isoformat()
                
                # Process frame (detection + IPM + visualization + GPS + DB)
                annotated_frame = self._process_frame(frame, frame_idx, timestamp)
                
                # Write to output
                if self.writer is not None:
                    self.writer.write(annotated_frame)
                
                # Update progress
                self.frames_processed += 1
                frame_idx += 1
                
                # Report progress via callback (throttled for performance)
                if status_callback and (frame_idx % update_interval == 0 or frame_idx == self.total_frames):
                    progress = (frame_idx / self.total_frames) * 100 if self.total_frames > 0 else 0
                    status = ProcessingStatus(
                        frame_idx=frame_idx,
                        total_frames=self.total_frames,
                        progress_percent=progress,
                        total_detections=self.tracks_committed,  # Show committed count
                        unique_tracks=len(self.track_data),
                        status_message=f"Frame {frame_idx}/{self.total_frames} | Committed: {self.tracks_committed}"
                    )
                    status_callback(status)
            
            # Calculate performance metrics
            end_time = time.time()
            processing_time = end_time - start_time
            average_fps = self.total_frames / processing_time if processing_time > 0 else 0
            
            # Get final stats
            stats = self.get_summary_stats()
            
            return ProcessingResult(
                track_data=self.track_data,
                stats=stats,
                output_video_path=str(self.output_path),
                processing_time_seconds=processing_time,
                average_fps=average_fps,
                total_frames=self.total_frames
            )
            
        finally:
            # Always release resources
            self._close_video()


def generate_csv_report(
    track_data: Dict[int, TrackData],
    output_path: Union[str, Path],
    only_committed: bool = True
) -> None:
    """
    Generate a CSV report from aggregated track data (Turkish localized).
    
    CSV columns (Turkish):
    - cukur_id: Unique tracking ID
    - ciddiyet_puani: Average severity score (0-100)
    - risk_seviyesi: Dusuk/Orta/Yuksek
    - duzensizlik: Average circularity value (0-1)
    - goreceli_alan_yuzde: Average IPM-normalized area (%)
    - maksimum_alan: Maximum pixel area detected
    - oncelik: DUSUK, ORTA, or YUKSEK
    - kaydedildi: Whether crossed Exit Line
    - gorunme_sayisi: Number of frames object appeared in
    - enlem: GPS latitude (if available)
    - boylam: GPS longitude (if available)
    
    Args:
        track_data: Dictionary from VideoProcessor.process()
        output_path: Path for output CSV file
        only_committed: If True, only include tracks that crossed Exit Line
    """
    import csv
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Filter to committed tracks if requested
    tracks_to_export = [
        t for t in track_data.values()
        if not only_committed or t.committed_to_db
    ]
    
    # Sort by severity (highest first)
    sorted_tracks = sorted(
        tracks_to_export,
        key=lambda t: t.avg_severity,
        reverse=True
    )
    
    # Turkish priority labels
    priority_tr = {'HIGH': 'YUKSEK', 'MEDIUM': 'ORTA', 'LOW': 'DUSUK'}
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Turkish Header
        writer.writerow([
            'cukur_id',
            'ciddiyet_puani',
            'risk_seviyesi',
            'duzensizlik',
            'goreceli_alan_yuzde',
            'maksimum_alan',
            'oncelik',
            'kaydedildi',
            'gorunme_sayisi',
            'enlem',
            'boylam'
        ])
        
        # Data rows
        for track in sorted_tracks:
            writer.writerow([
                track.track_id,
                f"{track.avg_severity:.2f}",
                track.risk_label,  # Already Turkish from get_risk_label()
                f"{track.avg_circularity:.4f}",
                f"{track.avg_relative_area * 100:.4f}",
                f"{track.max_area:.0f}",
                priority_tr.get(track.priority_level, track.priority_level),
                "Evet" if track.committed_to_db else "Hayir",
                track.frame_appearances,
                track.last_latitude if track.last_latitude else "",
                track.last_longitude if track.last_longitude else ""
            ])
    
    print(f"[Rapor] CSV kaydedildi: {output_path}")
    print(f"[Rapor] Toplam kayit: {len(sorted_tracks)} (kaydedilen: {sum(1 for t in sorted_tracks if t.committed_to_db)})")


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add parent directory to path for relative imports when running directly
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from src.video_processor import TrackData, generate_csv_report
    from src.metrics import get_priority_level
    
    print("=" * 60)
    print("video_processor.py - Module Test")
    print("=" * 60)
    
    # Test TrackData dataclass
    print("\n[Test 1] TrackData dataclass:")
    track = TrackData(track_id=1)
    track.frame_appearances = 5
    track.severity_scores = [45.0, 50.0, 55.0, 48.0, 52.0]
    track.circularity_values = [0.65, 0.70, 0.68, 0.72, 0.66]
    track.areas = [1000, 1200, 1500, 1100, 1300]
    
    print(f"  Track ID: {track.track_id}")
    print(f"  Appearances: {track.frame_appearances}")
    print(f"  Avg Severity: {track.avg_severity:.2f}")
    print(f"  Avg Circularity: {track.avg_circularity:.4f}")
    print(f"  Max Area: {track.max_area}")
    print(f"  Priority: {track.priority_level}")
    
    # Test CSV generation with mock data
    print("\n[Test 2] CSV Report Generation:")
    mock_data = {
        1: track,
        2: TrackData(
            track_id=2,
            frame_appearances=3,
            severity_scores=[75.0, 80.0, 78.0],
            circularity_values=[0.35, 0.38, 0.36],
            areas=[5000, 5500, 5200]
        ),
        3: TrackData(
            track_id=3,
            frame_appearances=2,
            severity_scores=[20.0, 22.0],
            circularity_values=[0.85, 0.88],
            areas=[500, 550]
        )
    }
    
    # Generate test CSV
    test_csv_path = Path("output/test_report.csv")
    generate_csv_report(mock_data, test_csv_path)
    
    # Read and display CSV
    if test_csv_path.exists():
        print(f"\n  CSV Contents:")
        with open(test_csv_path, 'r') as f:
            for line in f:
                print(f"    {line.strip()}")
    
    print("\n" + "=" * 60)
    print("Module test completed!")
    print("=" * 60)

