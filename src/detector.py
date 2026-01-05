"""
detector.py - YOLO Model Wrapper for Pothole Detection

This module provides a high-level interface for:
1. Loading TensorRT optimized YOLO models (.engine)
2. Running inference with instance segmentation
3. Object tracking with ByteTrack for temporal persistence

Usage:
    from src.detector import PotholeDetector
    
    detector = PotholeDetector("best.engine")
    results = detector.track_frame(frame)
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Any
import numpy as np

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False


class ModelLoadError(Exception):
    """Raised when the model file cannot be loaded."""
    pass


class PotholeDetector:
    """
    YOLO-based pothole detector with instance segmentation and tracking.
    
    This class wraps the Ultralytics YOLO model for:
    - TensorRT accelerated inference (.engine files)
    - Instance segmentation (mask extraction)
    - Object tracking with ByteTrack for ID persistence
    
    Attributes:
        model_path (Path): Path to the .engine model file
        model (YOLO): Loaded YOLO model instance
        conf_threshold (float): Confidence threshold for detections
        tracker_config (str): Path to tracker configuration (e.g., bytetrack.yaml)
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        tracker_config: str = "bytetrack.yaml"
    ):
        """
        Initialize the PotholeDetector with a TensorRT model.
        
        Args:
            model_path: Path to the .engine model file
            conf_threshold: Minimum confidence for detections (default: 0.5)
            iou_threshold: IoU threshold for NMS (default: 0.5)
            tracker_config: Tracker configuration file (default: bytetrack.yaml)
            
        Raises:
            ImportError: If ultralytics is not installed
            ModelLoadError: If model file doesn't exist or fails to load
        """
        # Check ultralytics availability
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics is required for model inference. "
                "Install with: pip install ultralytics"
            )
        
        # Validate model path
        self.model_path = Path(model_path)
        self._validate_model_path()
        
        # Store configuration
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.tracker_config = tracker_config
        
        # Load the model
        self.model = self._load_model()
        
        # Track if this is the first frame (for tracker initialization)
        self._tracking_initialized = False
    
    def _validate_model_path(self) -> None:
        """
        Validate that the model file exists and has correct extension.
        
        Raises:
            ModelLoadError: If file doesn't exist or has wrong extension
        """
        if not self.model_path.exists():
            raise ModelLoadError(
                f"Model file not found: {self.model_path.absolute()}\n"
                f"Please ensure 'best.engine' is in the project root directory."
            )
        
        if self.model_path.suffix.lower() != '.engine':
            raise ModelLoadError(
                f"Invalid model format: {self.model_path.suffix}\n"
                f"Expected TensorRT engine file (.engine)"
            )
    
    def _load_model(self) -> "YOLO":
        """
        Load the YOLO model with segmentation task.
        
        Returns:
            Loaded YOLO model instance
            
        Raises:
            ModelLoadError: If model fails to load
        """
        try:
            # Load model with explicit task='segment' for instance segmentation
            model = YOLO(str(self.model_path), task='segment')
            print(f"[Detector] Model loaded successfully: {self.model_path.name}")
            print(f"[Detector] Task: segment | Conf: {self.conf_threshold} | IoU: {self.iou_threshold}")
            return model
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load model from {self.model_path}: {str(e)}"
            )
    
    def track_frame(
        self,
        frame: np.ndarray,
        persist: bool = True,
        verbose: bool = False
    ) -> Any:
        """
        Run detection and tracking on a single video frame.
        
        This method performs:
        1. Instance segmentation to detect potholes
        2. ByteTrack tracking to assign persistent IDs
        
        Args:
            frame: BGR image array from cv2.VideoCapture (H, W, 3)
            persist: Keep tracking state between frames (default: True)
                     Must be True for consistent ID assignment across video
            verbose: Print inference logs (default: False)
            
        Returns:
            Ultralytics Results object containing:
            - boxes: Bounding boxes with tracking IDs
            - masks: Instance segmentation masks
            - Use results[0].boxes.id for track IDs
            - Use results[0].masks.xy for mask polygons
            
        Example:
            results = detector.track_frame(frame)
            if results[0].boxes.id is not None:
                for box, mask, track_id in zip(
                    results[0].boxes.xyxy,
                    results[0].masks.xy,
                    results[0].boxes.id
                ):
                    print(f"Track ID: {track_id}, Box: {box}")
        """
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker_config,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose
        )
        
        self._tracking_initialized = True
        return results
    
    def predict_frame(
        self,
        frame: np.ndarray,
        verbose: bool = False
    ) -> Any:
        """
        Run detection on a single frame WITHOUT tracking.
        
        Use this for single-image inference where tracking IDs aren't needed.
        
        Args:
            frame: BGR image array (H, W, 3)
            verbose: Print inference logs (default: False)
            
        Returns:
            Ultralytics Results object (without track IDs)
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=verbose
        )
        return results
    
    def reset_tracker(self) -> None:
        """
        Reset the tracker state.
        
        Call this when starting a new video to clear previous track IDs.
        """
        self._tracking_initialized = False
        # Re-load model to reset internal tracker state
        self.model = self._load_model()
        print("[Detector] Tracker state reset")
    
    def get_detection_count(self, results: Any) -> int:
        """
        Get the number of detections from a results object.
        
        Args:
            results: Results from track_frame or predict_frame
            
        Returns:
            Number of detected objects
        """
        if results is None or len(results) == 0:
            return 0
        
        result = results[0]
        if result.boxes is None:
            return 0
        
        return len(result.boxes)
    
    @staticmethod
    def extract_detections(results: Any) -> List[dict]:
        """
        Extract detection data from results into a list of dictionaries.
        
        Args:
            results: Results from track_frame or predict_frame
            
        Returns:
            List of dictionaries, each containing:
            - 'track_id': int or None (if tracking not used)
            - 'bbox': [x1, y1, x2, y2] bounding box
            - 'confidence': float detection confidence
            - 'mask_polygon': numpy array of mask polygon points or None
        """
        detections = []
        
        if results is None or len(results) == 0:
            return detections
        
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            return detections
        
        boxes = result.boxes.xyxy.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        # Track IDs (may be None if tracking not used)
        track_ids = None
        if result.boxes.id is not None:
            track_ids = result.boxes.id.cpu().numpy().astype(int)
        
        # Masks (may be None if segmentation failed)
        masks = None
        if result.masks is not None:
            masks = result.masks.xy  # List of polygon coordinates
        
        for i in range(len(boxes)):
            detection = {
                'track_id': int(track_ids[i]) if track_ids is not None else None,
                'bbox': boxes[i].tolist(),
                'confidence': float(confidences[i]),
                'mask_polygon': masks[i] if masks is not None else None
            }
            detections.append(detection)
        
        return detections
    
    @property
    def is_tracking_active(self) -> bool:
        """Check if tracking has been initialized."""
        return self._tracking_initialized
    
    def __repr__(self) -> str:
        return (
            f"PotholeDetector("
            f"model='{self.model_path.name}', "
            f"conf={self.conf_threshold}, "
            f"tracker='{self.tracker_config}')"
        )


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("detector.py - Module Test")
    print("=" * 60)
    
    # Test 1: Check for model file
    model_paths_to_try = [
        Path("best.engine"),
        Path("best1.engine"),  # Alternative name in project
    ]
    
    model_path = None
    for path in model_paths_to_try:
        if path.exists():
            model_path = path
            print(f"\n[Test 1] Found model file: {path}")
            break
    
    if model_path is None:
        print("\n[Test 1] No model file found. Skipping load test.")
        print("  Searched for: best.engine, best1.engine")
        sys.exit(0)
    
    # Test 2: Initialize detector
    try:
        print(f"\n[Test 2] Initializing PotholeDetector...")
        detector = PotholeDetector(
            model_path=model_path,
            conf_threshold=0.5,
            tracker_config="bytetrack.yaml"
        )
        print(f"  {detector}")
        print("  Initialization successful!")
    except ImportError as e:
        print(f"  Import Error: {e}")
        sys.exit(1)
    except ModelLoadError as e:
        print(f"  Model Load Error: {e}")
        sys.exit(1)
    
    # Test 3: Test with dummy frame (if OpenCV available)
    try:
        import cv2
        print(f"\n[Test 3] Testing inference with dummy frame...")
        
        # Create a dummy 640x480 black frame
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        results = detector.track_frame(dummy_frame, verbose=False)
        detection_count = detector.get_detection_count(results)
        
        print(f"  Frame shape: {dummy_frame.shape}")
        print(f"  Detections: {detection_count}")
        print("  Inference successful!")
        
    except ImportError:
        print(f"\n[Test 3] OpenCV not available. Skipping inference test.")
    except Exception as e:
        print(f"\n[Test 3] Inference error: {e}")
    
    print("\n" + "=" * 60)
    print("Module test completed!")
    print("=" * 60)

