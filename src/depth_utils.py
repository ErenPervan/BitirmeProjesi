"""
depth_utils.py - Depth Anything V2 Integration for Topographic Verification

This module provides depth estimation for pothole validation using Depth Anything V2.
The depth analysis helps filter out false positives like:
- Bumps (tümsek) - show as raised surfaces
- Shadows (gölge) - show inconsistent depth
- Stains/patches (leke/yama) - show flat depth
- True potholes - show clear depth depression

Features:
- CUDA acceleration (if available)
- Visual heatmap generation with INFERNO colormap
- Depth-based validation to reject non-pothole detections

Usage:
    validator = DepthValidator()
    heatmap = validator.get_heatmap(frame, bbox)
    is_valid = validator.is_valid_pothole(frame, bbox)
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_AVAILABLE = True
except ImportError:
    DEPTH_AVAILABLE = False
    logger.warning("Depth Anything V2 not found. Depth validation disabled.")


class DepthValidator:
    """
    Depth-based validation for pothole detections using Depth Anything V2.
    
    This class:
    1. Loads the VITS model from checkpoints/depth_anything_v2_vits.pth
    2. Generates depth heatmaps for visual inspection
    3. Validates detections based on depth characteristics
    
    Attributes:
        model: DepthAnythingV2 model instance
        device: Computation device ('cuda' or 'cpu')
        enabled: Whether depth validation is active
    """
    
    # Model configuration for VITS (Small model - fast inference)
    MODEL_CONFIGS = {
        'vits': {
            'encoder': 'vits',
            'features': 64,
            'out_channels': [48, 96, 192, 384]
        }
    }
    
    # Validation thresholds
    MIN_DEPTH_VARIATION = 0.15  # Minimum depth variation (15% of range) to be valid pothole
    MIN_DEPRESSION_RATIO = 0.3  # At least 30% of pixels should be below median depth
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize the DepthValidator.
        
        Args:
            model_path: Path to model weights (default: checkpoints/depth_anything_v2_vits.pth)
            device: Computation device ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.enabled = False
        self.model = None
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not DEPTH_AVAILABLE:
            logger.warning("[DepthValidator] Depth Anything V2 not available - validation disabled")
            print("[DepthValidator] WARNING: Depth Anything V2 not found - depth validation disabled")
            return
        
        # Set default model path
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_path = project_root / "checkpoints" / "depth_anything_v2_vits.pth"
        else:
            model_path = Path(model_path)
        
        # Validate model file exists
        if not model_path.exists():
            logger.error(f"[DepthValidator] Model weights not found: {model_path}")
            print(f"[DepthValidator] ERROR: Model weights not found at {model_path}")
            return
        
        try:
            # Initialize model with VITS configuration
            config = self.MODEL_CONFIGS['vits']
            self.model = DepthAnythingV2(
                encoder=config['encoder'],
                features=config['features'],
                out_channels=config['out_channels']
            )
            
            # Load weights with CPU mapping first (then move to device)
            state_dict = torch.load(str(model_path), map_location='cpu')
            self.model.load_state_dict(state_dict)
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
            self.enabled = True
            logger.info(f"[DepthValidator] Initialized successfully on {self.device}")
            print(f"[DepthValidator] Model loaded: {model_path.name} (Device: {self.device})")
            
        except Exception as e:
            logger.exception(f"[DepthValidator] Failed to initialize: {e}")
            print(f"[DepthValidator] ERROR: Failed to load model - {e}")
            self.enabled = False
    
    def get_heatmap(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float],
        colormap: int = cv2.COLORMAP_INFERNO
    ) -> Optional[np.ndarray]:
        """
        Generate a visual depth heatmap for a detection.
        
        Args:
            frame: BGR frame from video
            bbox: Bounding box (x1, y1, x2, y2) in pixels
            colormap: OpenCV colormap (default: INFERNO for hot-to-cold visualization)
            
        Returns:
            Colorized heatmap image (BGR) or None if validation disabled/failed
        """
        if not self.enabled:
            return None
        
        try:
            # Crop frame to bbox with safety checks
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            
            # Validate and clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Check for invalid crop
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"[DepthValidator] Invalid bbox: ({x1},{y1})-({x2},{y2})")
                return None
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                logger.warning("[DepthValidator] Empty crop after bbox extraction")
                return None
            
            # Run depth inference
            with torch.no_grad():
                depth = self.model.infer_image(crop)
            
            # Normalize to 0-255 for visualization
            depth_min = depth.min()
            depth_max = depth.max()
            
            if depth_max - depth_min < 1e-6:
                # Flat depth map - likely invalid
                logger.warning("[DepthValidator] Flat depth map detected (no variation)")
                depth_normalized = np.zeros_like(depth, dtype=np.uint8)
            else:
                depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
            
            # Apply colormap (INFERNO: dark blue = far, bright yellow = close)
            heatmap = cv2.applyColorMap(depth_normalized, colormap)
            
            return heatmap
            
        except Exception as e:
            logger.exception(f"[DepthValidator] Exception in get_heatmap: {e}")
            return None
    
    def is_valid_pothole(
        self,
        frame: np.ndarray,
        bbox: Tuple[float, float, float, float]
    ) -> bool:
        """
        Validate if a detection is a true pothole based on depth characteristics.
        
        True potholes should show:
        - Significant depth variation (not flat like stains/patches)
        - Depression pattern (pixels darker/deeper than surroundings)
        - Consistent depth profile (not noise like shadows)
        
        Args:
            frame: BGR frame from video
            bbox: Bounding box (x1, y1, x2, y2) in pixels
            
        Returns:
            True if detection passes depth validation (is likely a real pothole)
            False if detection fails (likely bump, shadow, stain, or flat surface)
        """
        if not self.enabled:
            # If depth validation is disabled, accept all detections
            return True
        
        try:
            # Crop frame to bbox
            x1, y1, x2, y2 = map(int, bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return True  # Can't validate - accept by default
            
            crop = frame[y1:y2, x1:x2]
            
            if crop.size == 0:
                return True  # Can't validate - accept by default
            
            # Run depth inference
            with torch.no_grad():
                depth = self.model.infer_image(crop)
            
            # Calculate depth statistics
            depth_min = depth.min()
            depth_max = depth.max()
            depth_range = depth_max - depth_min
            depth_median = np.median(depth)
            
            # Test 1: Depth variation (rejects flat surfaces like stains/patches)
            if depth_range < self.MIN_DEPTH_VARIATION * (depth_max + 1e-6):
                logger.info(f"[DepthValidator] REJECTED: Insufficient depth variation ({depth_range:.3f})")
                print(f"[DepthValidator] ❌ Detection rejected: Flat surface (likely stain/patch)")
                return False
            
            # Test 2: Depression pattern (rejects bumps - should have pixels below median)
            pixels_below_median = np.sum(depth < depth_median)
            total_pixels = depth.size
            depression_ratio = pixels_below_median / total_pixels
            
            if depression_ratio < self.MIN_DEPRESSION_RATIO:
                logger.info(f"[DepthValidator] REJECTED: Not a depression (ratio: {depression_ratio:.2f})")
                print(f"[DepthValidator] ❌ Detection rejected: Raised surface (likely bump)")
                return False
            
            # Test 3: Check for extreme outliers (rejects shadows with inconsistent depth)
            depth_std = np.std(depth)
            outlier_threshold = depth_median + 3 * depth_std
            outlier_ratio = np.sum(depth > outlier_threshold) / total_pixels
            
            if outlier_ratio > 0.3:  # More than 30% outliers = likely noise
                logger.info(f"[DepthValidator] REJECTED: Noisy depth (outlier ratio: {outlier_ratio:.2f})")
                print(f"[DepthValidator] ❌ Detection rejected: Inconsistent depth (likely shadow)")
                return False
            
            # Passed all tests
            logger.info(f"[DepthValidator] VALIDATED: Depth variation={depth_range:.3f}, Depression={depression_ratio:.2f}")
            print(f"[DepthValidator] ✅ Detection validated: True pothole (depth confirmed)")
            return True
            
        except Exception as e:
            logger.exception(f"[DepthValidator] Exception in validation: {e}")
            # On error, accept detection (fail-safe behavior)
            return True


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("depth_utils.py - Module Test")
    print("=" * 70)
    
    # Test initialization
    print("\n[Test 1] Initialize DepthValidator")
    validator = DepthValidator()
    
    if validator.enabled:
        print("  ✅ DepthValidator initialized successfully")
        print(f"  Device: {validator.device}")
        
        # Test with dummy image
        print("\n[Test 2] Generate dummy heatmap")
        dummy_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        dummy_bbox = (500, 400, 700, 600)
        
        heatmap = validator.get_heatmap(dummy_frame, dummy_bbox)
        
        if heatmap is not None:
            print(f"  ✅ Heatmap generated: {heatmap.shape}")
        else:
            print("  ❌ Heatmap generation failed")
        
        # Test validation
        print("\n[Test 3] Validate dummy detection")
        is_valid = validator.is_valid_pothole(dummy_frame, dummy_bbox)
        print(f"  Validation result: {'✅ VALID' if is_valid else '❌ INVALID'}")
    else:
        print("  ⚠️ DepthValidator not enabled (missing model or library)")
    
    print("\n" + "=" * 70)
