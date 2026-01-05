"""
main.py - Entry Point for Offline Video Processing Pipeline

Autonomous Road Damage Assessment System
========================================
This script orchestrates the complete pothole detection workflow:
1. Load TensorRT-optimized YOLO model
2. Process input video frame-by-frame with tracking
3. Calculate severity metrics (circularity, severity score)
4. Generate annotated output video
5. Export CSV report with aggregated statistics

Usage:
    python -m src.main
    
    Or with custom paths:
    python -m src.main --input path/to/video.mp4 --output runs/detect/exp1/

Architecture: Headless (No GUI) - Batch Processing Only
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import PotholeDetector, ModelLoadError
from src.video_processor import (
    VideoProcessor,
    VideoProcessingError,
    generate_csv_report
)


# =============================================================================
# CONFIGURATION - Default Paths
# =============================================================================

# Model path (TensorRT engine file)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "best1.engine"

# Input video (user should provide or place video here)
DEFAULT_INPUT_VIDEO = PROJECT_ROOT / "data" / "sample_video.mp4"

# Output directory for results
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "detect" / "experiment_1"

# Detection parameters
DEFAULT_CONFIDENCE = 0.5
DEFAULT_IOU_THRESHOLD = 0.5
DEFAULT_TRACKER = "bytetrack.yaml"

# GPS file (optional - None means GPS disabled, strict mode)
DEFAULT_GPS_FILE = None  # Set to path like "data/gps_log.csv" if available


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_banner():
    """Print application banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║       AUTONOMOUS ROAD DAMAGE ASSESSMENT SYSTEM                   ║
║       Offline Video Processing Pipeline v1.0                     ║
╠══════════════════════════════════════════════════════════════════╣
║  Model: YOLOv8 Instance Segmentation (TensorRT)                  ║
║  Mode:  Headless / Batch Processing                              ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_config(args):
    """Print current configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Input Video:  {args.input}")
    print(f"  Output Dir:   {args.output}")
    print(f"  Model:        {args.model}")
    print(f"  Confidence:   {args.conf}")
    print(f"  IoU:          {args.iou}")
    print(f"  Tracker:      {args.tracker}")
    print(f"  GPS File:     {args.gps if args.gps else 'None (disabled)'}")
    print("=" * 60)


def validate_input_video(video_path: Path) -> bool:
    """
    Validate that input video exists.
    
    Args:
        video_path: Path to input video file
        
    Returns:
        True if valid, False otherwise
    """
    if not video_path.exists():
        print(f"\n[ERROR] Input video not found: {video_path}")
        print("\n  Please provide a valid video file:")
        print("    Option 1: Place your video at the default location:")
        print(f"              {video_path}")
        print("    Option 2: Use --input argument to specify path:")
        print("              python -m src.main --input path/to/your/video.mp4")
        print("\n  Supported formats: .mp4, .avi, .mov, .mkv")
        return False
    
    # Check file extension
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv'}
    if video_path.suffix.lower() not in valid_extensions:
        print(f"\n[WARNING] Unusual video extension: {video_path.suffix}")
        print(f"  Supported formats: {', '.join(valid_extensions)}")
    
    return True


def validate_model(model_path: Path) -> bool:
    """
    Validate that model file exists.
    
    Args:
        model_path: Path to .engine model file
        
    Returns:
        True if valid, False otherwise
    """
    if not model_path.exists():
        print(f"\n[ERROR] Model file not found: {model_path}")
        print("\n  Please ensure the TensorRT engine file exists.")
        print("  Expected location: best1.engine in project root")
        return False
    
    return True


def setup_output_directory(output_dir: Path) -> Path:
    """
    Create output directory structure.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        Path to output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[Setup] Output directory: {output_dir}")
    return output_dir


def generate_output_paths(output_dir: Path, input_video: Path) -> dict:
    """
    Generate output file paths.
    
    Args:
        output_dir: Output directory
        input_video: Input video path (for naming)
        
    Returns:
        Dictionary with output paths
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_stem = input_video.stem
    
    return {
        'video': output_dir / f"{video_stem}_annotated_{timestamp}.mp4",
        'csv': output_dir / "final_report.csv",
        'summary': output_dir / f"summary_{timestamp}.txt",
        'db': output_dir / "detections.db"
    }


def save_summary_report(output_path: Path, stats: dict, track_data: dict):
    """
    Save a human-readable summary report.
    
    Args:
        output_path: Path for summary file
        stats: Processing statistics
        track_data: Track data dictionary
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("POTHOLE DETECTION SUMMARY REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("PROCESSING STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Frames Processed:    {stats['frames_processed']}\n")
        f.write(f"  Total Detections:    {stats['total_detections']}\n")
        f.write(f"  Detections w/ GPS:   {stats.get('detections_with_gps', 0)}\n")
        f.write(f"  Unique Potholes:     {stats['unique_tracks']}\n")
        f.write("\n")
        
        f.write("PRIORITY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        f.write(f"  HIGH Priority:       {stats['high_priority_count']}\n")
        f.write(f"  MEDIUM Priority:     {stats['medium_priority_count']}\n")
        f.write(f"  LOW Priority:        {stats['low_priority_count']}\n")
        f.write("\n")
        
        f.write("INTEGRATION STATUS\n")
        f.write("-" * 40 + "\n")
        gps_status = "Enabled" if stats.get('gps_available', False) else "Disabled (strict mode)"
        db_status = "Enabled" if stats.get('db_enabled', False) else "Disabled"
        f.write(f"  GPS Integration:     {gps_status}\n")
        f.write(f"  Database Logging:    {db_status}\n")
        f.write("\n")
        
        if track_data:
            f.write("TOP 5 SEVERE POTHOLES\n")
            f.write("-" * 40 + "\n")
            
            # Sort by severity
            sorted_tracks = sorted(
                track_data.values(),
                key=lambda t: t.avg_severity,
                reverse=True
            )[:5]
            
            for i, track in enumerate(sorted_tracks, 1):
                f.write(f"  #{i} ID:{track.track_id:3d} | "
                       f"Severity: {track.avg_severity:5.1f} | "
                       f"Priority: {track.priority_level}\n")
        
        f.write("\n" + "=" * 60 + "\n")
    
    print(f"[Report] Summary saved to: {output_path}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Road Damage Assessment - Offline Video Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main
  python -m src.main --input video.mp4
  python -m src.main --input video.mp4 --output runs/detect/exp2/
  python -m src.main --conf 0.6 --iou 0.45
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=DEFAULT_INPUT_VIDEO,
        help=f"Input video path (default: {DEFAULT_INPUT_VIDEO})"
    )
    
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})"
    )
    
    parser.add_argument(
        '--model', '-m',
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (default: {DEFAULT_MODEL_PATH})"
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=DEFAULT_CONFIDENCE,
        help=f"Confidence threshold (default: {DEFAULT_CONFIDENCE})"
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=DEFAULT_IOU_THRESHOLD,
        help=f"IoU threshold (default: {DEFAULT_IOU_THRESHOLD})"
    )
    
    parser.add_argument(
        '--tracker',
        type=str,
        default=DEFAULT_TRACKER,
        help=f"Tracker config (default: {DEFAULT_TRACKER})"
    )
    
    parser.add_argument(
        '--gps',
        type=Path,
        default=DEFAULT_GPS_FILE,
        help="GPS data file path (CSV/JSON). If not provided, GPS is disabled (strict mode - no fake data)"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the video processing pipeline.
    
    Workflow:
    1. Parse arguments and validate inputs
    2. Initialize detector and processor
    3. Process video
    4. Generate reports
    """
    # Print banner
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Print configuration
    print_config(args)
    
    # =================================================================
    # PHASE 1: Validation
    # =================================================================
    print("\n[Phase 1] Validating inputs...")
    
    # Validate model
    if not validate_model(args.model):
        sys.exit(1)
    print(f"  [OK] Model found: {args.model.name}")
    
    # Validate input video
    if not validate_input_video(args.input):
        sys.exit(1)
    print(f"  [OK] Input video found: {args.input.name}")
    
    # Setup output directory
    output_dir = setup_output_directory(args.output)
    output_paths = generate_output_paths(output_dir, args.input)
    
    # =================================================================
    # PHASE 2: Initialization
    # =================================================================
    print("\n[Phase 2] Initializing components...")
    
    try:
        # Initialize detector
        print("  Loading YOLO model...")
        detector = PotholeDetector(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            tracker_config=args.tracker
        )
        print(f"  [OK] Detector initialized: {detector}")
        
        # Initialize video processor with GPS and DB integration
        print("  Setting up video processor...")
        gps_path = str(args.gps) if args.gps else None
        processor = VideoProcessor(
            input_path=args.input,
            output_path=output_paths['video'],
            detector=detector,
            gps_file_path=gps_path,
            db_path=str(output_paths['db'])
        )
        print("  [OK] Video processor ready")
        
        # GPS status
        if args.gps:
            if processor.gps_manager.is_available:
                coverage = processor.gps_manager.get_coverage_info()
                print(f"  [OK] GPS loaded: {coverage['total_points']} points")
            else:
                print(f"  [WARNING] GPS file specified but not loaded")
        else:
            print("  [INFO] GPS disabled (strict mode - no fake coordinates)")
        
    except ModelLoadError as e:
        print(f"\n[FATAL] Model loading failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FATAL] Initialization failed: {e}")
        sys.exit(1)
    
    # =================================================================
    # PHASE 3: Processing
    # =================================================================
    print("\n[Phase 3] Starting video processing...")
    print("  Mode: Headless (no display)")
    print("  Progress bar will show below:\n")
    
    try:
        # Run the main processing loop
        track_data = processor.process()
        
    except VideoProcessingError as e:
        print(f"\n[FATAL] Video processing failed: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Processing cancelled by user.")
        sys.exit(0)
    
    # =================================================================
    # PHASE 4: Reporting
    # =================================================================
    print("\n[Phase 4] Generating reports...")
    
    # Get summary statistics
    stats = processor.get_summary_stats()
    
    # Generate CSV report
    if track_data:
        generate_csv_report(track_data, output_paths['csv'])
    else:
        print("  [WARNING] No detections to report")
    
    # Generate summary report
    save_summary_report(output_paths['summary'], stats, track_data)
    
    # =================================================================
    # COMPLETION
    # =================================================================
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\n  Output Files:")
    print(f"    Video:   {output_paths['video']}")
    print(f"    CSV:     {output_paths['csv']}")
    print(f"    DB:      {output_paths['db']}")
    print(f"    Summary: {output_paths['summary']}")
    print(f"\n  Statistics:")
    print(f"    Frames:     {stats['frames_processed']}")
    print(f"    Detections: {stats['total_detections']}")
    print(f"    With GPS:   {stats.get('detections_with_gps', 0)}")
    print(f"    Tracks:     {stats['unique_tracks']}")
    print(f"\n  Priority Breakdown:")
    print(f"    HIGH:   {stats['high_priority_count']}")
    print(f"    MEDIUM: {stats['medium_priority_count']}")
    print(f"    LOW:    {stats['low_priority_count']}")
    print(f"\n  Integration Status:")
    print(f"    GPS:    {'Enabled' if stats.get('gps_available', False) else 'Disabled (strict mode)'}")
    print(f"    DB:     {'Enabled' if stats.get('db_enabled', False) else 'Disabled'}")
    print("\n" + "=" * 60)
    
    return 0


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

