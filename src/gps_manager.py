"""
gps_manager.py - GPS Data Manager (Strict Mode - No Fake Data)

This module provides GPS coordinate retrieval for video frames.
STRICT POLICY: No simulated or mock coordinates are generated.

Design Philosophy:
- If real GPS data is available (from file), use it.
- If no GPS data is available, return (None, None) explicitly.
- NEVER generate fake coordinates.

Supported GPS Data Formats:
1. CSV: frame_idx,latitude,longitude
2. JSON: [{"frame": 0, "lat": 41.0, "lon": 28.9}, ...]
3. NMEA (future): Standard GPS sentence format

Usage:
    # Without GPS file - returns (None, None) for all frames
    gps = GPSManager()
    lat, lon = gps.get_location(frame_idx=100)
    # Returns: (None, None)
    
    # With GPS file - returns real coordinates
    gps = GPSManager(gps_file_path="gps_data.csv")
    lat, lon = gps.get_location(frame_idx=100)
    # Returns: (41.0082, 28.9784) or (None, None) if frame not in data
"""

import csv
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass


@dataclass
class GPSPoint:
    """Single GPS coordinate point."""
    latitude: float
    longitude: float
    frame_idx: Optional[int] = None
    timestamp: Optional[str] = None


class GPSDataError(Exception):
    """Raised when GPS data loading or parsing fails."""
    pass


class GPSManager:
    """
    GPS Data Manager for video frame geolocation.
    
    STRICT MODE: This class NEVER generates fake or simulated coordinates.
    If GPS data is not available, it returns (None, None) explicitly.
    
    Attributes:
        gps_file_path (Path or None): Path to GPS data file
        is_loaded (bool): Whether GPS data was successfully loaded
        frame_count (int): Number of frames with GPS data
        
    Supported File Formats:
        - CSV: frame_idx,latitude,longitude (with optional header)
        - JSON: Array of objects with frame/lat/lon keys
    """
    
    def __init__(self, gps_file_path: Optional[str] = None):
        """
        Initialize the GPSManager.
        
        Args:
            gps_file_path: Optional path to GPS data file.
                          If None, all get_location() calls return (None, None).
                          
        Note:
            This constructor does NOT generate fake data.
            If no file is provided, GPS will be unavailable.
        """
        self.gps_file_path: Optional[Path] = None
        self._gps_data: Dict[int, GPSPoint] = {}  # frame_idx -> GPSPoint
        self._is_loaded: bool = False
        self._load_error: Optional[str] = None
        
        if gps_file_path is not None:
            self.gps_file_path = Path(gps_file_path)
            self._load_gps_data()
        else:
            print("[GPSManager] No GPS file provided. GPS data unavailable.")
            print("[GPSManager] All get_location() calls will return (None, None)")
    
    def _load_gps_data(self) -> None:
        """
        Load GPS data from file.
        
        Automatically detects format (CSV or JSON) based on file extension.
        
        Raises:
            GPSDataError: If file cannot be loaded or parsed
        """
        if not self.gps_file_path.exists():
            self._load_error = f"GPS file not found: {self.gps_file_path}"
            print(f"[GPSManager] WARNING: {self._load_error}")
            print("[GPSManager] GPS data unavailable. Returning (None, None) for all frames.")
            return
        
        suffix = self.gps_file_path.suffix.lower()
        
        try:
            if suffix == '.csv':
                self._load_csv()
            elif suffix == '.json':
                self._load_json()
            else:
                # Try CSV as default
                print(f"[GPSManager] Unknown extension '{suffix}', attempting CSV parse...")
                self._load_csv()
            
            self._is_loaded = True
            print(f"[GPSManager] Loaded {len(self._gps_data)} GPS points from {self.gps_file_path.name}")
            
        except Exception as e:
            self._load_error = f"Failed to parse GPS file: {e}"
            print(f"[GPSManager] WARNING: {self._load_error}")
            print("[GPSManager] GPS data unavailable. Returning (None, None) for all frames.")
    
    def _load_csv(self) -> None:
        """
        Load GPS data from CSV file.
        
        Expected format:
            frame_idx,latitude,longitude
            0,41.0082,28.9784
            1,41.0083,28.9785
            ...
            
        Or with header:
            frame,lat,lon
            0,41.0082,28.9784
        """
        with open(self.gps_file_path, 'r', encoding='utf-8') as f:
            # Detect if first row is header
            first_line = f.readline().strip()
            f.seek(0)
            
            # Check if first line contains non-numeric values (header)
            has_header = False
            try:
                parts = first_line.split(',')
                float(parts[0])  # Try parsing first value as number
            except ValueError:
                has_header = True
            
            reader = csv.reader(f)
            
            if has_header:
                next(reader)  # Skip header
            
            for row in reader:
                if len(row) < 3:
                    continue
                
                try:
                    frame_idx = int(row[0])
                    latitude = float(row[1])
                    longitude = float(row[2])
                    
                    self._gps_data[frame_idx] = GPSPoint(
                        latitude=latitude,
                        longitude=longitude,
                        frame_idx=frame_idx
                    )
                except (ValueError, IndexError):
                    continue  # Skip malformed rows
    
    def _load_json(self) -> None:
        """
        Load GPS data from JSON file.
        
        Expected format:
            [
                {"frame": 0, "lat": 41.0082, "lon": 28.9784},
                {"frame": 1, "lat": 41.0083, "lon": 28.9785},
                ...
            ]
            
        Alternative keys: frame_idx, latitude, longitude
        """
        with open(self.gps_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise GPSDataError("JSON must contain an array of GPS points")
        
        for point in data:
            # Support multiple key naming conventions
            frame_idx = point.get('frame') or point.get('frame_idx')
            latitude = point.get('lat') or point.get('latitude')
            longitude = point.get('lon') or point.get('longitude') or point.get('lng')
            
            if frame_idx is not None and latitude is not None and longitude is not None:
                self._gps_data[int(frame_idx)] = GPSPoint(
                    latitude=float(latitude),
                    longitude=float(longitude),
                    frame_idx=int(frame_idx),
                    timestamp=point.get('timestamp')
                )
    
    def get_location(self, frame_idx: int) -> Tuple[Optional[float], Optional[float]]:
        """
        Get GPS coordinates for a specific video frame.
        
        STRICT MODE BEHAVIOR:
        - If GPS data is loaded AND frame exists: Returns (latitude, longitude)
        - If GPS data is loaded BUT frame not found: Returns (None, None)
        - If NO GPS data loaded: Returns (None, None)
        - NEVER generates fake/simulated coordinates
        
        Args:
            frame_idx: Video frame index (0-based)
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if unavailable
            
        Example:
            gps = GPSManager("gps_data.csv")
            
            lat, lon = gps.get_location(100)
            if lat is not None:
                print(f"Location: {lat}, {lon}")
            else:
                print("GPS data not available for this frame")
        """
        # No GPS data loaded - return None explicitly
        if not self._is_loaded:
            return (None, None)
        
        # GPS data loaded but frame not found
        if frame_idx not in self._gps_data:
            return (None, None)
        
        # Return real coordinates
        point = self._gps_data[frame_idx]
        return (point.latitude, point.longitude)
    
    def get_nearest_location(
        self, 
        frame_idx: int, 
        max_distance: int = 30
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Get GPS coordinates for nearest frame within max_distance.
        
        Useful when GPS data is sampled (e.g., every 30 frames).
        
        Args:
            frame_idx: Target video frame index
            max_distance: Maximum frame distance to search
            
        Returns:
            Tuple of (latitude, longitude) or (None, None) if none found
        """
        if not self._is_loaded or not self._gps_data:
            return (None, None)
        
        # Check exact match first
        if frame_idx in self._gps_data:
            point = self._gps_data[frame_idx]
            return (point.latitude, point.longitude)
        
        # Search nearby frames
        best_distance = max_distance + 1
        best_point = None
        
        for stored_frame, point in self._gps_data.items():
            distance = abs(stored_frame - frame_idx)
            if distance < best_distance:
                best_distance = distance
                best_point = point
        
        if best_point is not None and best_distance <= max_distance:
            return (best_point.latitude, best_point.longitude)
        
        return (None, None)
    
    def has_gps_data(self) -> bool:
        """
        Check if GPS data is available.
        
        Returns:
            True if GPS data was loaded successfully, False otherwise
        """
        return self._is_loaded and len(self._gps_data) > 0
    
    def get_coverage_info(self) -> Dict[str, Any]:
        """
        Get information about GPS data coverage.
        
        Returns:
            Dictionary with coverage statistics:
            - is_loaded: Whether data was loaded
            - total_points: Number of GPS points
            - frame_range: (min_frame, max_frame) or None
            - file_path: Source file path or None
        """
        if not self._is_loaded or not self._gps_data:
            return {
                'is_loaded': False,
                'total_points': 0,
                'frame_range': None,
                'file_path': str(self.gps_file_path) if self.gps_file_path else None,
                'load_error': self._load_error
            }
        
        frames = list(self._gps_data.keys())
        return {
            'is_loaded': True,
            'total_points': len(self._gps_data),
            'frame_range': (min(frames), max(frames)),
            'file_path': str(self.gps_file_path),
            'load_error': None
        }
    
    def get_all_points(self) -> List[GPSPoint]:
        """
        Get all loaded GPS points.
        
        Returns:
            List of GPSPoint objects, empty if no data loaded
        """
        return list(self._gps_data.values())
    
    @property
    def is_available(self) -> bool:
        """Check if GPS data is available for use."""
        return self._is_loaded
    
    @property
    def frame_count(self) -> int:
        """Number of frames with GPS data."""
        return len(self._gps_data)
    
    def __repr__(self) -> str:
        if self._is_loaded:
            return f"GPSManager(loaded={self.frame_count} points from '{self.gps_file_path}')"
        elif self.gps_file_path:
            return f"GPSManager(file='{self.gps_file_path}', loaded=False)"
        else:
            return "GPSManager(no_file=True, returns_none=True)"


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    import sys
    import tempfile
    from pathlib import Path
    
    print("=" * 60)
    print("gps_manager.py - Module Test (STRICT MODE)")
    print("=" * 60)
    
    # Test 1: No GPS file provided
    print("\n[Test 1] GPSManager without file:")
    gps_no_file = GPSManager()
    lat, lon = gps_no_file.get_location(0)
    print(f"  get_location(0) = ({lat}, {lon})")
    assert lat is None and lon is None, "Should return (None, None)"
    print("  PASS: Returns (None, None) as expected")
    
    # Test 2: GPS file does not exist
    print("\n[Test 2] GPSManager with non-existent file:")
    gps_missing = GPSManager("nonexistent_gps.csv")
    lat, lon = gps_missing.get_location(0)
    print(f"  get_location(0) = ({lat}, {lon})")
    assert lat is None and lon is None, "Should return (None, None)"
    print("  PASS: Returns (None, None) as expected")
    
    # Test 3: Create temp CSV and load
    print("\n[Test 3] GPSManager with CSV file:")
    
    # Create temporary CSV file
    csv_content = """frame_idx,latitude,longitude
0,41.0082,28.9784
10,41.0083,28.9785
20,41.0084,28.9786
30,41.0085,28.9787
"""
    temp_csv = Path("output/test_gps.csv")
    temp_csv.parent.mkdir(exist_ok=True)
    temp_csv.write_text(csv_content)
    
    gps_csv = GPSManager(str(temp_csv))
    print(f"  {gps_csv}")
    print(f"  Coverage: {gps_csv.get_coverage_info()}")
    
    # Test exact match
    lat, lon = gps_csv.get_location(10)
    print(f"  get_location(10) = ({lat}, {lon})")
    assert lat == 41.0083 and lon == 28.9785, "Should match CSV data"
    print("  PASS: Exact frame match works")
    
    # Test frame not in data
    lat, lon = gps_csv.get_location(5)
    print(f"  get_location(5) = ({lat}, {lon})")
    assert lat is None and lon is None, "Frame 5 not in data"
    print("  PASS: Returns (None, None) for missing frame")
    
    # Test nearest location
    lat, lon = gps_csv.get_nearest_location(12, max_distance=5)
    print(f"  get_nearest_location(12, max=5) = ({lat}, {lon})")
    assert lat == 41.0083, "Should find frame 10"
    print("  PASS: Nearest frame lookup works")
    
    # Test 4: Create temp JSON and load
    print("\n[Test 4] GPSManager with JSON file:")
    
    json_content = """[
        {"frame": 0, "lat": 40.0, "lon": 29.0},
        {"frame": 100, "lat": 40.1, "lon": 29.1}
    ]"""
    temp_json = Path("output/test_gps.json")
    temp_json.write_text(json_content)
    
    gps_json = GPSManager(str(temp_json))
    lat, lon = gps_json.get_location(100)
    print(f"  get_location(100) = ({lat}, {lon})")
    assert lat == 40.1 and lon == 29.1, "Should match JSON data"
    print("  PASS: JSON loading works")
    
    # Test 5: Verify NO fake data generation
    print("\n[Test 5] Verify STRICT MODE (no fake data):")
    gps_strict = GPSManager()
    results = []
    for i in range(10):
        lat, lon = gps_strict.get_location(i)
        results.append((lat, lon))
    
    all_none = all(lat is None and lon is None for lat, lon in results)
    print(f"  10 calls without GPS file: all returned (None, None) = {all_none}")
    assert all_none, "STRICT MODE: Must never generate fake data"
    print("  PASS: STRICT MODE verified - no fake data generated")
    
    print("\n" + "=" * 60)
    print("All tests passed! STRICT MODE confirmed.")
    print("=" * 60)

