"""
database_manager.py - SQLite Database Manager for Detection Storage

This module provides persistent storage for pothole detections using SQLite.
Designed to handle missing GPS data gracefully (NULL values allowed).

Features:
- Lightweight SQLite database (no external server needed)
- Nullable GPS coordinates (lat/lon can be None)
- Thread-safe connection handling
- Automatic table creation

Usage:
    from src.database_manager import DatabaseManager
    
    db = DatabaseManager("detections.db")
    db.insert_detection(
        track_id=5,
        timestamp="2026-01-05 12:30:45",
        latitude=None,  # GPS not available
        longitude=None,
        severity_score=78.5
    )
"""

import sqlite3
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Raised when database operations fail."""
    pass


class DatabaseManager:
    """
    SQLite database manager for pothole detection storage.
    
    This class handles:
    - Database creation and connection management
    - Detection record insertion with nullable GPS fields
    - Query operations for reporting
    - Snapshot image path storage
    
    Table Schema (detections) - Updated for Graduation Project:
    - id: INTEGER PRIMARY KEY AUTOINCREMENT
    - track_id: INTEGER NOT NULL
    - timestamp: TEXT NOT NULL
    - latitude: REAL (NULLABLE) - Can be NULL if GPS unavailable
    - longitude: REAL (NULLABLE) - Can be NULL if GPS unavailable
    - severity_score: REAL NOT NULL
    - priority_level: TEXT
    - risk_label: TEXT - Human readable (Low/Medium/High)
    - circularity: REAL - Shape irregularity index
    - relative_area: REAL - IPM-normalized area ratio
    - image_path: TEXT (NULLABLE) - Path to snapshot .jpg
    - heatmap_path: TEXT (NULLABLE) - Path to depth heatmap .jpg
    - created_at: TEXT (auto-generated)
    
    Attributes:
        db_path (Path): Path to SQLite database file
        snapshots_dir (Path): Directory for saving snapshot images
    """
    
    # SQL for table creation - Updated schema
    CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        severity_score REAL NOT NULL,
        priority_level TEXT,
        risk_label TEXT,
        circularity REAL,
        relative_area REAL,
        image_path TEXT,
        heatmap_path TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
    
    # SQL for index creation (improves query performance)
    CREATE_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS idx_track_id ON detections(track_id);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON detections(timestamp);
    CREATE INDEX IF NOT EXISTS idx_severity ON detections(severity_score);
    """
    
    def __init__(self, db_path: str = "detections.db", snapshots_dir: Optional[str] = None):
        """
        Initialize the DatabaseManager.
        
        Args:
            db_path: Path to SQLite database file (created if not exists)
            snapshots_dir: Directory for snapshot images (default: data/snapshots/)
            
        Raises:
            DatabaseError: If database initialization fails
        """
        self.db_path = Path(db_path)
        
        # Set snapshots directory
        if snapshots_dir:
            self.snapshots_dir = Path(snapshots_dir)
        else:
            # Default to data/snapshots/ relative to each run directory
            self.snapshots_dir = self.db_path.parent / "data" / "snapshots"
        
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Database initialized: {self.db_path}")
        logger.info(f"Snapshots directory: {self.snapshots_dir}")
        
        print(f"[Database] Initialized: {self.db_path}")
        print(f"[Database] Snapshots dir: {self.snapshots_dir}")
    
    def _init_database(self) -> None:
        """
        Initialize database with required tables and indexes.
        
        Raises:
            DatabaseError: If initialization fails
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create detections table
                cursor.execute(self.CREATE_TABLE_SQL)
                
                # Create indexes
                cursor.executescript(self.CREATE_INDEX_SQL)
                
                conn.commit()
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    def save_snapshot(
        self,
        frame: "np.ndarray",
        track_id: int,
        bbox: Optional[tuple] = None,
        quality: int = 90
    ) -> str:
        """
        Save a snapshot image of a pothole detection.
        
        Args:
            frame: BGR image (full frame or cropped)
            track_id: Track ID for filename
            bbox: Optional (x1, y1, x2, y2) to crop around pothole
            quality: JPEG quality (1-100)
            
        Returns:
            Path to saved image file
        """
        import cv2
        from datetime import datetime
        
        try:
            # Ensure snapshots directory exists
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate frame
            if frame is None or frame.size == 0:
                logger.error(f"Invalid frame for track {track_id}: frame is None or empty")
                return ""
            
            # Crop if bbox provided
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox)
                # Add padding
                pad = 20
                h, w = frame.shape[:2]
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(w, x2 + pad)
                y2 = min(h, y2 + pad)
                
                # Validate crop dimensions
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid crop dimensions for track {track_id}: ({x1},{y1})-({x2},{y2})")
                    snapshot = frame
                else:
                    snapshot = frame[y1:y2, x1:x2]
            else:
                snapshot = frame
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pothole_{track_id}_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), snapshot, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if success:
                logger.debug(f"Snapshot saved: {filename} ({snapshot.shape[1]}x{snapshot.shape[0]})")
                print(f"[Snapshot] Saved: {filename}")
            else:
                logger.error(f"cv2.imwrite failed for {filename} - check disk space and permissions")
                print(f"[Snapshot] ERROR: Failed to save {filename}")
            
            return str(filepath) if success else ""
            
        except Exception as e:
            logger.exception(f"Exception while saving snapshot for track {track_id}: {e}")
            print(f"[Snapshot] ERROR: Exception saving pothole_{track_id}: {e}")
            return ""
    
    def save_heatmap(
        self,
        heatmap: "np.ndarray",
        track_id: int,
        quality: int = 90
    ) -> str:
        """
        Save a depth heatmap image for a pothole detection.
        
        Args:
            heatmap: BGR heatmap image from depth analysis
            track_id: Track ID for filename
            quality: JPEG quality (1-100)
            
        Returns:
            Path to saved heatmap file
        """
        import cv2
        from datetime import datetime
        
        try:
            # Ensure snapshots directory exists
            self.snapshots_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate heatmap
            if heatmap is None or heatmap.size == 0:
                logger.error(f"Invalid heatmap for track {track_id}: heatmap is None or empty")
                return ""
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_{track_id}_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename
            
            # Save image
            success = cv2.imwrite(str(filepath), heatmap, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            if success:
                logger.debug(f"Heatmap saved: {filename} ({heatmap.shape[1]}x{heatmap.shape[0]})")
                print(f"[Heatmap] Saved: {filename}")
            else:
                logger.error(f"cv2.imwrite failed for {filename} - check disk space and permissions")
                print(f"[Heatmap] ERROR: Failed to save {filename}")
            
            return str(filepath) if success else ""
            
        except Exception as e:
            logger.exception(f"Exception while saving heatmap for track {track_id}: {e}")
            print(f"[Heatmap] ERROR: Exception saving heatmap_{track_id}: {e}")
            return ""
    
    @contextmanager
    def _get_connection(self):
        """
        Context manager for database connections.
        
        Yields:
            sqlite3.Connection: Database connection
            
        Note:
            Connection is automatically closed after use.
            Uses WAL mode for better concurrent access.
        """
        conn = None
        try:
            # Connect with explicit parameters for Windows file locking
            conn = sqlite3.connect(
                str(self.db_path),
                timeout=10.0,  # Wait up to 10 seconds if database is locked
                check_same_thread=False  # Allow multi-threaded access
            )
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            
            # Enable WAL mode for better concurrent access (reduces file locking)
            conn.execute("PRAGMA journal_mode=WAL")
            
            yield conn
        finally:
            if conn:
                conn.close()
    
    def insert_detection(
        self,
        track_id: int,
        timestamp: str,
        severity_score: float,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        priority_level: Optional[str] = None,
        risk_label: Optional[str] = None,
        circularity: Optional[float] = None,
        relative_area: Optional[float] = None,
        image_path: Optional[str] = None,
        heatmap_path: Optional[str] = None
    ) -> int:
        """
        Insert a detection record into the database.
        
        This method handles nullable fields gracefully.
        If latitude/longitude is None, it will be stored as SQL NULL.
        
        Args:
            track_id: Unique tracking ID for the detected pothole
            timestamp: ISO format timestamp string
            severity_score: Severity score (0-100)
            latitude: GPS latitude (None if unavailable)
            longitude: GPS longitude (None if unavailable)
            priority_level: Priority classification (LOW/MEDIUM/HIGH)
            risk_label: Human-readable risk label (Low/Medium/High)
            circularity: Shape irregularity index (0-1)
            relative_area: IPM-normalized area ratio
            image_path: Path to snapshot image file
            
        Returns:
            int: ID of the inserted record
            
        Raises:
            DatabaseError: If insertion fails
            
        Example:
            db.insert_detection(
                track_id=5,
                timestamp="2026-01-05T12:30:45",
                latitude=41.0082,
                longitude=28.9784,
                severity_score=78.5,
                priority_level="HIGH",
                risk_label="High",
                circularity=0.45,
                relative_area=0.023,
                image_path="data/snapshots/pothole_5.jpg"
            )
        """
        sql = """
        INSERT INTO detections (
            track_id, timestamp, latitude, longitude, 
            severity_score, priority_level, risk_label,
            circularity, relative_area, image_path, heatmap_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    track_id,
                    timestamp,
                    latitude,
                    longitude,
                    severity_score,
                    priority_level,
                    risk_label,
                    circularity,
                    relative_area,
                    image_path,
                    heatmap_path
                ))
                conn.commit()
                return cursor.lastrowid
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to insert detection: {e}")
    
    def insert_many_detections(
        self,
        detections: List[Dict[str, Any]]
    ) -> int:
        """
        Batch insert multiple detection records.
        
        Args:
            detections: List of detection dictionaries with keys:
                - track_id, timestamp, severity_score
                - latitude (optional), longitude (optional)
                - priority_level (optional)
                
        Returns:
            int: Number of records inserted
            
        Raises:
            DatabaseError: If batch insertion fails
        """
        sql = """
        INSERT INTO detections (
            track_id, timestamp, latitude, longitude, 
            severity_score, priority_level
        ) VALUES (?, ?, ?, ?, ?, ?)
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                records = [
                    (
                        d['track_id'],
                        d['timestamp'],
                        d.get('latitude'),  # None if not present
                        d.get('longitude'),  # None if not present
                        d['severity_score'],
                        d.get('priority_level')
                    )
                    for d in detections
                ]
                
                cursor.executemany(sql, records)
                conn.commit()
                return cursor.rowcount
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to batch insert detections: {e}")
    
    def get_all_detections(self) -> List[Dict[str, Any]]:
        """
        Retrieve all detection records.
        
        Returns:
            List of detection dictionaries
        """
        sql = "SELECT * FROM detections ORDER BY timestamp DESC"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve detections: {e}")
    
    def get_detections_by_track_id(self, track_id: int) -> List[Dict[str, Any]]:
        """
        Retrieve all detections for a specific track ID.
        
        Args:
            track_id: Track ID to filter by
            
        Returns:
            List of detection dictionaries
        """
        sql = "SELECT * FROM detections WHERE track_id = ? ORDER BY timestamp"
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (track_id,))
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve detections: {e}")
    
    def get_detections_with_gps(self) -> List[Dict[str, Any]]:
        """
        Retrieve only detections that have GPS coordinates.
        
        Returns:
            List of detection dictionaries with valid GPS data
        """
        sql = """
        SELECT * FROM detections 
        WHERE latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY timestamp DESC
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve GPS detections: {e}")
    
    def get_detections_without_gps(self) -> List[Dict[str, Any]]:
        """
        Retrieve detections that are missing GPS coordinates.
        
        Returns:
            List of detection dictionaries without GPS data
        """
        sql = """
        SELECT * FROM detections 
        WHERE latitude IS NULL OR longitude IS NULL
        ORDER BY timestamp DESC
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve non-GPS detections: {e}")
    
    def get_high_priority_detections(self) -> List[Dict[str, Any]]:
        """
        Retrieve all HIGH priority detections.
        
        Returns:
            List of HIGH priority detection dictionaries
        """
        sql = """
        SELECT * FROM detections 
        WHERE priority_level = 'HIGH'
        ORDER BY severity_score DESC
        """
        
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to retrieve high priority detections: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with statistics:
            - total_records: Total number of detections
            - with_gps: Detections with GPS coordinates
            - without_gps: Detections without GPS coordinates
            - by_priority: Count by priority level
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Total records
                cursor.execute("SELECT COUNT(*) FROM detections")
                total = cursor.fetchone()[0]
                
                # With GPS
                cursor.execute("""
                    SELECT COUNT(*) FROM detections 
                    WHERE latitude IS NOT NULL AND longitude IS NOT NULL
                """)
                with_gps = cursor.fetchone()[0]
                
                # By priority
                cursor.execute("""
                    SELECT priority_level, COUNT(*) 
                    FROM detections 
                    GROUP BY priority_level
                """)
                by_priority = {row[0]: row[1] for row in cursor.fetchall()}
                
                return {
                    'total_records': total,
                    'with_gps': with_gps,
                    'without_gps': total - with_gps,
                    'by_priority': by_priority
                }
                
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get statistics: {e}")
    
    def clear_all(self) -> int:
        """
        Delete all records from the detections table.
        
        Returns:
            int: Number of records deleted
            
        Warning:
            This operation cannot be undone!
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM detections")
                conn.commit()
                deleted = cursor.rowcount
                logger.info(f"Database cleared: {deleted} records deleted")
                print(f"[Veritabani] {deleted} kayit silindi")
                
                # Vacuum to release disk space and close connections properly
                conn.execute("VACUUM")
                
                return deleted
                
        except sqlite3.Error as e:
            logger.error(f"Failed to clear database: {e}")
            raise DatabaseError(f"Veritabani temizlenemedi: {e}")
    
    def clear_all_data(self) -> int:
        """
        Clear all detection data and snapshots (Turkish presentation mode).
        
        This method:
        1. Deletes all records from detections table
        2. Clears snapshot images from snapshots directory
        
        Returns:
            int: Number of records deleted
            
        Warning:
            Bu islem geri alinamaz!
        """
        # Clear database records
        deleted = self.clear_all()
        
        # Clear snapshots directory
        if self.snapshots_dir.exists():
            snapshot_count = 0
            for file in self.snapshots_dir.glob("*.jpg"):
                try:
                    file.unlink()
                    snapshot_count += 1
                except Exception as e:
                    logger.warning(f"Could not delete snapshot {file.name}: {e}")
            
            if snapshot_count > 0:
                logger.info(f"Deleted {snapshot_count} snapshot images")
                print(f"[Veritabani] {snapshot_count} anlik goruntu temizlendi: {self.snapshots_dir}")
        
        return deleted
        
        return deleted
    
    def close(self) -> None:
        """
        Close any open connections.
        
        Note: With context manager pattern, connections are auto-closed.
        This method is provided for explicit cleanup if needed.
        """
        # Connections are managed via context manager
        # This method exists for API completeness
        pass
    
    def __repr__(self) -> str:
        return f"DatabaseManager(db_path='{self.db_path}')"


# =============================================================================
# Module Self-Test
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    print("=" * 60)
    print("database_manager.py - Module Test")
    print("=" * 60)
    
    # Use test database
    test_db_path = Path("output/test_detections.db")
    
    try:
        # Initialize database
        print("\n[Test 1] Initialize Database:")
        db = DatabaseManager(str(test_db_path))
        print(f"  {db}")
        
        # Clear any existing data
        db.clear_all()
        
        # Test insertion WITH GPS
        print("\n[Test 2] Insert detection WITH GPS:")
        id1 = db.insert_detection(
            track_id=1,
            timestamp="2026-01-05T12:30:45",
            latitude=41.0082,
            longitude=28.9784,
            severity_score=78.5,
            priority_level="HIGH"
        )
        print(f"  Inserted record ID: {id1}")
        print(f"  GPS: (41.0082, 28.9784)")
        
        # Test insertion WITHOUT GPS (None values)
        print("\n[Test 3] Insert detection WITHOUT GPS:")
        id2 = db.insert_detection(
            track_id=2,
            timestamp="2026-01-05T12:30:46",
            latitude=None,  # GPS not available
            longitude=None,  # GPS not available
            severity_score=45.2,
            priority_level="MEDIUM"
        )
        print(f"  Inserted record ID: {id2}")
        print(f"  GPS: (None, None) - Stored as SQL NULL")
        
        # Test batch insertion
        print("\n[Test 4] Batch insert:")
        batch = [
            {'track_id': 3, 'timestamp': '2026-01-05T12:30:47', 
             'severity_score': 22.0, 'priority_level': 'LOW'},
            {'track_id': 4, 'timestamp': '2026-01-05T12:30:48',
             'latitude': 40.9923, 'longitude': 29.0245,
             'severity_score': 89.5, 'priority_level': 'HIGH'}
        ]
        count = db.insert_many_detections(batch)
        print(f"  Inserted {count} records")
        
        # Get statistics
        print("\n[Test 5] Database Statistics:")
        stats = db.get_statistics()
        print(f"  Total records: {stats['total_records']}")
        print(f"  With GPS: {stats['with_gps']}")
        print(f"  Without GPS: {stats['without_gps']}")
        print(f"  By Priority: {stats['by_priority']}")
        
        # Query detections
        print("\n[Test 6] Query Detections:")
        all_det = db.get_all_detections()
        for det in all_det:
            gps = f"({det['latitude']}, {det['longitude']})" if det['latitude'] else "NO GPS"
            print(f"  ID:{det['track_id']} | {det['priority_level']} | {gps}")
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        sys.exit(1)

