"""
app.py - Autonomous Road Damage Assessment Dashboard (Optimized)

A professional Streamlit web application for pothole detection and mapping.
OPTIMIZED: No live preview during processing for maximum performance.

Features:
- High-performance video processing (no live preview overhead)
- FPS metrics display
- Post-processing video playback
- Interactive pothole mapping with Folium
- Real-time statistics and charts

Usage:
    streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from datetime import datetime
import time

# Page config must be first Streamlit command
st.set_page_config(
    page_title="Yol Hasari Degerlendirme Sistemi",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import project modules
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import PotholeDetector, ModelLoadError
from src.video_processor import VideoProcessor, VideoProcessingError, generate_csv_report
from src.config_loader import load_config, get_config
from src.logger import setup_logging, get_logger

# Initialize logging system
try:
    config = load_config()
    log_config = config.get('logging', {})
    setup_logging(
        log_dir=config['paths']['logs_dir'],
        log_level=log_config.get('level', 'INFO'),
        max_bytes=log_config.get('max_file_size_mb', 10) * 1024 * 1024,
        backup_count=log_config.get('backup_count', 5),
        console_output=log_config.get('console_logging', True),
        file_output=log_config.get('file_logging', True)
    )
except Exception as e:
    # Fallback if config loading fails
    setup_logging()
    print(f"Warning: Failed to load config, using defaults: {e}")

logger = get_logger(__name__)


# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================
def apply_custom_css():
    st.markdown("""
    <style>
    /* Main theme - Dark professional */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        padding: 1rem;
        font-family: 'Segoe UI', sans-serif;
    }
    
    .sub-header {
        color: #a0a0a0;
        text-align: center;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Performance metric highlight */
    .fps-metric {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .fps-value {
        font-size: 3rem;
        font-weight: 800;
        color: white;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(233, 69, 96, 0.3);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 10px 20px;
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #e94560 0%, #ff6b6b 100%);
    }
    
    /* Status indicators */
    .status-ready { color: #4ecdc4; }
    .status-processing { color: #ffe66d; }
    .status-complete { color: #95e1d3; }
    .status-error { color: #ff6b6b; }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'processing': False,
        'results_ready': False,
        'track_data': None,
        'df_results': None,
        'stats': None,
        'detector': None,
        'output_video_path': None,
        'processing_fps': None,
        'processing_time': None,
        # Calibration mode settings
        'calibration_mode': False,
        'roi_top_width': 40.0,
        'roi_bottom_width': 90.0,
        'roi_horizon': 60.0,
        'roi_bottom_height': 90.0,
        'roi_horizontal_offset': 0.0,
        'exit_line_y_ratio': 85.0,
        'calibration_frame': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
@st.cache_resource
def load_detector(model_path: str, confidence: float):
    """Load and cache the YOLO detector."""
    logger.info(f"Loading detector: {model_path} (confidence: {confidence})")
    try:
        detector = PotholeDetector(
            model_path=model_path,
            conf_threshold=confidence,
            tracker_config="bytetrack.yaml"
        )
        logger.info("Detector loaded successfully")
        return detector
    except ModelLoadError as e:
        logger.error(f"Model load error: {e}")
        st.error(f"Model yukleme hatasi: {e}")
        return None
    except Exception as e:
        logger.exception("Unexpected error loading detector")
        st.error(f"Beklenmeyen hata: {e}")
        return None


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    """Save uploaded file to temp directory and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def _safe_cleanup_temp_files(*file_paths) -> None:
    """
    Safely delete temporary files without raising exceptions.
    
    Args:
        *file_paths: Variable number of file paths to delete
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.debug(f"Cleaned up temp file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file {file_path}: {e}")


def create_folium_map(df: pd.DataFrame):
    """Create an interactive Folium map with pothole markers."""
    try:
        import folium
        from streamlit_folium import st_folium
        
        # Filter rows with valid GPS coordinates
        df_gps = df[df['Enlem'].notna() & df['Boylam'].notna()]
        
        if df_gps.empty:
            st.warning("🗺️ Haritalama icin GPS verisi bulunamadi. Haritayi etkinlestirmek icin GPS dosyasi yukleyin.")
            return None
        
        # Calculate map center
        center_lat = df_gps['Enlem'].mean()
        center_lon = df_gps['Boylam'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='cartodbdark_matter'
        )
        
        # Add markers for each pothole
        for _, row in df_gps.iterrows():
            severity = row['Ciddiyet']
            
            # Color coding based on severity (aligned with Plan 5.1: 0-30 Green, 30-70 Yellow, 70+ Red)
            if severity >= 70:
                color = 'red'
                icon = 'exclamation-triangle'
            elif severity >= 30:
                color = 'orange'
                icon = 'exclamation'
            else:
                color = 'green'
                icon = 'check'
            
            # Create popup content
            popup_html = f"""
            <div style="font-family: Arial; width: 200px;">
                <h4 style="color: {color}; margin-bottom: 10px;">
                    Cukur #{int(row['Cukur ID'])}
                </h4>
                <table style="width: 100%;">
                    <tr><td><b>Ciddiyet:</b></td><td>{row['Ciddiyet']:.1f}</td></tr>
                    <tr><td><b>Oncelik:</b></td><td style="color: {color};">{row['Oncelik']}</td></tr>
                    <tr><td><b>Duzensizlik:</b></td><td>{row['Duzensizlik']:.4f}</td></tr>
                    <tr><td><b>Kareler:</b></td><td>{row['Kareler']}</td></tr>
                </table>
            </div>
            """
            
            folium.Marker(
                location=[row['Enlem'], row['Boylam']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"ID: {int(row['Cukur ID'])} | Ciddiyet: {row['Ciddiyet']:.0f}",
                icon=folium.Icon(color=color, icon_color='white', icon=icon, prefix='fa')
            ).add_to(m)
        
        return m
        
    except ImportError:
        st.error("Lutfen folium kurun: `pip install folium streamlit-folium`")
        return None


# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">🛣️ Otonom Yol Hasari Degerlendirme Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Yapay Zeka Destekli Cukur Tespit ve Haritalama Sistemi | Yuksek Performans Modu</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar configuration panel."""
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar")
        st.markdown("---")
        
        # Model settings
        st.markdown("### 🤖 Model Ayarlari")
        model_path = st.text_input(
            "Model Yolu",
            value=str(PROJECT_ROOT / "best1.engine"),
            help="TensorRT motor dosyasinin yolu"
        )
        
        confidence = st.slider(
            "Guven Esigi",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Tespit icin minimum guven degeri"
        )
        
        st.markdown("---")
        
        # Calibration Mode Section
        st.markdown("### 📐 Kalibrasyon Modu")
        calibration_mode = st.checkbox(
            "🎯 Kalibrasyon Modunu Etkinlestir",
            value=st.session_state.calibration_mode,
            help="Kamera acisina gore ROI'yi ayarlamak icin etkinlestirin"
        )
        st.session_state.calibration_mode = calibration_mode
        
        # ROI Settings (only shown when calibration mode is active)
        roi_top_width = st.session_state.roi_top_width
        roi_bottom_width = st.session_state.roi_bottom_width
        roi_horizon = st.session_state.roi_horizon
        roi_horizontal_offset = st.session_state.roi_horizontal_offset
        
        if calibration_mode:
            with st.expander("📐 ROI Ayarlari (Kamera Acisi)", expanded=True):
                st.info("💡 Slider'lari hareket ettirerek Kus Bakisi gorunumunde yol cizgilerinin paralel olmasini saglayin.")
                
                roi_top_width = st.slider(
                    "Ust Genislik (%)",
                    min_value=10,
                    max_value=100,
                    value=int(st.session_state.roi_top_width),
                    step=5,
                    help="ROI'nin ust kenarinin (ufuk cizgisi) genisligi. Daha dar = daha uzak."
                )
                st.session_state.roi_top_width = float(roi_top_width)
                
                roi_horizontal_offset = st.slider(
                    "Yatay Kayma (%) 🎯",
                    min_value=-20,
                    max_value=20,
                    value=int(st.session_state.roi_horizontal_offset),
                    step=1,
                    help="Kamera merkezden saga/sola kayiksa ROI'yi kaydirin. Negatif = sola, Pozitif = saga."
                )
                st.session_state.roi_horizontal_offset = float(roi_horizontal_offset)
                
                roi_bottom_width = st.slider(
                    "Alt Genislik (%)",
                    min_value=10,
                    max_value=100,
                    value=int(st.session_state.roi_bottom_width),
                    step=5,
                    help="ROI'nin alt kenarinin (kameraya yakin) genisligi."
                )
                st.session_state.roi_bottom_width = float(roi_bottom_width)
                
                roi_horizon = st.slider(
                    "Ufuk Cizgisi (Yukseklik %)",
                    min_value=20,
                    max_value=80,
                    value=int(st.session_state.roi_horizon),
                    step=5,
                    help="ROI'nin basladigi yukseklik. Dusuk deger = daha yuksekte."
                )
                st.session_state.roi_horizon = float(roi_horizon)
                
                roi_bottom_height = st.slider(
                    "Alt Kenar Konumu (Yukseklik %)",
                    min_value=50,
                    max_value=100,
                    value=int(st.session_state.get('roi_bottom_height', 90)),
                    step=1,
                    help="Mavi cerceveyi kaputun uzerinde bitirmek icin bunu yukari cekin."
                )
                st.session_state.roi_bottom_height = float(roi_bottom_height)
                
                st.markdown("---")
                
                exit_line_y_ratio = st.slider(
                    "Cikis/Kayit Cizgisi (Yukseklik %)",
                    min_value=50,
                    max_value=99,
                    value=int(st.session_state.get('exit_line_y_ratio', 85)),
                    step=1,
                    help="Cukurlar bu cizgiyi gecince veritabanina kaydedilir. Kirmizi cizgi ile gosterilir."
                )
                st.session_state.exit_line_y_ratio = float(exit_line_y_ratio)
                
                st.markdown("---")
                st.caption("📌 **Ipucu:** Yol cizgileri kus bakisinda paralel olmalidir.")
        
        st.markdown("---")
        
        # File uploads
        st.markdown("### 📁 Giris Dosyalari")
        
        video_file = st.file_uploader(
            "🎥 Video Yukle",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Analiz icin video dosyasi yukleyin"
        )
        
        gps_file = st.file_uploader(
            "📍 GPS Dosyasi (Opsiyonel)",
            type=['csv', 'json'],
            help="Haritalama icin GPS koordinatlari. Yuklenmezse GPS ozellikleri devre disi kalir."
        )
        
        st.markdown("---")
        
        # Status
        st.markdown("### 📊 Durum")
        if st.session_state.processing:
            st.markdown('<p class="status-processing">🔄 Isleniyor...</p>', unsafe_allow_html=True)
        elif st.session_state.results_ready:
            st.markdown('<p class="status-complete">✅ Analiz Tamamlandi</p>', unsafe_allow_html=True)
            if st.session_state.processing_fps:
                st.metric("⚡ Isleme Hizi", f"{st.session_state.processing_fps:.2f} FPS")
        else:
            st.markdown('<p class="status-ready">⏳ Hazir</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action button
        start_button = st.button(
            "🚀 ANALIZI BASLAT",
            disabled=video_file is None or st.session_state.processing,
            width='stretch'
        )
        
        # Database cleanup button
        st.markdown("---")
        st.markdown("### 🗑️ Veritabani")
        if st.button("🗑️ Gecmisi Temizle", width='stretch', type="secondary"):
            try:
                import shutil
                import time
                import gc
                
                logger.info("Starting database cleanup")
                
                # Force garbage collection to close any lingering connections
                gc.collect()
                
                # Clear all run directories
                runs_dir = PROJECT_ROOT / "runs" / "streamlit"
                if runs_dir.exists():
                    deleted_count = 0
                    failed_count = 0
                    
                    for run_folder in runs_dir.iterdir():
                        if run_folder.is_dir():
                            try:
                                # Try to close any open database connections in this run
                                db_file = run_folder / "detections.db"
                                if db_file.exists():
                                    # Force close any SQLite connections
                                    import sqlite3
                                    try:
                                        conn = sqlite3.connect(str(db_file))
                                        conn.close()
                                    except:
                                        pass
                                    
                                    # Wait a moment for Windows to release file handles
                                    time.sleep(0.1)
                                
                                # Now try to delete the folder
                                shutil.rmtree(run_folder)
                                deleted_count += 1
                                logger.info(f"Deleted run folder: {run_folder.name}")
                                
                            except PermissionError as e:
                                failed_count += 1
                                logger.warning(f"Could not delete {run_folder.name}: {e}")
                            except Exception as e:
                                failed_count += 1
                                logger.error(f"Error deleting {run_folder.name}: {e}")
                    
                    if deleted_count > 0:
                        st.success(f"✅ {deleted_count} analiz kaydi temizlendi!")
                    if failed_count > 0:
                        st.warning(f"⚠️ {failed_count} klasor silinemedi (hala acik olabilir)")
                else:
                    st.info("Temizlenecek veri bulunamadi.")
                
                # Clear snapshots
                snapshots_dir = PROJECT_ROOT / "data" / "snapshots"
                if snapshots_dir.exists():
                    snapshot_count = 0
                    for f in snapshots_dir.glob("*.jpg"):
                        try:
                            f.unlink()
                            snapshot_count += 1
                        except Exception as e:
                            logger.warning(f"Could not delete snapshot {f.name}: {e}")
                    
                    if snapshot_count > 0:
                        logger.info(f"Deleted {snapshot_count} snapshots")
                
                # Reset session state
                st.session_state.results_ready = False
                st.session_state.track_data = None
                st.session_state.df_results = None
                st.session_state.stats = None
                
                logger.info("Database cleanup completed")
                
            except Exception as e:
                logger.exception("Database cleanup failed")
                st.error(f"Temizleme hatasi: {e}")
                st.info("Ipucu: Analiz devam ediyorsa oncelikle durdurun, sonra temizleyin.")
        
        return {
            'model_path': model_path,
            'confidence': confidence,
            'video_file': video_file,
            'gps_file': gps_file,
            'start_button': start_button,
            'calibration_mode': calibration_mode,
            'roi_top_width': roi_top_width,
            'roi_bottom_width': roi_bottom_width,
            'roi_horizon': roi_horizon,
            'roi_horizontal_offset': roi_horizontal_offset,
            'roi_bottom_height': st.session_state.roi_bottom_height,
            'exit_line_y_ratio': st.session_state.exit_line_y_ratio
        }


def render_calibration_tab(config: dict):
    """Render the calibration tab with ROI preview."""
    st.markdown("### 📐 ROI Kalibrasyon Onizleme")
    st.info("🎯 Sol taraftaki slider'lari kullanarak ROI'yi ayarlayin. Kus Bakisi gorunumunde yol cizgileri paralel olmalidir.")
    
    video_file = config.get('video_file')
    
    if video_file is None:
        st.warning("⚠️ Kalibrasyon icin lutfen once bir video yukleyin.")
        return
    
    # Import geometry utils
    from src.geometry_utils import get_calibration_debug_frame
    import cv2
    
    # Save video temporarily and extract first frame
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        
        # Reset file pointer for later use
        video_file.seek(0)
        
        # Open video and get first frame
        cap = cv2.VideoCapture(tmp_path)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            st.error("❌ Video okunamadi. Lutfen gecerli bir video dosyasi yukleyin.")
            os.unlink(tmp_path)
            return
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Get ROI settings from config
        top_width = config.get('roi_top_width', 40.0)
        bottom_width = config.get('roi_bottom_width', 90.0)
        horizon = config.get('roi_horizon', 60.0)
        bottom_height = config.get('roi_bottom_height', 90.0)
        horizontal_offset = config.get('roi_horizontal_offset', 0.0)
        exit_line = config.get('exit_line_y_ratio', 85.0)
        
        # Generate calibration debug frames
        original_view, bev_view = get_calibration_debug_frame(
            frame, 
            top_width_pct=top_width,
            bottom_width_pct=bottom_width,
            horizon_pct=horizon,
            exit_line_y_ratio=exit_line,
            bottom_height_pct=bottom_height,
            horizontal_offset_pct=horizontal_offset
        )
        
        # Display side by side
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎥 Orijinal Goruntu + ROI")
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(original_view, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, width='stretch')
        
        with col2:
            st.markdown("#### 🦅 Kus Bakisi (BEV)")
            bev_rgb = cv2.cvtColor(bev_view, cv2.COLOR_BGR2RGB)
            st.image(bev_rgb, width='stretch')
        
        # Show current settings
        st.markdown("---")
        st.markdown("#### 📊 Mevcut ROI Ayarlari")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ust Genislik", f"{top_width:.0f}%")
        with col2:
            st.metric("Alt Genislik", f"{bottom_width:.0f}%")
        with col3:
            st.metric("Ufuk Cizgisi", f"{horizon:.0f}%")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Alt Kenar Konumu", f"{bottom_height:.0f}%", help="ROI'nin bittigi nokta")
        with col2:
            st.metric("Cikis Cizgisi", f"{exit_line:.0f}%", help="Kayit tetikleyici cizgi")
        
        st.success("✅ Ayarlar otomatik olarak analize uygulanacaktir.")
        
    except Exception as e:
        st.error(f"❌ Kalibrasyon hatasi: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_metrics_row(stats: dict):
    """Render the metrics row at the top."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Show unique committed tracks (actual potholes), not per-frame detections
        st.metric("🎯 Tespit Sayisi", f"{stats.get('tracks_committed', 0):,}")
    
    with col2:
        st.metric("🔴 YUKSEK", stats.get('high_priority_count', 0))
    
    with col3:
        st.metric("🟡 ORTA", stats.get('medium_priority_count', 0))
    
    with col4:
        st.metric("🟢 DUSUK", stats.get('low_priority_count', 0))
    
    with col5:
        if st.session_state.processing_fps:
            st.metric("⚡ FPS", f"{st.session_state.processing_fps:.1f}")


def render_processing_tab(config: dict):
    """Render the Processing tab with optimized UI."""
    st.markdown("### 🎬 Video Isleme")
    
    # Performance notice
    st.info("💡 **Optimize Mod:** Canli onizleme devre disi, maksimum isleme hizi icin. Video tamamlandiktan sonra gosterilecektir.")
    
    # Create placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_container = st.empty()
    
    # Results area (shown after processing)
    results_container = st.container()
    
    return {
        'progress_bar': progress_bar,
        'status_text': status_text,
        'metrics_container': metrics_container,
        'results_container': results_container
    }


def render_results_tab():
    """Render the Mapping & Results tab."""
    if not st.session_state.results_ready:
        st.info("🔍 Sonuclari gormek icin once analiz calistirin")
        return
    
    df_all = st.session_state.df_results
    stats = st.session_state.stats
    
    if df_all is None or df_all.empty:
        st.warning("Tespit sonucu bulunamadi")
        return
    
    # === SADECE COMMIT EDİLEN ÇUKURLARI GÖSTER ===
    if 'Kaydedildi' in df_all.columns:
        df = df_all[df_all['Kaydedildi'] == 'Evet'].copy()
    else:
        df = df_all.copy()
    
    if df.empty:
        st.info("📌 Bu bölümde sadece veritabanına commit edilen çukurlar gösterilir. Şu anda commit edilen tespit bulunmuyor.")
        return
    
    # Performance metrics prominently displayed
    st.markdown("### ⚡ Performans Metrikleri")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Isleme Hizi",
            f"{st.session_state.processing_fps:.2f} FPS",
            help="Saniyede islenen ortalama kare sayisi"
        )
    
    with col2:
        st.metric(
            "Isleme Suresi",
            f"{st.session_state.processing_time:.1f} sn",
            help="Toplam isleme suresi"
        )
    
    with col3:
        st.metric(
            "Islenen Kare",
            f"{stats.get('frames_processed', 0):,}",
            help="Analiz edilen toplam video karesi"
        )
    
    st.markdown("---")
    
    # Results section
    st.markdown("### 📋 Tespit Sonuclari")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Priority distribution chart
        st.markdown("#### Oncelik Dagilimi")
        priority_data = df['Oncelik'].value_counts()
        
        import plotly.express as px
        
        colors = {'HIGH': '#ff4444', 'MEDIUM': '#ffaa00', 'LOW': '#44ff44'}
        fig = px.pie(
            values=priority_data.values,
            names=priority_data.index,
            color=priority_data.index,
            color_discrete_map=colors,
            hole=0.4
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        # Severity histogram
        st.markdown("#### Ciddiyet Dagilimi")
        fig = px.histogram(df, x='Ciddiyet', nbins=20, color_discrete_sequence=['#e94560'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Ciddiyet Puani",
            yaxis_title="Adet"
        )
        st.plotly_chart(fig, width='stretch')
    
    # Data table
    st.markdown("#### 📊 Detayli Sonuc Tablosu")
    st.dataframe(
        df.style.background_gradient(subset=['Ciddiyet'], cmap='RdYlGn_r'),
        width='stretch',
        height=300
    )
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Sonuclari Indir (CSV)",
            data=csv,
            file_name=f"cukur_tespit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width='stretch'
        )
    
    with col2:
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            with open(st.session_state.output_video_path, 'rb') as f:
                st.download_button(
                    label="📥 Islenmis Videoyu Indir",
                    data=f.read(),
                    file_name=f"isaretli_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4",
                    width='stretch'
                )
    
    # Interactive Map
    st.markdown("### 🗺️ Etkilesimli Cukur Haritasi")
    
    map_obj = create_folium_map(df)
    if map_obj is not None:
        from streamlit_folium import st_folium
        st_folium(map_obj, width=None, height=500)
    
    # === NEW: Detection Gallery with Snapshots and Heatmaps ===
    st.markdown("---")
    st.markdown("### 🔍 Tespit Galeris\u0131 (Goruntu + Derinlik Analizi)")
    
    # Load detection data from database if available
    if st.session_state.track_data and hasattr(st.session_state, 'output_video_path'):
        # Get database path from output video path
        output_dir = Path(st.session_state.output_video_path).parent
        db_path = output_dir / "detections.db"
        
        if db_path.exists():
            from src.database_manager import DatabaseManager
            db = DatabaseManager(str(db_path))
            detections = db.get_all_detections()
            
            if detections:
                st.info(f"📊 Toplam {len(detections)} tespit görüntüleniyor")
                
                # Display detections in grid format (2 per row)
                for i in range(0, len(detections), 2):
                    cols = st.columns(2)
                    
                    for col_idx, col in enumerate(cols):
                        det_idx = i + col_idx
                        if det_idx < len(detections):
                            detection = detections[det_idx]
                            
                            with col:
                                # Create expander for each detection
                                risk_emoji = {
                                    'Yuksek': '⚠️',
                                    'Orta': '🟡',
                                    'Dusuk': '✅'
                                }.get(detection.get('risk_label', ''), '🔴')
                                
                                with st.expander(
                                    f"{risk_emoji} Cukur #{detection['track_id']} - "
                                    f"{detection.get('risk_label', 'N/A')} "
                                    f"(Ciddiyet: {detection.get('severity_score', 0):.1f})",
                                    expanded=False
                                ):
                                    # Side-by-side: Snapshot and Heatmap
                                    img_col1, img_col2 = st.columns(2)
                                    
                                    with img_col1:
                                        st.markdown("**📷 Goruntu**")
                                        if detection.get('image_path') and os.path.exists(detection['image_path']):
                                            st.image(detection['image_path'], use_container_width=True)
                                        else:
                                            st.warning("Goruntu yok")
                                    
                                    with img_col2:
                                        st.markdown("**🌡️ Topografik Analiz (Derinlik)**")
                                        if detection.get('heatmap_path') and os.path.exists(detection['heatmap_path']):
                                            st.image(detection['heatmap_path'], use_container_width=True)
                                        else:
                                            st.info("Derinlik analizi yok")
                                    
                                    # Detection info
                                    st.markdown("**Detaylar:**")
                                    info_cols = st.columns(3)
                                    with info_cols[0]:
                                        st.metric("Ciddiyet", f"{detection.get('severity_score', 0):.1f}")
                                    with info_cols[1]:
                                        st.metric("D\u00fczensizlik", f"{detection.get('circularity', 0):.3f}")
                                    with info_cols[2]:
                                        st.metric("G\u00f6receli Alan %", f"{detection.get('relative_area', 0)*100:.2f}")
                                    
                                    # GPS if available
                                    if detection.get('latitude') and detection.get('longitude'):
                                        st.markdown(
                                            f"📍 Konum: `{detection['latitude']:.6f}, "
                                            f"{detection['longitude']:.6f}`"
                                        )
            else:
                st.info("Galeri i\u00e7in tespit bulunamad\u0131")
        else:
            st.info("Veritaban\u0131 dosyas\u0131 bulunamad\u0131")
    else:
        st.info("Galeriyi g\u00f6rmek i\u00e7in \u00f6nce analiz \u00e7al\u0131\u015ft\u0131r\u0131n")


def run_optimized_analysis(config: dict, placeholders: dict):
    """Run the video analysis with optimized processing (no live preview)."""
    st.session_state.processing = True
    st.session_state.results_ready = False
    
    try:
        # Save uploaded files to temp paths
        placeholders['status_text'].text("📁 Dosyalar hazirlaniyor...")
        video_path = save_uploaded_file(config['video_file'], '.mp4')
        gps_path = None
        if config['gps_file'] is not None:
            suffix = '.csv' if config['gps_file'].name.endswith('.csv') else '.json'
            gps_path = save_uploaded_file(config['gps_file'], suffix)
        
        # Create output directory
        output_dir = PROJECT_ROOT / "runs" / "streamlit" / datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video = output_dir / "annotated_output.mp4"
        db_path = output_dir / "detections.db"
        
        # Load detector
        placeholders['status_text'].text("🤖 Yapay zeka modeli yukleniyor...")
        detector = load_detector(config['model_path'], config['confidence'])
        
        if detector is None:
            st.error("Model yuklenemedi")
            return
        
        # Initialize processor with ROI calibration settings
        placeholders['status_text'].text("⚙️ Islemci baslatiliyor...")
        processor = VideoProcessor(
            input_path=video_path,
            output_path=str(output_video),
            detector=detector,
            gps_file_path=gps_path,
            db_path=str(db_path),
            roi_top_width=config.get('roi_top_width', 40.0),
            roi_bottom_width=config.get('roi_bottom_width', 90.0),
            roi_horizon=config.get('roi_horizon', 60.0),
            roi_bottom_height=config.get('roi_bottom_height', 90.0),
            roi_horizontal_offset=config.get('roi_horizontal_offset', 0.0),
            exit_line_y_ratio=config.get('exit_line_y_ratio', 85.0)
        )
        
        # Status callback for progress updates
        def update_status(status):
            placeholders['progress_bar'].progress(status.progress_percent / 100)
            placeholders['status_text'].text(status.status_message)
            
            with placeholders['metrics_container'].container():
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Kare", f"{status.frame_idx}/{status.total_frames}")
                with col2:
                    st.metric("Bulunan Cukur", committed_count)
        
        # Run optimized processing
        placeholders['status_text'].text("🚀 Video isleniyor (optimize mod)...")
        result = processor.process_optimized(status_callback=update_status)
        
        # Processing complete
        placeholders['progress_bar'].progress(1.0)
        placeholders['status_text'].text("✅ Isleme tamamlandi!")
        
        # Store results
        st.session_state.track_data = result.track_data
        st.session_state.df_results = processor.get_track_data_as_dataframe()
        st.session_state.stats = result.stats
        st.session_state.output_video_path = result.output_video_path
        st.session_state.processing_fps = result.average_fps
        st.session_state.processing_time = result.processing_time_seconds
        st.session_state.results_ready = True
        
        # Generate CSV report (only committed potholes)
        csv_path = output_dir / "final_report.csv"
        generate_csv_report(result.track_data, str(csv_path), only_committed=True)
        
        # Show results in the same tab
        with placeholders['results_container']:
            st.success(f"✅ Analiz tamamlandi! {result.total_frames} kare {result.processing_time_seconds:.1f} saniyede islendi")
            
            # Performance metrics
            st.markdown("### ⚡ Performans Sonuclari")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Isleme Hizi", f"{result.average_fps:.2f} FPS")
            with col2:
                st.metric("Toplam Sure", f"{result.processing_time_seconds:.1f} sn")
            with col3:
                # Show committed tracks (potholes saved to database)
                committed_count = sum(1 for t in result.track_data.values() if t.committed_to_db)
                st.metric("Tespit Edilen Cukur", committed_count)
            
            # Show processed video
            st.markdown("### 🎬 Islenmis Video")
            if os.path.exists(result.output_video_path):
                st.video(result.output_video_path)
            
            st.info("👆 Detayli analiz ve etkilesimli harita icin **Harita ve Rapor** sekmesine gecin!")
        
        # Safe cleanup of temp files
        _safe_cleanup_temp_files(video_path, gps_path)
            
    except FileNotFoundError as e:
        logger.error(f"File not found during processing: {e}")
        st.error(f"❌ Dosya bulunamadi: {str(e)}")
        st.info("Lutfen video dosyasinin gecerli oldugunu kontrol edin.")
        _safe_cleanup_temp_files(video_path, gps_path)
    except VideoProcessingError as e:
        logger.error(f"Video processing error: {e}")
        st.error(f"❌ Video isleme hatasi: {str(e)}")
        st.info("Video codec'i desteklenmiyor olabilir. Farkli bir video deneyin.")
        _safe_cleanup_temp_files(video_path, gps_path)
    except ModelLoadError as e:
        logger.error(f"Model error during processing: {e}")
        st.error(f"❌ Model hatasi: {str(e)}")
        st.info("Model dosyasinin dogru konumda oldugunu kontrol edin.")
        _safe_cleanup_temp_files(video_path, gps_path)
    except Exception as e:
        logger.exception("Unexpected error during video processing")
        st.error(f"❌ Beklenmeyen hata: {str(e)}")
        with st.expander("Hata Detaylari"):
            st.code(traceback.format_exc())
        _safe_cleanup_temp_files(video_path, gps_path)
    
    finally:
        st.session_state.processing = False
        logger.info("Processing session ended")


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    """Main application entry point."""
    # Apply styling
    apply_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Render header
    render_header()
    
    # Render sidebar and get config
    config = render_sidebar()
    
    # Display metrics if results available
    if st.session_state.stats:
        render_metrics_row(st.session_state.stats)
    
    st.markdown("---")
    
    # Create tabs - add calibration tab if calibration mode is active
    if config['calibration_mode']:
        tab1, tab2, tab3 = st.tabs(["📐 Kalibrasyon", "🎬 Canli Analiz", "🗺️ Harita ve Rapor"])
        
        with tab1:
            render_calibration_tab(config)
        
        with tab2:
            placeholders = render_processing_tab(config)
            if config['start_button']:
                run_optimized_analysis(config, placeholders)
        
        with tab3:
            render_results_tab()
    else:
        tab1, tab2 = st.tabs(["🎬 Canli Analiz", "🗺️ Harita ve Rapor"])
        
        with tab1:
            placeholders = render_processing_tab(config)
            if config['start_button']:
                run_optimized_analysis(config, placeholders)
        
        with tab2:
            render_results_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <p>Otonom Yol Hasari Degerlendirme Sistemi v1.0 | Optimize Performans Modu</p>
            <p>YOLOv8 + TensorRT ile Calisir | Streamlit ile Gelistirildi</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
