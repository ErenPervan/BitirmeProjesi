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
    page_title="Yol Hasarı Değerlendirme Sistemi",
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
    try:
        detector = PotholeDetector(
            model_path=model_path,
            conf_threshold=confidence,
            tracker_config="bytetrack.yaml"
        )
        return detector
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def save_uploaded_file(uploaded_file, suffix: str) -> str:
    """Save uploaded file to temp directory and return path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def create_folium_map(df: pd.DataFrame):
    """Create an interactive Folium map with pothole markers."""
    try:
        import folium
        from streamlit_folium import st_folium
        
        # Filter rows with valid GPS coordinates
        df_gps = df[df['Latitude'].notna() & df['Longitude'].notna()]
        
        if df_gps.empty:
            st.warning("🗺️ Haritalama için GPS verisi bulunamadı. Haritayı etkinleştirmek için GPS dosyası yükleyin.")
            return None
        
        # Calculate map center
        center_lat = df_gps['Latitude'].mean()
        center_lon = df_gps['Longitude'].mean()
        
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='cartodbdark_matter'
        )
        
        # Add markers for each pothole
        for _, row in df_gps.iterrows():
            severity = row['Severity']
            
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
                    Çukur #{int(row['Pothole ID'])}
                </h4>
                <table style="width: 100%;">
                    <tr><td><b>Ciddiyet:</b></td><td>{row['Severity']:.1f}</td></tr>
                    <tr><td><b>Öncelik:</b></td><td style="color: {color};">{row['Priority']}</td></tr>
                    <tr><td><b>Düzensizlik:</b></td><td>{row['Irregularity']:.4f}</td></tr>
                    <tr><td><b>Kare:</b></td><td>{row['Frames']}</td></tr>
                </table>
            </div>
            """
            
            folium.Marker(
                location=[row['Latitude'], row['Longitude']],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"ID: {int(row['Pothole ID'])} | Ciddiyet: {row['Severity']:.0f}",
                icon=folium.Icon(color=color, icon_color='white', icon=icon, prefix='fa')
            ).add_to(m)
        
        return m
        
    except ImportError:
        st.error("Lütfen folium kurun: `pip install folium streamlit-folium`")
        return None


# =============================================================================
# UI COMPONENTS
# =============================================================================
def render_header():
    """Render the application header."""
    st.markdown('<h1 class="main-header">🛣️ Otonom Yol Hasarı Değerlendirme Sistemi</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Yapay Zeka Destekli Çukur Tespit ve Haritalama Sistemi | Yüksek Performans Modu</p>', unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar configuration panel."""
    with st.sidebar:
        st.markdown("## ⚙️ Ayarlar")
        st.markdown("---")
        
        # Model settings
        st.markdown("### 🤖 Model Ayarları")
        model_path = st.text_input(
            "Model Yolu",
            value=str(PROJECT_ROOT / "best1.engine"),
            help="TensorRT motor dosyasının yolu"
        )
        
        confidence = st.slider(
            "Güven Eşiği",
            min_value=0.1,
            max_value=1.0,
            value=0.25,
            step=0.05,
            help="Tespit için minimum güven değeri"
        )
        
        st.markdown("---")
        
        # Calibration Mode Section
        st.markdown("### 📐 Kalibrasyon Modu")
        calibration_mode = st.checkbox(
            "🎯 Kalibrasyon Modunu Etkinleştir",
            value=st.session_state.calibration_mode,
            help="Kamera açısına göre ROI'yi ayarlamak için etkinleştirin"
        )
        st.session_state.calibration_mode = calibration_mode
        
        # ROI Settings (only shown when calibration mode is active)
        roi_top_width = st.session_state.roi_top_width
        roi_bottom_width = st.session_state.roi_bottom_width
        roi_horizon = st.session_state.roi_horizon
        
        if calibration_mode:
            with st.expander("📐 ROI Ayarları (Kamera Açısı)", expanded=True):
                st.info("💡 Slider'ları hareket ettirerek Kuş Bakışı görünümünde yol çizgilerinin paralel olmasını sağlayın.")
                
                roi_top_width = st.slider(
                    "Üst Genişlik (%)",
                    min_value=10,
                    max_value=100,
                    value=int(st.session_state.roi_top_width),
                    step=5,
                    help="ROI'nin üst kenarının (ufuk çizgisi) genişliği. Daha dar = daha uzak."
                )
                st.session_state.roi_top_width = float(roi_top_width)
                
                roi_bottom_width = st.slider(
                    "Alt Genişlik (%)",
                    min_value=10,
                    max_value=100,
                    value=int(st.session_state.roi_bottom_width),
                    step=5,
                    help="ROI'nin alt kenarının (kameraya yakın) genişliği."
                )
                st.session_state.roi_bottom_width = float(roi_bottom_width)
                
                roi_horizon = st.slider(
                    "Ufuk Çizgisi (Yükseklik %)",
                    min_value=20,
                    max_value=80,
                    value=int(st.session_state.roi_horizon),
                    step=5,
                    help="ROI'nin başladığı yükseklik. Düşük değer = daha yüksekte."
                )
                st.session_state.roi_horizon = float(roi_horizon)
                
                st.markdown("---")
                st.caption("📌 **İpucu:** Yol çizgileri kuş bakışında paralel olmalıdır.")
        
        st.markdown("---")
        
        # File uploads
        st.markdown("### 📁 Giriş Dosyaları")
        
        video_file = st.file_uploader(
            "🎥 Video Yükle",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Analiz için video dosyası yükleyin"
        )
        
        gps_file = st.file_uploader(
            "📍 GPS Dosyası (Opsiyonel)",
            type=['csv', 'json'],
            help="Haritalama için GPS koordinatları. Yüklenmezse GPS özellikleri devre dışı kalır."
        )
        
        st.markdown("---")
        
        # Status
        st.markdown("### 📊 Durum")
        if st.session_state.processing:
            st.markdown('<p class="status-processing">🔄 İşleniyor...</p>', unsafe_allow_html=True)
        elif st.session_state.results_ready:
            st.markdown('<p class="status-complete">✅ Analiz Tamamlandı</p>', unsafe_allow_html=True)
            if st.session_state.processing_fps:
                st.metric("⚡ İşleme Hızı", f"{st.session_state.processing_fps:.2f} FPS")
        else:
            st.markdown('<p class="status-ready">⏳ Hazır</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Action button
        start_button = st.button(
            "🚀 ANALİZİ BAŞLAT",
            disabled=video_file is None or st.session_state.processing,
            use_container_width=True
        )
        
        # Database cleanup button
        st.markdown("---")
        st.markdown("### 🗑️ Veritabanı")
        if st.button("🗑️ Geçmişi Temizle", use_container_width=True, type="secondary"):
            try:
                # Clear all run directories
                runs_dir = PROJECT_ROOT / "runs" / "streamlit"
                if runs_dir.exists():
                    import shutil
                    for run_folder in runs_dir.iterdir():
                        if run_folder.is_dir():
                            shutil.rmtree(run_folder)
                # Clear snapshots
                snapshots_dir = PROJECT_ROOT / "data" / "snapshots"
                if snapshots_dir.exists():
                    for f in snapshots_dir.glob("*.jpg"):
                        f.unlink()
                st.success("✅ Veritabanı temizlendi, sunuma hazır!")
                # Reset session state
                st.session_state.results_ready = False
                st.session_state.track_data = None
                st.session_state.df_results = None
                st.session_state.stats = None
            except Exception as e:
                st.error(f"Temizleme hatası: {e}")
        
        return {
            'model_path': model_path,
            'confidence': confidence,
            'video_file': video_file,
            'gps_file': gps_file,
            'start_button': start_button,
            'calibration_mode': calibration_mode,
            'roi_top_width': roi_top_width,
            'roi_bottom_width': roi_bottom_width,
            'roi_horizon': roi_horizon
        }


def render_calibration_tab(config: dict):
    """Render the calibration tab with ROI preview."""
    st.markdown("### 📐 ROI Kalibrasyon Önizleme")
    st.info("🎯 Sol taraftaki slider'ları kullanarak ROI'yi ayarlayın. Kuş Bakışı görünümünde yol çizgileri paralel olmalıdır.")
    
    video_file = config.get('video_file')
    
    if video_file is None:
        st.warning("⚠️ Kalibrasyon için lütfen önce bir video yükleyin.")
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
            st.error("❌ Video okunamadı. Lütfen geçerli bir video dosyası yükleyin.")
            os.unlink(tmp_path)
            return
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Get ROI settings from config
        top_width = config.get('roi_top_width', 40.0)
        bottom_width = config.get('roi_bottom_width', 90.0)
        horizon = config.get('roi_horizon', 60.0)
        
        # Generate calibration debug frames
        original_view, bev_view = get_calibration_debug_frame(
            frame, 
            top_width_pct=top_width,
            bottom_width_pct=bottom_width,
            horizon_pct=horizon
        )
        
        # Display side by side
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🎥 Orijinal Görüntü + ROI")
            # Convert BGR to RGB for display
            original_rgb = cv2.cvtColor(original_view, cv2.COLOR_BGR2RGB)
            st.image(original_rgb, use_container_width=True)
        
        with col2:
            st.markdown("#### 🦅 Kuş Bakışı (BEV)")
            bev_rgb = cv2.cvtColor(bev_view, cv2.COLOR_BGR2RGB)
            st.image(bev_rgb, use_container_width=True)
        
        # Show current settings
        st.markdown("---")
        st.markdown("#### 📊 Mevcut ROI Ayarları")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Üst Genişlik", f"{top_width:.0f}%")
        with col2:
            st.metric("Alt Genişlik", f"{bottom_width:.0f}%")
        with col3:
            st.metric("Ufuk Çizgisi", f"{horizon:.0f}%")
        
        st.success("✅ Ayarlar otomatik olarak analize uygulanacaktır.")
        
    except Exception as e:
        st.error(f"❌ Kalibrasyon hatası: {str(e)}")
        import traceback
        st.code(traceback.format_exc())


def render_metrics_row(stats: dict):
    """Render the metrics row at the top."""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("🎯 Tespit Sayısı", f"{stats.get('total_detections', 0):,}")
    
    with col2:
        st.metric("🔴 YÜKSEK", stats.get('high_priority_count', 0))
    
    with col3:
        st.metric("🟡 ORTA", stats.get('medium_priority_count', 0))
    
    with col4:
        st.metric("🟢 DÜŞÜK", stats.get('low_priority_count', 0))
    
    with col5:
        if st.session_state.processing_fps:
            st.metric("⚡ FPS", f"{st.session_state.processing_fps:.1f}")


def render_processing_tab(config: dict):
    """Render the Processing tab with optimized UI."""
    st.markdown("### 🎬 Video İşleme")
    
    # Performance notice
    st.info("💡 **Optimize Mod:** Canlı önizleme devre dışı, maksimum işleme hızı için. Video tamamlandıktan sonra gösterilecektir.")
    
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
        st.info("🔍 Sonuçları görmek için önce analiz çalıştırın")
        return
    
    df = st.session_state.df_results
    stats = st.session_state.stats
    
    if df is None or df.empty:
        st.warning("Tespit sonucu bulunamadı")
        return
    
    # Performance metrics prominently displayed
    st.markdown("### ⚡ Performans Metrikleri")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "İşleme Hızı",
            f"{st.session_state.processing_fps:.2f} FPS",
            help="Saniyede işlenen ortalama kare sayısı"
        )
    
    with col2:
        st.metric(
            "İşleme Süresi",
            f"{st.session_state.processing_time:.1f} sn",
            help="Toplam işleme süresi"
        )
    
    with col3:
        st.metric(
            "İşlenen Kare",
            f"{stats.get('frames_processed', 0):,}",
            help="Analiz edilen toplam video karesi"
        )
    
    st.markdown("---")
    
    # Results section
    st.markdown("### 📋 Tespit Sonuçları")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Priority distribution chart
        st.markdown("#### Öncelik Dağılımı")
        priority_data = df['Priority'].value_counts()
        
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
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity histogram
        st.markdown("#### Ciddiyet Dağılımı")
        fig = px.histogram(df, x='Severity', nbins=20, color_discrete_sequence=['#e94560'])
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            xaxis_title="Ciddiyet Puanı",
            yaxis_title="Adet"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Data table
    st.markdown("#### 📊 Detaylı Sonuç Tablosu")
    st.dataframe(
        df.style.background_gradient(subset=['Severity'], cmap='RdYlGn_r'),
        use_container_width=True,
        height=300
    )
    
    # Download buttons
    col1, col2 = st.columns(2)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Sonuçları İndir (CSV)",
            data=csv,
            file_name=f"cukur_tespit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        if st.session_state.output_video_path and os.path.exists(st.session_state.output_video_path):
            with open(st.session_state.output_video_path, 'rb') as f:
                st.download_button(
                    label="📥 İşlenmiş Videoyu İndir",
                    data=f.read(),
                    file_name=f"isaretli_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                    mime="video/mp4",
                    use_container_width=True
                )
    
    # Interactive Map
    st.markdown("### 🗺️ Etkileşimli Çukur Haritası")
    
    map_obj = create_folium_map(df)
    if map_obj is not None:
        from streamlit_folium import st_folium
        st_folium(map_obj, width=None, height=500)


def run_optimized_analysis(config: dict, placeholders: dict):
    """Run the video analysis with optimized processing (no live preview)."""
    st.session_state.processing = True
    st.session_state.results_ready = False
    
    try:
        # Save uploaded files to temp paths
        placeholders['status_text'].text("📁 Dosyalar hazırlanıyor...")
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
        placeholders['status_text'].text("🤖 Yapay zeka modeli yükleniyor...")
        detector = load_detector(config['model_path'], config['confidence'])
        
        if detector is None:
            st.error("Model yüklenemedi")
            return
        
        # Initialize processor with ROI calibration settings
        placeholders['status_text'].text("⚙️ İşlemci başlatılıyor...")
        processor = VideoProcessor(
            input_path=video_path,
            output_path=str(output_video),
            detector=detector,
            gps_file_path=gps_path,
            db_path=str(db_path),
            roi_top_width=config.get('roi_top_width', 40.0),
            roi_bottom_width=config.get('roi_bottom_width', 90.0),
            roi_horizon=config.get('roi_horizon', 60.0)
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
                    st.metric("Bulunan Çukur", status.unique_tracks)
        
        # Run optimized processing
        placeholders['status_text'].text("🚀 Video işleniyor (optimize mod)...")
        result = processor.process_optimized(status_callback=update_status)
        
        # Processing complete
        placeholders['progress_bar'].progress(1.0)
        placeholders['status_text'].text("✅ İşleme tamamlandı!")
        
        # Store results
        st.session_state.track_data = result.track_data
        st.session_state.df_results = processor.get_track_data_as_dataframe()
        st.session_state.stats = result.stats
        st.session_state.output_video_path = result.output_video_path
        st.session_state.processing_fps = result.average_fps
        st.session_state.processing_time = result.processing_time_seconds
        st.session_state.results_ready = True
        
        # Generate CSV report
        csv_path = output_dir / "final_report.csv"
        generate_csv_report(result.track_data, str(csv_path))
        
        # Show results in the same tab
        with placeholders['results_container']:
            st.success(f"✅ Analiz tamamlandı! {result.total_frames} kare {result.processing_time_seconds:.1f} saniyede işlendi")
            
            # Performance metrics
            st.markdown("### ⚡ Performans Sonuçları")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("İşleme Hızı", f"{result.average_fps:.2f} FPS")
            with col2:
                st.metric("Toplam Süre", f"{result.processing_time_seconds:.1f} sn")
            with col3:
                st.metric("Tespit Edilen Çukur", len(result.track_data))
            
            # Show processed video
            st.markdown("### 🎬 İşlenmiş Video")
            if os.path.exists(result.output_video_path):
                st.video(result.output_video_path)
            
            st.info("👆 Detaylı analiz ve etkileşimli harita için **Harita ve Rapor** sekmesine geçin!")
        
        # Cleanup temp files
        os.unlink(video_path)
        if gps_path:
            os.unlink(gps_path)
            
    except Exception as e:
        st.error(f"❌ Analiz sırasında hata: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    finally:
        st.session_state.processing = False


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
        tab1, tab2, tab3 = st.tabs(["📐 Kalibrasyon", "🎬 Canlı Analiz", "🗺️ Harita ve Rapor"])
        
        with tab1:
            render_calibration_tab(config)
        
        with tab2:
            placeholders = render_processing_tab(config)
            if config['start_button']:
                run_optimized_analysis(config, placeholders)
        
        with tab3:
            render_results_tab()
    else:
        tab1, tab2 = st.tabs(["🎬 Canlı Analiz", "🗺️ Harita ve Rapor"])
        
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
            <p>Otonom Yol Hasarı Değerlendirme Sistemi v1.0 | Optimize Performans Modu</p>
            <p>YOLOv8 + TensorRT ile Çalışır | Streamlit ile Geliştirildi</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
