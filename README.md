# 🛣️ Otonom Yol Hasarı Değerlendirme Sistemi

Yapay zeka destekli, gerçek zamanlı yol hasarı tespit ve değerlendirme sistemi. YOLOv11 ile hasar tespiti, derinlik analizi ve GPS entegrasyonu içerir.

## 🎯 Özellikler

- **Gerçek Zamanlı Tespit**: YOLOv11 tabanlı çukur ve yol hasarı tespiti
- **Derinlik Analizi**: Depth Anything V2 ile hasar derinliği ölçümü
- **GPS Entegrasyonu**: Hasarların coğrafi konumlarının kaydedilmesi
- **Risk Değerlendirmesi**: Otomatik risk puanlama sistemi (Düşük/Orta/Yüksek)
- **Web Arayüzü**: Streamlit tabanlı görselleştirme ve analiz paneli
- **ROI Filtreleme**: Yol alanına odaklı akıllı tespit
- **Snapshot Sistemi**: Her hasar için en iyi kalite fotoğraf kaydı

## 📋 Gereksinimler

### Sistem Gereksinimleri
- Python 3.8+
- CUDA destekli GPU (opsiyonel, önerilir)
- 8GB+ RAM

### Yazılım Gereksinimleri
```bash
pip install -r requirements.txt
```

### Model Dosyaları
Projenin çalışması için aşağıdaki model dosyalarını indirmeniz gerekmektedir:

1. **YOLOv11 Model**: `best1.engine` veya `YOLOV11M.engine`
   - Yol hasarı tespiti için özel eğitilmiş YOLOv11 modeli
   - **Model İndirme:** [YOLOv11 Road Damage Model](https://github.com/ErenPervan/Yolo11Model)
   - Model dosyasını proje ana dizinine yerleştirin
   - Desteklenen formatlar: `.engine` (TensorRT), `.pt` (PyTorch)

2. **Depth Anything V2**: `checkpoints/depth_anything_v2_vits.pth`
   - Derinlik analizi için monoküler derinlik tahmin modeli
   - **İndirme:** [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)
   - `checkpoints/` klasörüne yerleştirin
   - Model boyutu: ~100MB (VITS versiyonu)

## 🚀 Kurulum

1. **Depoyu Klonlayın**
```bash
git clone https://github.com/ErenPervan/BitirmeProjesi.git
cd BitirmeProjesi
```

2. **Sanal Ortam Oluşturun**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
venv\Scripts\activate  # Windows
```

3. **Gereksinimleri Yükleyin**
```bash
pip install -r requirements.txt
```

4. **Model Dosyalarını İndirin**
   - **YOLOv11 Model**: [Yolo11Model](https://github.com/ErenPervan/Yolo11Model) deposundan indirin
     - Dosyayı `best1.engine` olarak proje ana dizinine kaydedin
   - **Depth Anything V2**: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) deposundan indirin
     - `depth_anything_v2_vits.pth` dosyasını `checkpoints/` klasörüne yerleştirin

## 🎮 Kullanım

### Web Arayüzü (Streamlit)
```bash
streamlit run src/app.py
```

Tarayıcınızda `http://localhost:8501` adresini açın.

### Komut Satırı
```bash
python src/main.py --video path/to/video.mp4
```

### Parametreler
- `--video`: İşlenecek video dosyası
- `--config`: Yapılandırma dosyası (varsayılan: `config.yaml`)
- `--output`: Çıktı klasörü (varsayılan: `runs/detect`)

## ⚙️ Yapılandırma

`config.yaml` dosyasından aşağıdaki ayarları özelleştirebilirsiniz:

```yaml
model:
  confidence: 0.5          # Tespit güven eşiği
  iou_threshold: 0.5       # NMS IoU eşiği

roi:
  top_width: 40.0          # ROI üst genişliği (%)
  bottom_width: 90.0       # ROI alt genişliği (%)
  horizon: 60.0            # Ufuk çizgisi yüksekliği (%)

severity:
  circularity_weight: 0.4  # Şekil düzensizliği ağırlığı
  area_weight: 0.6         # Alan ağırlığı

risk:
  low_threshold: 40.0      # Düşük risk eşiği
  medium_threshold: 65.0   # Orta risk eşiği
```

## 📁 Proje Yapısı

```
├── src/                      # Kaynak kodlar
│   ├── app.py               # Streamlit web uygulaması
│   ├── main.py              # CLI ana program
│   ├── detector.py          # Tespit motoru
│   ├── depth_utils.py       # Derinlik analizi
│   ├── video_processor.py   # Video işleme
│   └── ...
├── depth_anything_v2/       # Derinlik tahmin modeli
├── config.yaml              # Ana yapılandırma
├── requirements.txt         # Python bağımlılıkları
└── README.md                # Bu dosya
```

## 📊 Çıktılar

Sistem aşağıdaki çıktıları üretir:

- **Snapshots**: Her hasar için en iyi kalite fotoğraf (`data/snapshots/`)
- **CSV Rapor**: Tüm tespitlerin detaylı listesi (`final_report.csv`)
- **GPS Data**: Coğrafi konum bilgileri (JSON/CSV)
- **Logs**: İşlem logları (`logs/`)

## 🧪 Test

```bash
# Derinlik entegrasyonu testi
python test_depth_integration.py

# GPS entegrasyon testi
python src/main.py --video test_video.mp4 --gps-test
```

## 🛠️ Geliştirme

### Yeni Özellik Ekleme
1. `src/` altında ilgili modülü güncelleyin
2. `config.yaml`'a gerekli parametreleri ekleyin
3. Test edin ve dokümante edin

### Model Eğitimi
```bash
# YOLOv11 modelini eğitin
yolo task=detect mode=train model=yolo11m.pt data=road_damage.yaml epochs=100
```

## 📝 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır.





## 📚 Dökümantasyon

Detaylı entegrasyon kılavuzları:
- [DERINLIK_ENTEGRASYONU](DERINLIK_ENTEGRASYONU_TR.md)
- [DERINLIK_ENTEGRASYON_KILAVUZU](DERINLIK_ENTEGRASYON_KILAVUZU.md
)

---

