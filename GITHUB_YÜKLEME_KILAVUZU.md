# 🚀 GitHub'a Yükleme Kılavuzu

Bu doküman, projeyi GitHub'a nasıl yükleyeceğinizi adım adım açıklar.

## 📋 Ön Hazırlık

### 1. Git Kurulumu
Git yüklü değilse:
```bash
# Windows için: https://git-scm.com/download/win
# Git Bash veya PowerShell'den kontrol edin:
git --version
```

### 2. GitHub Hesabı
- GitHub hesabınız yoksa: https://github.com/signup
- Hesabınıza giriş yapın

## 🎯 GitHub'a Yükleme Adımları

### Adım 1: Yeni Repository Oluşturun

1. GitHub'da sağ üst köşedeki **+** işaretine tıklayın
2. **New repository** seçin
3. Repository ayarları:
   - **Repository name**: `road-damage-assessment` (veya istediğiniz isim)
   - **Description**: `Yapay zeka destekli yol hasarı tespit ve değerlendirme sistemi`
   - **Public** veya **Private** seçin
   - ⚠️ **README**, **.gitignore**, ve **license** ekleyin EKLEMEYIN (zaten var)
4. **Create repository** butonuna tıklayın

### Adım 2: Local Git Repository Başlatın

Proje klasöründe terminali açın ve şu komutları çalıştırın:

```bash
# Proje dizinine gidin
cd "c:\Users\EREN1\Desktop\Bitirme Projesi2"

# Git repository başlatın (eğer yoksa)
git init

# Tüm dosyaları staging area'ya ekleyin
git add .

# İlk commit'i yapın
git commit -m "Initial commit: Road damage assessment system"
```

### Adım 3: GitHub Repository'ye Bağlayın

GitHub'da oluşturduğunuz repository sayfasında gösterilen komutları kullanın:

```bash
# Remote repository ekleyin (USERNAME ve REPO-NAME'i değiştirin)
git remote add origin https://github.com/USERNAME/REPO-NAME.git

# Ana branch ismini main olarak ayarlayın
git branch -M main

# İlk push'u yapın
git push -u origin main
```

**Örnek:**
```bash
git remote add origin https://github.com/yourname/road-damage-assessment.git
git branch -M main
git push -u origin main
```

### Adım 4: Kimlik Doğrulama

GitHub push yaparken kimlik doğrulama gerekir:

#### Seçenek A: HTTPS (Personal Access Token)
1. GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. **Generate new token (classic)** tıklayın
3. **repo** yetkisini seçin
4. Token'ı kopyalayın (sadece bir kez gösterilir!)
5. Push yaparken:
   - Username: GitHub kullanıcı adınız
   - Password: Oluşturduğunuz token

#### Seçenek B: SSH (Önerilen)
```bash
# SSH key oluşturun
ssh-keygen -t ed25519 -C "your-email@example.com"

# Public key'i kopyalayın
cat ~/.ssh/id_ed25519.pub

# GitHub → Settings → SSH and GPG keys → New SSH key
# Kopyaladığınız key'i yapıştırın

# Remote URL'i SSH'ye çevirin
git remote set-url origin git@github.com:USERNAME/REPO-NAME.git
```

## 📦 Büyük Dosyalar (Git LFS)

Model dosyaları (*.engine, *.pth) 100MB'den büyükse Git LFS kullanın:

```bash
# Git LFS kurulumu
git lfs install

# Büyük dosyaları track edin
git lfs track "*.engine"
git lfs track "*.pth"

# .gitattributes'u commit edin
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

## ⚠️ Önemli Notlar

### Yüklenmemesi Gereken Dosyalar
`.gitignore` dosyası aşağıdakileri otomatik olarak hariç tutar:
- ✅ Model dosyaları (`*.engine`, `*.pth`) - Çok büyük
- ✅ Veritabanı (`*.db`) - Her ortamda yeni oluşturulmalı
- ✅ Video dosyaları (`*.mp4`, `*.avi`) - Çok büyük
- ✅ Snapshot'lar (`data/snapshots/*`) - Gereksiz
- ✅ Logs (`logs/`) - Her çalıştırmada yeni oluşur
- ✅ Runs (`runs/`) - Çıktı dosyaları
- ✅ Virtual environment (`venv/`) - Her ortamda yeni kurulur

### README'de Belirtilmesi Gerekenler
README.md'de kullanıcılara şunu bildirin:
- Model dosyalarını nereden indirecekleri
- Virtual environment nasıl kurulacak
- Gerekli sistem gereksinimleri

## 🔄 Güncellemeleri Push Etme

Değişikliklerinizi GitHub'a göndermek için:

```bash
# Değişen dosyaları görmek için
git status

# Değişiklikleri ekle
git add .

# Commit yap
git commit -m "Açıklayıcı commit mesajı"

# GitHub'a gönder
git push
```

## 🌿 Branch Kullanımı (Opsiyonel)

Yeni özellik geliştirirken:

```bash
# Yeni branch oluştur
git checkout -b feature/yeni-ozellik

# Değişiklikleri commit et
git add .
git commit -m "Yeni özellik eklendi"

# Branch'i push et
git push -u origin feature/yeni-ozellik

# GitHub'da Pull Request açın
```

## ✅ Kontrol Listesi

Yüklemeden önce kontrol edin:
- [ ] `.gitignore` dosyası var ve doğru
- [ ] `README.md` güncel ve açıklayıcı
- [ ] `requirements.txt` tüm bağımlılıkları içeriyor
- [ ] Hassas bilgiler (API keys, passwords) yok
- [ ] Model dosyaları yüklenmiyor (çok büyük)
- [ ] `.env` dosyası ignore edilmiş
- [ ] Test dosyaları ve gereksiz loglar temizlenmiş

## 🆘 Sorun Giderme

### "fatal: remote origin already exists"
```bash
git remote remove origin
git remote add origin https://github.com/USERNAME/REPO-NAME.git
```

### "failed to push some refs"
```bash
# Remote'taki değişiklikleri çek
git pull origin main --rebase
git push
```

### Büyük dosya hatası (file exceeds 100 MB)
```bash
# Dosyayı commit'ten kaldır
git rm --cached dosya_adi.engine

# .gitignore'a ekle (zaten var)
# Yeni commit yap
git commit -m "Remove large file"
```

## 📚 Yararlı Kaynaklar

- [Git Komutları Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [GitHub Docs](https://docs.github.com)
- [Git LFS](https://git-lfs.github.com/)

## 🎉 Başarılı!

Projeniz artık GitHub'da! Repository URL'iniz:
```
https://github.com/USERNAME/REPO-NAME
```

README.md'de bu URL'i kullanıcılarla paylaşabilirsiniz.

---

**İpucu:** Repository'yi public yaparsanız, README.md'deki görsellerin ve badge'lerin düzgün göründüğünden emin olun!
