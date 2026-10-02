# MedPortal Kullanım Kılavuzu

MedPortal, NVIDIA DGX Spark (`192.168.1.162`, aarch64, GB10, 128 GB unified bellek)
üzerinde çalışan iki CLI projesini tek bir web arayüzünde birleştirir:

- **RADAR:** Karın BT (`.nii.gz`) → aksiyel kesit görüntüleyici + 146 bulgu olasılık tablosu.
- **ClinFusion-32B:** 2D görüntü / 3D NIfTI / metin ile çok turlu sohbet.

Tarayıcıdan `http://192.168.1.162:8080` adresine gidilir. Bütün işlem Spark üzerinde
yapılır; istemci makinede yalnızca tarayıcı yeterlidir.

Arayüz **CORDATUS AI** markası altında koyu radyoloji konsolu olarak tasarlanmıştır;
tamamen el yazımı CSS ile çalışır, internet/CDN bağımlılığı yoktur (offline LAN'de
eksiksiz görünür).

> Modellerin ne yaptığı, sistemin nasıl çalıştığı ve katmanların ne alıp verdiği için
> bkz. [`DOKUMANTASYON.md`](DOKUMANTASYON.md).

## Mimari

Sistem üç katmandan oluşur:

1. **RADAR (subprocess):** Her RADAR isteğinde `backend` tarafından ayrı bir Python
   süreci olarak çalıştırılır. İstek bittiğinde süreç kapanır, bellek serbest kalır.
   Karın BT NIfTI dosyasını okur, aksiyel kesitleri PNG'ye çevirir ve 146 bulgu için
   olasılık skorlarını üretir.
2. **ClinFusion 32B (resident worker):** `worker/clinfusion_worker.py` süreci 32B
   modeli **bir kere** yükler ve bellekte tutar. `:8100` portunda küçük bir HTTP
   arayüzü (`/health`, `/generate`) sunar. Model yüklüyken diğer istekler bekler.
   Süreç düşerse `run.sh` tarafından başlatılan **supervisor** 10 saniye sonra
   yeniden başlatır.
3. **MedPortal backend (FastAPI):** `:8080` portunda çalışır. Yüklenen dosyaları
   saklar, GPU erişimini tek bir kuyruk (`gpu_queue`) ile serileştirir, RADAR
   subprocess'lerini ve worker çağrılarını yönetir, frontend statik dosyalarını
   servis eder.

Üç ayrı conda ortamı kullanılır:

| Ortam | Amaç | Python yolu | İçerik |
|-------|------|-------------|--------|
| `medportal` | Backend + testler | `/home/nvidia/miniconda3/envs/medportal/bin/python` | FastAPI, uvicorn, nibabel, pytest |
| `radar` | RADAR subprocess | `RADAR_PY` = `/home/nvidia/miniconda3/envs/radar/bin/python` | RADAR çıkarım bağımlılıkları (torch vb.) |
| `clinfusion` | 32B worker | `/home/nvidia/miniconda3/envs/clinfusion/bin/python` | transformers/torch, ayrıca `fastapi` + `uvicorn` |

Backend, RADAR subprocess'ini `config.RADAR_PY` ile (varsayılan
`/home/nvidia/miniconda3/envs/radar/bin/python`; `RADAR_PY` ortam değişkeniyle
geçersiz kılınabilir) başlatır.

## Başlatma / Durdurma / Durum

```bash
cd ~/medportal
./run.sh        # başlat (32B yüklenmesi ~14 dk)
./stop.sh       # durdur (--force: KILL ile zorla)
./status.sh     # süreç/port/health raporu
```

`run.sh` sırayla:

1. Eski kurulumu `stop.sh` ile temizler (supervisor + worker + backend).
2. Worker supervisor'ı `setsid nohup` ile başlatır (SSH oturumu kapansa bile yaşar)
   ve PID'sini `logs/worker-supervisor.pid`'e yazar.
3. **32B modelin yüklenmesini bekler.** İlk yükleme ~14 dakika sürer; script
   `:8100/health` yanıtında `"ready": true` görene kadar (en fazla 40 dk) bekler
   ve her dakika son log satırıyla ilerleme gösterir. Ayrıntı: `tail -f logs/worker.log`.
4. Backend'i `:8080` portunda başlatır; PID `logs/backend.pid`'dedir.

`stop.sh` sırasıyla worker **supervisor döngüsünü** (yoksa worker 10 sn'de geri
doğar), worker'ı ve backend'i durdurur; sonunda 8080/8100 portlarının boşaldığını
doğrular.

```
Erişim: http://192.168.1.162:8080
Model durumu: curl -s http://127.0.0.1:8100/health
```

Loglar `logs/worker.log` ve `logs/backend.log` dosyalarındadır.

## RADAR Sekmesi

1. Sol üstteki kutuya karın BT dosyanızı **sürükle-bırak** edin veya tıklayıp seçin
   (`.nii.gz` / `.nii`). Yükleme → analiz aşama göstergesi sağ panelde görünür.
2. **Kesit görüntüleyici:**
   - **Pencere ön ayarları:** Yumuşak doku (WL 40 / WW 400), Akciğer (WL -600 / WW 1500),
     Kemik (WL 400 / WW 1800), Tam (WL 0 / WW 2000). Sol üstte aktif pencere yazar.
   - **Kesit gezinme:** kaydırıcı, kesit numarası kutusu, `⇤`/`⇥`; klavyeyle
     `←` `→` (±1), `PgUp` `PgDn` (±10), `Home`/`End`.
   - **Yakınlaştırma:** `+`/`−`, fare tekerleği; `Sıfırla`. Zoomdayken sürükleyerek kaydırın.
3. Sağdaki **tablo 146 bulguyu** skorlarına göre listeler:
   - **Öne çıkanlar:** eşik üstü en yüksek 6 skor çip olarak en üstte.
   - **Bulgu ara:** metin girerek filtreleyin.
   - **Eşik:** kaydırıcı veya `0.30 / 0.50 / 0.70` hazır düğmeleri.
   - **Organ grupları:** bulguları organ başlıkları altında toplar.
   - **Skor ▾:** başlığa tıklayarak artan/azalan sıralamayı değiştirin.
   - Skorlar renkli **bar** ile gösterilir (≥0.7 yüksek/kırmızı, ≥0.3 orta/turuncu).
4. **CSV indir:** RADAR tamamlandığında sonuçları CSV olarak indirir
   (`/api/radar/{job_id}/csv`).

## ClinFusion Sekmesi

1. **Mesaj kutusuna** sorunuzu yazıp **Gönder**'e basın (`Enter` gönderir,
   `Shift+Enter` satır sonu). Yanıt gelene kadar "düşünüyor" göstergesi görünür.
   Model yanıtları basit markdown biçiminde (kod, liste, tablo, kalın) render edilir;
   cevabın üstündeki **Kopyala** ile panoya alınabilir.
2. **📎** düğmesiyle 2D görüntü (`.jpg/.png`), 3D NIfTI (`.nii/.nii.gz`) veya çoklu
   dosya ekleyebilirsiniz. Ekler tıklayarak büyütülebilir; göndermeden önce `×` ile
   kaldırılabilir.
3. Sohbet çok turludur; önceki mesajlar bağlam olarak korunur. **Ekler** yeni bir
   sohbet başlatılana kadar konuşma boyunca geçerlidir; her turda modele yeniden
   iletilir, böylece takip sorularında görüntü kaybolmaz.
4. **+ Yeni sohbet** ile geçmişi temizleyip baştan başlayabilirsiniz.

## API Uçları (özet)

| Yöntem | Uç | Açıklama |
|--------|-----|----------|
| `GET`  | `/api/status` | GPU/durum: `clinfusion_ready`, `busy`, `queue_len` |
| `POST` | `/api/upload` | `multipart/form-data`, alan: `file`. Yanıt: `upload_id`, `kind`, `n_slices` |
| `GET`  | `/api/nifti/{upload_id}/slice?idx=N&wc=&ww=` | PNG kesit (WC/WW penceresi); başlıkta `X-Total-Slices` |
| `POST` | `/api/radar` | Gövde: `{"upload_id": "..."}`. Yanıt: `{"job_id": "..."}` |
| `GET`  | `/api/radar/{job_id}` | `status` (`running`/`done`/`error`), `findings`, `csv_url` |
| `GET`  | `/api/radar/{job_id}/csv` | Sonuç CSV dosyası |
| `POST` | `/api/chat` | Gövde: `{"prompt","history","attachment_ids"}`. Yanıt: `{"job_id":...}` |
| `GET`  | `/api/chat/{job_id}` | `status`, `reply`, `error` |
| `POST` | `/api/chat/reset` | Sohbet durumunu sıfırlar (yanıt: `{"ok": true}`) |

Not: `slice` ucunda `wc`/`ww` atlanırsa varsayılan pencere WC 0 / WW 2000 uygulanır
(eski davranış). `ww <= 0` → 400.

## Sorun Giderme

- **Backend başlamıyor / `fastapi` veya `uvicorn` bulunamadı:** Backend `medportal`
  ortamında çalışır. Eksikse bu ortama kurun:
  ```bash
  /home/nvidia/miniconda3/envs/medportal/bin/pip install fastapi uvicorn
  ```
- **Worker başlamıyor / `fastapi` veya `uvicorn` bulunamadı (worker tarafı):**
  Worker `clinfusion` ortamında çalışır; bu ortama `fastapi` ve `uvicorn`
  kurulmuştur. Eksikse:
  ```bash
  /home/nvidia/miniconda3/envs/clinfusion/bin/pip install fastapi uvicorn
  ```
  (İleride `worker/requirements.txt`'e eklenebilir.)
- **Worker `503 "model yükleniyor"`:** 32B hâlâ yükleniyor ya da süreç düştü.
  `./status.sh` veya `curl -s http://127.0.0.1:8100/health` ile kontrol edin;
  `"ready": false` ise `logs/worker.log`'a bakın. Supervisor otomatik yeniden başlatır.
- **Sistem durmuyor (worker geri geliyor):** Supervisor döngüsü önce durdurulmalı;
  bunu `./stop.sh` otomatik yapar. Elle durdurduysanız `./stop.sh --force`.
- **Triton cache izin hatası:** Derlenmiş kernel cache'i yanlış sahipliğe düşmüş
  olabilir. Düzeltmek için:
  ```bash
  sudo chown -R nvidia:nvidia ~/.triton
  ```
- **Port çakışması:** `WORKER_PORT` / `BACKEND_PORT` ortam değişkenleriyle
  değiştirilebilir: `WORKER_PORT=8200 BACKEND_PORT=8081 ./run.sh`.

## Test

Backend çalışırken (veya en az worker hazırken) uçtan uca duman testi:

```bash
./scripts/smoke.sh
```

Bu script; durum sorgusu, dosya yükleme, kesit indirme, RADAR çalıştırma
(beklenen çıktı: `findings: 146`) ve backend pytest paketini sırayla çalıştırır.
Başarılıysa son satırda `SMOKE OK` yazar. Farklı bir sunucu için:

```bash
BASE=http://127.0.0.1:8081 NII=/yol/ornek.nii.gz ./scripts/smoke.sh
```

Yalnızca birim/API testleri (GPU gerekmez):

```bash
/home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests -q
```

## Not

`worker` süreci için `clinfusion` ortamına `fastapi` ve `uvicorn` kurulmuştur.
Bu bağımlılıklar `worker/requirements.txt` dosyasına eklenerek kalıcı hale
getirilebilir.