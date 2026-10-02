# MedPortal — Sistem ve Model Dokümantasyonu

Bu doküman, MedPortal'ın **hangi modeli ne için kullandığını**, sistemin **nasıl çalıştığını** ve
her katmanın **ne aldığı / ne verdiğini** anlatır. Operasyonel kullanım (kurulum, çalıştırma,
komutlar) için bkz. [`KULLANIM.md`](KULLANIM.md).

- **Hedef ortam:** NVIDIA DGX Spark (`192.168.1.162`, aarch64, GB10, 128GB unified)
- **Erişim:** `http://192.168.1.162:8080`
- **İki proje:** RADAR (bulgu tespiti) + ClinFusion-32B (tıbbi multimodal sohbet)

---

## 1. Genel bakış

MedPortal, iki bağımsız tıbbi yapay zekâ projesini tek bir web arayüzünde birleştirir:

| Sekme | Model | Görev | Çıktı |
|---|---|---|---|
| **RADAR** | RADAR (görsel encoder + BERT metin) | Karın BT'de **146 bulgu** için pozitif olasılık skoru | Bulgu tablosu (isim + skor), CSV |
| **ClinFusion** | ClinFusion-32B (Qwen3-VL-32B tabanlı MLLM) | 2D görüntü / 3D NIfTI / metin üzerine **çok turlu sohbet** | Serbest metin cevap |

MedPortal bir **orkestratördür**: modelleri kendi ortamlarında çalıştırır, GPU'yu serileştirir,
dosya yüklemeyi, iş kuyruğunu ve sonuç görüntülemeyi yönetir. Kendisi model eğitmez.

---

## 2. Sistem mimarisi

```
   ┌──────────────────────────┐
   │  Tarayıcı (LAN)          │
   │  http://192.168.1.162:8080│
   └────────────┬─────────────┘
                │ HTTP (JSON + multipart)
                ▼
   ┌──────────────────────────────────────────┐
   │  Portal Backend (FastAPI)                 │
   │  env: medportal  (Python 3.11)            │
   │  - statik SPA servis eder                 │
   │  - /api/upload, /api/nifti, /api/radar,   │
   │    /api/chat, /api/status                 │
   │  - tek GPU asyncio kuyruğu                │
   └───────┬───────────────────────┬──────────┘
           │ subprocess            │ localhost HTTP
           ▼                       ▼
 ┌───────────────────┐   ┌──────────────────────────────┐
 │ RADAR runner      │   │ ClinFusion-32B Worker         │
 │ env: radar (3.10) │   │ env: clinfusion (3.11)        │
 │ istek başına      │   │ 127.0.0.1:8100, RESIDENT      │
 │ ~2GB, ~10-30sn    │   │ 32B bir kez yüklenir (~89GB)  │
 └───────────────────┘   └──────────────────────────────┘
```

**Neden üç ayrı ortam?** RADAR `transformers 4.25`, ClinFusion `transformers 4.57` gerektirir;
aynı Python ortamında yaşayamazlar. Backend hafif tutulur; modeller kendi env'lerinde süreç olarak
çalıştırılır.

---

## 3. Modeller

### 3.1 RADAR — bulgu tespiti

| | |
|---|---|
| **Ne yapar** | Kontrastlı karın BT'de 36 organa dağılmış **146 bulgu** için 0–1 arası pozitif olasılık skoru üretir. |
| **Girdi** | `.nii.gz` NIfTI, **kontrastlı karın BT**, aksiyal. (HU değerleri beklenir.) |
| **Ön işleme** | Yeniden örnekleme (referans spacing `1×1×5 mm`), HU kırpma (`-300..400`), min-max normalize, non-zero bölgeyi kırpma, `96×256×384` boyutuna padding. |
| **Çıktı** | 146 satır: `{name (English), label (中文 (English)), score}`; skora göre azalan. |
| **Ortam / ağırlık** | env `radar` (3.10) · `~/radar/ckpt/checkpoint_radar_pretrain.pth` (~1.57GB) + `bert-base-chinese/` (~393MB) |
| **Çalışma** | İstek başına **subprocess** (`inference_demo.py`); model küçük (başlangıç ~10-30sn). |
| **Sınırlar** | Tek GPU; `batch_size=1`; RMB: GPU şart (CPU fallback yok); tek dosya klasörü. |

RADAR iki koldan oluşur: bir **3D görsel encoder** (UNet tabanlı) görüntüyü temsil eder; bir **metin
encoder'ı (BERT)** 146 bulgu adının dilsel temsilini üretir. Model, görüntü–metin eşleşmesinden her
bulgu için pozitif olasılık skorlar.

### 3.2 ClinFusion-32B — tıbbi multimodal sohbet

| | |
|---|---|
| **Ne yapar** | 2D tıbbi görüntü, native 3D NIfTI (CT/MRI) ve/veya metin üzerine serbest metin cevap üretir; çok turlu sohbeti destekler. |
| **Temel model** | **Qwen3-VL-32B-Instruct** (~63GB) |
| **Ek encoder'lar** | **DINOv2-large** (~2.3GB, yoğun uzamsal semantik) + **CLIP-ConvNeXt-large** (~2.7GB, yapısal özellik) + **3D pe encoder** — Cascade Spatial-Aware Locality Fusion ile birleştirilir. |
| **Fine-tuned ckpt** | `Alibaba-DAMO-Academy/ClinFusion-32B` (~71GB) |
| **Girdi** | `prompt` (metin) + opsiyonel `image[]` (`.jpg/.png`) + opsiyonel `nifti[]` (`.nii.gz`) + sohbet geçmişi |
| **3D ön işleme** | MONAI ile RAS oryantasyonu, `1×1×1 mm` yeniden örnekleme, HU penceresi `-1000..1000`, `(64,512,512)` boyut; hacimden `num_slices` dilim örneklenip modele verilir. |
| **Çıktı** | Serbest metin (`reply`) |
| **Ortam / ağırlık** | env `clinfusion` (3.11) · `~/ClinFusion/cache/models/` |
| **Çalışma** | **Resident worker**: 32B bir kez yüklenir (~13 dk), localhost HTTP ile sürekli hizmet verir. Bu sayede her sohbet isteğinde yeniden yüklenmez. |
| **Bellek** | ~**89GB** (128GB unified içinde); tek seferde bir üretim (generation kilidi). |
| **Sınırlar** | `batch_size=1`, `max_new_tokens=4096`; 3D en ağır; uzun cevaplarda OOM riski (marj ~39GB). |

### 3.3 Karşılaştırma

| Model | Girdi | Çıktı | Bellek | Portal sekmesi |
|---|---|---|---|---|
| **RADAR** | Karın BT `.nii.gz` | 146 bulgu skoru | ~2GB | RADAR |
| **ClinFusion-32B** | 2D / 3D / metin | Serbest metin | ~89GB | ClinFusion |
| ClinFusion-8B (indirilmiş, kullanılmıyor) | 2D / 3D / metin | Serbest metin | ~25GB | — |
| Qwen3-VL-8B/32B (taban, kullanılmıyor doğrudan) | — | — | — | (ClinFusion tabanı) |

---

## 4. Bileşenler

| Bileşen | Konum | Sorumluluk |
|---|---|---|
| **Backend** | `backend/app/` (`medportal` env) | HTTP rotaları, yükleme doğrulama, RADAR orkestrasyonu, ClinFusion worker'a iletim, NIfTI kesit PNG, GPU kuyruğu |
| **Worker** | `worker/clinfusion_worker.py` (`clinfusion` env) | ClinFusion-32B'yi yükleyip `/generate` ile üretim; `/health` ile durum |
| **Frontend** | `frontend/` | Tek sayfa SPA (vanilla JS + el yazımı CSS (CDN yok)): RADAR ve ClinFusion sekmeleri |
| **RADAR CLI** | `~/radar/RADAR_inference/inference_demo.py` | Gerçek RADAR inference (backend subprocess olarak çağırır) |

Backend modülleri ve tek sorumlulukları:

| Dosya | İş |
|---|---|
| `config.py` | Yollar, portlar, limitler (tek kaynak) |
| `store.py` | Job durum kaydı (queued/running/done/error) |
| `gpu_queue.py` | Tek GPU asyncio kilidi |
| `uploads.py` | Dosya türü/boyut doğrulama, güvenli kayıt, `resolve` |
| `nifti.py` | NIfTI hacim okuma (LRU önbellek) + aksiyel kesit PNG (WC/WW pencereleme) |
| `radar.py` | RADAR subprocess + CSV → findings |
| `clinfusion.py` | Worker HTTP istemcisi |
| `main.py` | FastAPI rotaları |

---

## 5. Uçtan uca akışlar

### 5.1 RADAR sekmesi (BT → 146 bulgu)

```
1. Kullanıcı .nii.gz sürükler
   → POST /api/upload (multipart)
   ← {upload_id, kind:"nifti", filename, n_slices}
2. Frontend kesiti yükler
   → GET /api/nifti/{upload_id}/slice?idx=N&wc=&ww=
   ← PNG (+ header X-Total-Slices)
3. Frontend analiz başlatır
   → POST /api/radar {upload_id}
   ← {job_id}                     (iş kuyruğa alınır)
4. Frontend 1-2 sn'de bir yoklar
   → GET /api/radar/{job_id}
   ← {status:"running", ...}  → ... → {status:"done", findings:[...], csv_url}
5. Kullanıcı sonucu inceler (eşik/arama/sıralama) ve CSV indirir
   → GET /api/radar/{job_id}/csv
```

Backend tarafında RADAR yolu: `uploads/<upload_id>` → `jobs/<job_id>/in/` kopyası → subprocess
(`radar` env) → `jobs/<job_id>/out/RADAR_infer_results_<job_id>.csv` → parse → `findings[]`.

### 5.2 ClinFusion sekmesi (sohbet)

```
1. (Opsiyonel) Kullanıcı dosya ekler
   → POST /api/upload → {upload_id}
2. Kullanıcı mesaj gönderir (ekler konuşma boyunca korunur)
   → POST /api/chat {prompt, history:[{role,text}], attachment_ids:[...]}
   ← {job_id}
3. Frontend yoklar
   → GET /api/chat/{job_id}
   ← {status:"running"} → {status:"done", reply:"..."}
4. "Yeni sohbet" → POST /api/chat/reset; geçmiş ve ekler sıfırlanır
```

Backend, `attachment_ids`'i gerçek dosya yollarına çevirir ve worker'a **yerel dosya yolu** olarak
gönderir (binary JSON'da taşınmaz). Worker, `apply_chat_template` ile çok turlu prompt kurar
(geçmiş metin + bu turun ekleri) ve 32B ile üretir.

---

## 6. API sözleşmeleri

Tümü JSON; dosya içerenler multipart.

| Method | Yol | Gövde / Parametre | Yanıt |
|---|---|---|---|
| GET | `/api/status` | – | `{gpu, radar_ready, clinfusion_ready, busy, current_job, queue_len}` |
| POST | `/api/upload` | multipart `file` | `{upload_id, kind, filename, n_slices?}` |
| GET | `/api/nifti/{upload_id}/slice` | `?idx=N&wc=&ww=` | PNG + `X-Total-Slices` (WC/WW penceresi, varsayılan 0/2000) |
| POST | `/api/radar` | `{upload_id}` | `{job_id}` |
| GET | `/api/radar/{job_id}` | – | `{status, progress, findings, csv_url, error}` |
| GET | `/api/radar/{job_id}/csv` | – | CSV dosyası |
| POST | `/api/chat` | `{prompt, history, attachment_ids}` | `{job_id}` |
| GET | `/api/chat/{job_id}` | – | `{status, reply, error}` |
| POST | `/api/chat/reset` | – | `{ok:true}` |

**Hata durumları:** `400` (geçersiz tür/boyut), `404` (upload_id/job yok veya NIfTI değil),
`503`/`500` (worker yükleniyor / worker hatası). Job durumları: `queued → running → done | error`.

Örnek — RADAR tamamlanmış yanıt:
```json
{
  "status": "done",
  "progress": null,
  "findings": [
    {"name": "Heart_Cardiomegaly", "label": "心脏_心影（脏）增大 (Heart_Cardiomegaly)", "score": 0.833},
    {"name": "Aorta_Calcification", "label": "主动脉_钙化 (Aorta_Calcification)", "score": 0.697}
  ],
  "csv_url": "/api/radar/ab12cd34ef56/csv",
  "error": null
}
```

Örnek — sohbet tamamlanmış yanıt:
```json
{"status": "done", "reply": "2 + 2 equals 4.", "error": null}
```

---

## 7. İç sözleşme: Backend ↔ Worker

Worker yalnız **localhost** (`127.0.0.1:8100`) dinler; LAN'a kapalıdır ve yalnız backend kuyruğu
üzerinden çağrılmalıdır (GPU'yu RADAR ile paylaşır).

| Method | Yol | Gövde | Yanıt |
|---|---|---|---|
| GET | `/health` | – | `{ready, loading, error, model}` |
| POST | `/generate` | `{history:[{role,text}], prompt, attachment_paths:[...]}` | `{reply}` |

- `ready=false, loading=true` → model yükleniyor (backend 503 gösterir).
- `error` doluysa worker süreci `os._exit(1)` ile kapanır; `run.sh` supervisor 10 sn sonra yeniden başlatır.

---

## 8. GPU kuyruğu ve eşzamanlılık

- Tek GB10 var; aynı anda tek iş GPU'ya biner.
- Backend'de **tek `asyncio.Lock`** RADAR subprocess'ini ve ClinFusion isteklerini **serileştirir**.
- Worker içinde ayrıca bir **generation kilidi** vardır (doğrudan çağrılırsa bile çakışma olmaz).
- `/api/status` → `busy`, `current_job`, `queue_len` ile durum gösterilir.
- 32B resident kaldığı için RADAR yalnız kısa süreliğine (~2GB) yüklenir; ikisi aynı anda hesaplamaz.

---

## 9. Dizin yapısı, portlar, ortamlar

```
~/medportal/
├── backend/            FastAPI backend (app/ + tests/)
├── worker/             ClinFusion-32B worker
├── frontend/           index.html, app.js, styles.css
├── scripts/smoke.sh    uçtan uca duman testi
├── run.sh              worker'ı (health bekleyerek) + backend'i başlatır (PID kayıtlı)
├── stop.sh             supervisor + worker + backend durdurur (port doğrulamalı)
├── status.sh           süreç/port/health raporu
├── workspace/          çalışma zamanı: uploads/ ve jobs/ (gitignore)
├── logs/               worker.log, backend.log (gitignore)
├── docs/               spec + plan
├── KULLANIM.md         operasyonel kılavuz
└── DOKUMANTASYON.md    bu dosya
```

| Ortam | Python | İçerik |
|---|---|---|
| `medportal` | 3.11 | FastAPI, uvicorn, httpx, nibabel, numpy, pillow |
| `radar` | 3.10 | torch, transformers 4.25, monai, SimpleITK |
| `clinfusion` | 3.11 | torch 2.14+cu130, transformers 4.57, flash-attn 2.8.3 (aarch64), ray, monai |

| Port | Servis | Erişim |
|---|---|---|
| 8080 | Backend + SPA | LAN (`0.0.0.0`) |
| 8100 | ClinFusion worker | yalnız localhost |

---

## 10. Bellek ve performans

| Model | Yükleme | Çalışma belleği (peak) | Tipik süre |
|---|---|---|---|
| RADAR | ~10-30 sn (istek başına) | ~2GB | demo 1 hasta ~4 sn |
| ClinFusion-32B | ~13 dk (bir kez) | ~89GB | cevap ~saniyeler–dakikalar |

128GB unified bellek: 32B için ~39GB marj kalır. Uzun context / çok görüntü / batch>1'de OOM riski.

---

## 11. Sınırlar, varsayımlar, güvenlik

- **Kapsam:** araştırma/demo; tek kullanıcı grubu, **LAN, girişsiz**. Üretim için kimlik doğrulama/kuyruk/ölçeklenme gerekir.
- **Yükleme:** yalnız `.nii.gz/.nii` (RADAR/3D) ve `.jpg/.jpeg/.png` (2D); sınır **500MB**; tür/boyut okumadan **önce** doğrulanır.
- **`upload_id`** biçimi `[0-9a-f]{12}` ile doğrulanır (path traversal engeli).
- **Frontend XSS:** tüm sunucu verisi `textContent` ile basılır.
- **Tailwind CDN:** ✅ kaldırıldı (2026-09-28) — arayüz el yazımı CSS'e taşındı, internet gerekmez.
- **HTML/JS:** tarayıcı arayüzü; model çıktıları tıbbi karar için tek başına yeterli değildir.
- RADAR çıktısı **146 bulgu** içerir (148 değil); etiketler iki dilli (`中文 (English)`).

---

## 12. Sorun giderme (özet)

Ayrıntı için [`KULLANIM.md`](KULLANIM.md).

| Belirti | Neden / Çözüm |
|---|---|
| `/api/status` boş, sohbet 503 | Worker 32B'yi yüklüyor; `curl :8100/health` ile `ready` bekle (~13 dk) |
| Arayüz stilsiz | (eski Tailwind sorunu — kaldırıldı; hâlâ olursa styles.css 404 mü kontrol edin) |
| Yükleme 400 | Tür/boyut kuralı; NIfTI beklenir |
| `PermissionError ... ~/.triton/cache` | `sudo chown -R nvidia:nvidia ~/.triton` |

---

## 13. Referanslar

- Tasarım: `docs/superpowers/specs/2026-09-21-medportal-web-ui-design.md`
- Uygulama planı: `docs/superpowers/plans/2026-09-21-medportal-web-ui.md`
- Operasyon: `KULLANIM.md`
- RADAR: `~/radar/KULLANIM.md`, `https://github.com/alibaba-damo-academy/damo-radar`
- ClinFusion: `~/ClinFusion/KULLANIM.md`, `https://github.com/Alibaba-DAMO-Academy/ClinFusion`
