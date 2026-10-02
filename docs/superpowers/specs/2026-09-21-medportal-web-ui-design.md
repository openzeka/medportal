# MedPortal — RADAR + ClinFusion Web Arayüzü (Tasarım)

- **Tarih:** 2026-09-21
- **Durum:** Onaylandı (uygulama planı bekliyor)
- **Kapsam:** Araştırma/demo (tek kullanıcı grubu, LAN, girişsiz)

## 1. Amaç ve hedef
İki mevcut CLI projesini (RADAR ve ClinFusion-32B) tek bir web portalında sunmak:

- **RADAR:** Karın BT (`.nii.gz`) yükle → aksiyel kesit görüntüleyici + 146 bulgu olasılık tablosu (eşik, sıralama, CSV indir).
- **ClinFusion-32B:** 2D görüntü / 3D NIfTI / metin ile çok turlu sohbet; "yeni sohbet".

**Hedef ortam:** NVIDIA DGX Spark, `192.168.1.162` (aarch64, GB10, 128GB unified). Tarayıcı aynı LAN'dan `http://192.168.1.162:<port>` ile erişir. Uygulamanın tamamı Spark'ta çalışır; geliştirme makinesinde hiçbir bileşen çalışmaz.

### Non-goals (V1 dışı)
- Kalıcı geçmiş / oturum listesi
- Çoklu dosya toplu analiz
- Kullanıcı girişi, rol yönetimi, audit
- Çok kullanıcılı eşzamanlılık

## 2. Kritik kısıt: iki ayrı conda ortamı
| Proje | Env | Python | transformers |
|---|---|---|---|
| RADAR | `~/miniconda3/envs/radar` | 3.10 | 4.25 |
| ClinFusion | `~/miniconda3/envs/clinfusion` | 3.11 | 4.57 |

Aynı env'de çalışamazlar. Bu nedenle portal **orkestratör**dür; modelleri kendi env'lerinde ayrı süreçlerde çalıştırır.

## 3. Mimari
```
[Tarayıcı, LAN] ──HTTP──> Portal Backend (FastAPI, env: medportal)  :8080
                               │
                ┌──────────────┴───────────────┐
                ▼                              ▼
    RADAR subprocess (env: radar)     ClinFusion Worker (env: clinfusion)
    istek başına, ~2GB                localhost HTTP, 32B resident (~89GB)
```

- **ClinFusion-32B** bir kez yüklenir (startup ~10 dk), oturum boyunca bellekte kalır.
- **RADAR** istek başına subprocess olarak çalışır (küçük model; ~10–30 sn yükleme + inference).

## 4. Bileşenler
### 4.1 Backend — `~/medportal/backend` (env: `medportal`)
- FastAPI + uvicorn; SPA statik dosyalarını servis eder.
- Yükleme (multipart), dosya doğrulama (`.nii.gz`, boyut sınırı).
- RADAR orkestrasyonu: subprocess + CSV parse.
- ClinFusion worker'a HTTP iletimi.
- NIfTI aksiyel kesit PNG üretimi (nibabel + Pillow).
- **Tek GPU kilidi / kuyruğu** ve `/api/status`.
- Gereken paketler: `fastapi`, `uvicorn`, `python-multipart`, `nibabel`, `numpy`, `pillow`, `httpx`.

### 4.2 ClinFusion worker — `~/medportal/clinfusion_worker.py` (env: `clinfusion`)
- `MedEvalKitAdapter` ile 32B'yi bir kez yükler.
- localhost HTTP sunar (FastAPI/uvicorn, env: `clinfusion`).
- **Çok turlu** sohbet: `processor.apply_chat_template` ile tüm mesaj geçmişini (user/assistant) işler; ekler (2D PIL / NIfTI hacmi) adapter içi yardımcılarla işlenir.
- Not: adapter'in `generate()`'i tek turludur; worker kendi üretim fonksiyonunu içerir.

**Worker HTTP sözleşmesi (iç ağ):**
| Method | Yol | Gövde | Yanıt |
|---|---|---|---|
| GET | `/health` | – | `{ready, model, loading}` |
| POST | `/generate` | `{history:[{role,text}], prompt, attachment_paths:[]}` | `{reply}` |

### 4.3 Frontend — `~/medportal/frontend`
- Tek sayfa: `index.html`, `app.js`, `styles.css` (Tailwind CDN; Node gerekmez).
- Üst: durum çubuğu (GPU, RADAR hazır, ClinFusion-32B hazır) + sekmeler.
- RADAR sekmesi: dropzone, kesit görüntüleyici (slider), bulgu tablosu (arama, eşik slider, skora göre sıralama), CSV indir.
- ClinFusion sekmesi: sohbet alanı, dosya eki, metin girişi, "Yeni sohbet".

### 4.4 RADAR entegrasyon detayı
**Çağrı (mevcut CLI, değiştirilmez):**
```
~/miniconda3/envs/radar/bin/python ~/radar/RADAR_inference/inference_demo.py \
    --img_dir <workspace>/<job_id>/in \
    --save_dir <workspace>/<job_id>/out \
    --save_tag <job_id>
```
- Girdi klasörü `<job_id>/in/` içine **yalnızca tek** `.nii.gz` konur (CLI klasör bekler; `os.listdir` uzantı filtrelemez).
- Çıktı dosyası: `<workspace>/<job_id>/out/RADAR_infer_results_<job_id>.csv`
- CSV: `utf-8-sig` (BOM) ile yazılır, tek satır veri. İlk kolon `file_name`; diğer **146** kolon `"{中文}_{...} ({English})"` biçiminde bulgu adı + skor (0–1).
- Parse: `pandas.read_csv(..., encoding="utf-8-sig")`; `findings[] = [{name: <English kısım>, label: <tam kolon>, score: <float>}]`.
- Referans: `~/radar/RADAR_inference/inference_demo.py:606` (argümanlar), `:552-555` (CSV yazımı), `:156` (146 bulgu).

**ClinFusion worker kod referansları:** adapter `~/ClinFusion/custom_model/medevalkit_adapter_qwen3_vl.py` (`MedEvalKitAdapter.generate`, `_pre_sample_slices_from_volume`, `_read_pil_image`, `processor`); örnek test: `~/ClinFusion/eval/test/test_clinfusion_3d.yaml`.

## 5. API
Tüm istekler JSON; **dosya içeren yüklemeler ayrı multipart endpoint** ile yapılır.

| Method | Yol | Gövde / Parametre | Yanıt |
|---|---|---|---|
| GET | `/api/status` | – | `{gpu, radar_ready, clinfusion_ready, busy, current_job, queue_len}` |
| POST | `/api/upload` | multipart `file` (`.nii.gz`/`.jpg`/`.png`) | `{upload_id, kind, filename, n_slices?}` |
| POST | `/api/radar` | JSON `{upload_id}` | `{job_id}` |
| GET | `/api/radar/{job_id}` | – | `{status, progress, findings[], csv_url, error}` |
| GET | `/api/radar/{job_id}/csv` | – | CSV dosyası |
| GET | `/api/nifti/{upload_id}/slice` | `?idx=N` | PNG + header `X-Total-Slices` |
| POST | `/api/chat` | JSON `{history:[{role,text}], prompt, attachment_ids:[]}` | `{job_id}` |
| GET | `/api/chat/{job_id}` | – | `{status, reply, error}` |
| POST | `/api/chat/reset` | JSON `{conversation_id?}` | `{ok}` |

### 5.1 Ek (attachment) taşıma
- Binary veri **asla** JSON içinde taşınmaz. `/api/upload` dosyayı `workspace/<upload_id>/` altına yazar ve bir `upload_id` döndürür.
- `/api/chat`, `attachment_ids[]` listesi alır (dosya yolları backend tarafında çözülür).
- Backend → worker iletiminde de binary taşınmaz: worker ile backend **aynı dosya sistemini** paylaşır; backend worker'a **yerel dosya yollarını** JSON olarak gönderir (`attachment_paths[]`). Worker `MedEvalKitAdapter` yardımcılarını bu yollarla çağırır.

### 5.2 Uzun süren işler (job modeli)
32B üretimi dakikalar sürebildiğinden `/api/chat` **senkron değil, job tabanlıdır** (RADAR ile aynı desen):
- `POST /api/radar` ve `POST /api/chat` bir `job_id` döndürür.
- İstemci `GET /api/radar/{job_id}` / `GET /api/chat/{job_id}` ile periyodik yoklar (1–2 sn).
- `status`: `queued | running | done | error`.
- V1'de streaming/SSE yok.

## 6. GPU kuyruğu
- Backend'de tek `asyncio.Lock`: RADAR subprocess ve ClinFusion istekleri sırayla.
- Durum: `busy` + `current_job` (radar/clinfusion) + kuyruk uzunluğu.

## 7. Veri akışı ve depolama
- Yükleme: `~/medportal/workspace/uploads/<upload_id>/<filename>` (uzantı korunur).
- İş: `~/medportal/workspace/jobs/<job_id>/` (`in/`, `out/`, `job.log`).
- **ID ayrımı:** `upload_id` (yüklenen dosya) ile `job_id` (çalıştırma) farklıdır. `POST /api/radar`, `upload_id`'yi alır ve ilgili dosyayı `jobs/<job_id>/in/` içine **kopyalar** (CLI klasör beklediği ve klasörde yalnızca tek `.nii.gz` olması gerektiği için).
- ClinFusion ekleri için kopya gerekmez; worker `uploads/<upload_id>/...` yolunu doğrudan okur.
- Kalıcı geçmiş yok; V1'de workspace temizliği manuel/opsiyonel.

## 8. Hata yönetimi ve sabit varsayılanlar
- Subprocess exit≠0 → log kuyruğu + kullanıcıya hata.
- Worker düşerse → backend yeniden başlatmayı dener; durum "model yükleniyor".
- Yükleme: yalnız `.nii.gz` (RADAR/3D) ve `.jpg/.png` (ClinFusion 2D).

**Sabit varsayılanlar (V1):**
| Parametre | Değer |
|---|---|
| Yükleme boyut sınırı | 500 MB |
| RADAR job timeout | 600 sn |
| ClinFusion job timeout | 1800 sn |
| `max_new_tokens` (ClinFusion) | 4096 |
| `batch_size` (ClinFusion) | 1 |
| Polling aralığı (istemci) | 1–2 sn |
| Kesit toplamı iletimi | `GET /api/nifti/{id}/slice` yanıtı `X-Total-Slices` header'ı |

## 9. Test
- **Unit (pytest):** API + kuyruk; subprocess/worker mock'lu (GPU gerekmez).
- **Entegrasyon smoke:** RADAR demo case; ClinFusion metin + 3D örnek; JSON şema doğrulama.
- Tarayıcıda manuel kontrol.

## 10. Dağıtım
- `~/medportal/` (backend + worker + frontend + `run.sh` + `docs/`).
- `run.sh`: worker'ı başlatır (health'e kadar bekler) → backend'i başlatır.
- Port: boş bir port (ör. 8080) Spark'ta; erişim `http://192.168.1.162:8080`.
- Yeni env: `medportal` (Python 3.11 + §4.1 paketleri).
- Geliştirme makinesinden Spark'a `rsync` ile aktarım.

## 11. Riskler
- 32B resident ~89GB; portal + OS ile bellek marjı dar → batch=1, düşük `max_new_tokens`.
- Çok turlu sohbet için adaptör dışı özel üretim kodu (bakım yükü).
- İlk 32B yükleme ~10 dk; startup deneyimi için "yükleniyor" durumu şart.
- İki env/alt süreç orkestrasyonu karmaşıklığı.

## 12. Kesinleşen kararlar ve açık uçlar
**Kesinleşen:**
- Worker protokolü: **localhost HTTP** (`/health`, `/generate`) — karar verildi.
- Backend env: yeni `medportal` (Python 3.11, §4.1 paketleri) — karar verildi.
- Uzun işler job tabanlı, polling (SSE yok) — karar verildi.

**Açık:**
- Port: kurulumda boş port seçilecek (ör. 8080).
