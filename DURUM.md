# MedPortal — Durum / Devir Notları

**Tarih:** 2026-09-21 (ilk kurulum) · **2026-09-28 güncellemesi** · **Branch:** `medportal-v1` (henüz `master`'a merge edilmedi)

## Çalışan sistem
| | |
|---|---|
| Backend | `http://192.168.1.162:8080` (env `medportal`, PID `logs/backend.pid`) |
| Worker | `127.0.0.1:8100` — ClinFusion-32B **resident**, `ready:true` |
| Başlatma | `cd ~/medportal && ./run.sh` (worker 32B'yi yükler, ~14 dk; sonra backend) |
| Durdurma | `./stop.sh` (`--force`: KILL) — supervisor + worker + backend, port doğrulamalı |
| Durum | `./status.sh` — süreç/port/health raporu |
| Loglar | `~/medportal/logs/worker.log`, `logs/backend.log`, `logs/run.log` |

> Not: 32B resident olduğu için Spark/worker kapanırsa bir sonraki açılışta yeniden yükleme ~14 dk sürer.

## Yapılan iş (özet)
- İki proje tek portala bağlandı: **RADAR** (146 bulgu, ayrı `radar` env'de subprocess) + **ClinFusion-32B** (resident worker, `clinfusion` env).
- Backend (FastAPI, `medportal` env): yükleme, NIfTI kesit (WC/WW pencereleme), RADAR/chat job kuyruğu (tek GPU), statik SPA.
- Frontend: RADAR sekmesi (kesit + eşik/sıralama/CSV), ClinFusion sekmesi (çok turlu sohbet + 2D/3D ek, thumbnail önizleme).
- Dokümanlar: `README.md`, `KULLANIM.md`, `DOKUMANTASYON.md`, `docs/superpowers/{specs,plans}/`.
- Testler: **74 pytest** (backend), GPU'suz çalışır. `scripts/smoke.sh` uçtan uca.

### 2026-09-28 güncellemesi
- **Arayüz tam yenileme:** koyu radyoloji konsolu, **OpenZeka** markası (`frontend/assets/openzeka-logo*.png`, favicon); el yazımı CSS tasarım sistemi — **Tailwind CDN tamamen kaldırıldı** (offline LAN'de eksiksiz çalışır).
- **Görüntüleyici:** WC/WW ön ayarları (Yumuşak doku/Akciğer/Kemik/Tam), zoom+sürükle, klavye gezinme (←→/PgUp/PgDn/Home/End), kesit numarası girişi, HUD.
- **Bulgu paneli:** skor barları, "öne çıkanlar" çipleri, organ gruplama, eşik preset'leri, sayaç rozeti.
- **Sohbet:** markdown-lite güvenli render, kopyala butonu, düşünüyor animasyonu, toast bildirimleri (alert yok).
- **API:** `GET /api/nifti/{id}/slice` artık `?wc=&ww=` alır (varsayılan 0/2000 = eski davranış); +6 yeni test.
- **Script'ler:** `stop.sh` + `status.sh` eklendi; `run.sh` PID dosyaları yazar, periyodik ilerleme gösterir.
- **Sağlamlaştırma:** RADAR subprocess'inde geçici CUDA OOM'a **tek seferlik otomatik yeniden deneme** + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Doğrulananlar (canlı)
- 2026-09-21: `/api/status` `clinfusion_ready:true`; `smoke.sh` → `findings: 146`, `SMOKE OK`; `.txt`/`.png` reddi; sohbet (metin + 3D ek + çok turlu + reset).
- 2026-09-28: `smoke.sh` → `findings: 146`, **`SMOKE OK`** (74 test geçti); Playwright ile uçtan uca arayüz testi (98MB demo BT → 146 bulgu, pencere/zoom/grup ekranları); **canlı 32B sohbet cevabı** doğrulandı; `run.sh` → `stop.sh` bisikleti test edildi.

## İleriye dönük açık maddeler
- RADAR, 32B yükleme bitişinin hemen ardından ilk denemede nadiren CUDA OOM alıyor → retry eklendi; **tam çözüm** için worker boşta iken periyodik `torch.cuda.empty_cache()` (uygulanmadı).
- `/api/status` worker hatasını "yükleniyor" gibi gösteriyor (worker `error` alanı arayüze taşınmıyor).
- `queue_len` çalışan işi de sayıyor (bekleyen-only değil).
- Ölü kod/atıl config: `schemas.JobResponse`, `config.MEDPORTAL_PY/CLINFUSION_PY/...`.
- Worker bağımlılıkları pinlenmedi; `worker/requirements.txt` eklenebilir (clinfusion env'e fastapi/uvicorn kuruldu).
- Birkaç uç-durum testi eksik (aşırı yükleme reddi, bilinmeyen job/ek, gerçek 500 yolu).
- `medportal-v1` → `master` merge edilmedi.

## Yedek (2026-09-28)
Bu çalışma sonunda sistemin **tam yedeği** alındı (medportal + radar + ClinFusion,
model ağırlıkları dahil ~185GB): geliştirme makinesindeki `damo-radar-exp/` altında
`medportal/`, `radar/`, `ClinFusion/` klasörleri. Kaynak gerçek: `spark1:~/medportal` (git repo).

## Hızlı referans
```bash
# durum
./status.sh
# başlat / durdur
./run.sh ; ./stop.sh
# test
cd ~/medportal && /home/nvidia/miniconda3/envs/medportal/bin/python -m pytest backend/tests -q
./scripts/smoke.sh
# log izleme
tail -f ~/medportal/logs/worker.log
```