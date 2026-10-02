# MedPortal — RADAR + ClinFusion Web Arayüzü

İki CLI projesini tek web portalında sunar:

- **RADAR:** karın BT (`.nii.gz`) → aksiyel kesit görüntüleyici + 146 bulgu olasılık tablosu (eşik, sıralama, CSV).
- **ClinFusion-32B:** 2D görüntü / 3D NIfTI / metin ile çok turlu sohbet.

Hedef ortam: NVIDIA DGX Spark (`192.168.1.162`, aarch64, GB10, 128GB unified).
Uygulamanın tamamı Spark'ta çalışır; tarayıcıdan `http://192.168.1.162:<port>` ile erişilir.

Tasarım dokümanı: `docs/superpowers/specs/2026-09-21-medportal-web-ui-design.md`
Kullanım rehberi (RADAR / ClinFusion: ne alır, ne üretir): `user-guide.md`
Kullanım kılavuzu: `KULLANIM.md`
Sistem ve model dokümantasyonu: `DOKUMANTASYON.md`

## Çalıştırma

```bash
cd ~/medportal
./run.sh          # 32B model yüklenirken bekler (dakikalar sürebilir)
```

Ardından tarayıcıdan `http://192.168.1.162:8080` adresini açın.

Uçtan uca duman testi (backend/worker çalışırken):

```bash
./scripts/smoke.sh   # beklenen çıktı: findings: 146 ve SMOKE OK
```

Ayrıntılı kullanım, API uçları ve sorun giderme için `KULLANIM.md`'ye bakın.
