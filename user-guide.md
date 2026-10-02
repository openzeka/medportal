# MedPortal — Kullanım Kılavuzu

`http://192.168.1.162:8080` adresinde çalışan uygulama iki bağımsız sistemi tek ekranda toplar.
Aşağıda her sistem için ayrı ayrı: **ne alıyor**, **ne üretiyor**, **nasıl kullanılıyor**.

Ortak kural: iki sistem tek bir GPU'yu paylaşır, aynı anda yalnızca biri çalışır. Durum çubuğu
hangi işin sırada olduğunu gösterir. Sistemler hakkında teknik derinlik için
[`DOKUMANTASYON.md`](DOKUMANTASYON.md), sunucu komutları için [`KULLANIM.md`](KULLANIM.md).

---

# 1. RADAR — karın BT'de bulgu tespiti

Görevi tek cümleyle: **karın BT'de 146 bulgu için olasılık skoru hesaplar.**

## Ne alır?

| | |
|---|---|
| **Dosya** | `.nii.gz` veya `.nii` — 3D NIfTI formatında karın BT |
| **Boyut** | En fazla 500 MB |
| **Beklenen içerik** | **Kontrastlı** karın BT, aksiyel kesitler, HU değerleri |

Bu protokol dışındaki girdiler (kontrastsız BT, göğüs BT, farklı vücut bölgesi) modelin
eğitildiği dağılımın dışındadır; çıkan skorlar güvenilir olmaz.

## Ne üretir?

| | |
|---|---|
| **Bulgu tablosu** | 146 bulgunun her biri için 0–1 arası pozitif olasılık skoru, yüksekten düşüğe sıralı |
| **Filtrelenebilir liste** | Eşik kaydırıcısı, arama kutusu, organ gruplama |
| **CSV** | Tüm skorlar indirilebilir dosya olarak dışa aktarılır |
| **Kesit görüntüleyici** | Yüklediğiniz BT'nin aksiyel kesitleri, ayarlanabilir pencere ile |

**Skor nasıl okunur:** 0–1 arası bir olasılıktır — "bu bulgu var" değil, "bu vakada bu bulgunun
bulunma olasılığı şu kadar". Karar, klinik değerlendirme ve radyoloj raporu ile verilir; bu ekran
karar desteğidir.

## Nasıl kullanılır?

1. **RADAR** sekmesini açın.
2. Soldaki kutuya BT dosyasını **sürükleyip bırakın** veya tıklayıp seçin.
3. Kesit görüntüleyici açılır ve analiz arka planda başlar. İlk analiz model yüklenmesi
   nedeniyle biraz uzun sürebilir; sonraki analizler hızlıdır.
4. Analiz bitince bulgu tablosu dolar. Sağ üstteki sayaç **kaç bulgunun / 146** gösterildiğini yazar.
5. **Download CSV** ile tüm sonuçları indirin.

**Eşiği ayarlama:** kaydırıcı ile yalnızca belirli bir olasılığın üstündeki bulguları görürsünüz.
Hazır ayarlar: `0.30` (duyarlı, çok bulgu), `0.50` (varsayılan), `0.70` (seçici). **Düşük eşik daha
fazla bulgu gösterir ama gürültüyü de artırır.**

**Organ gruplama:** `Group by organ` kutusu sonuçları organa göre toplar, en yüksek skorlu grup en üstte.

**Arama:** `Search findings…` kutusuna bulgu adı yazarak tabloyu daraltın.

**Kesitlerle çalışmak:** pencere düğmeleri (`Soft tissue`, `Lung`, `Bone`, `Full`) dokunun farklı
görünmesini sağlar. Klavyeyle de gezebilirsiniz: `←` `→` dilim değiştirir, `PgUp` `PgDn` ±10 dilim,
`Home` / `End` ilk/son dilim, tekerlek yakınlaştırır, sürükleyerek kaydırır.

## Sınırlar

- Bir analiz en fazla **10 dakika** sürer; aşarsa iş hata döner.
- Aynı anda tek RADAR işi çalışır.
- GPU belleği model yüklemesi sırasında sıkışırsa iş otomatik **bir kez** tekrar denenir.
- Çıktı yalnızca bu 146 bulgu ile sınırlıdır; rapor metni üretmez.

---

# 2. ClinFusion-32B — görüntüler üzerine tıbbi sohbet

Görevi tek cümleyle: **yüklediğiniz görüntüleri ve metni konuşturup serbest metin cevap üretir.**

Bu model sohbet tutar: bir kez sorduğunuz şeyin bağlamı sonraki turlarda hatırlanır.

## Ne alır?

| | |
|---|---|
| **Zorunlu** | Metin sorusu (Türkçe veya İngilizce — cevap soru dilini izler) |
| **İsteğe bağlı** | 2D görüntü: `.jpg`, `.jpeg`, `.png` — birden fazla eklenebilir |
| **İsteğe bağlı** | 3D BT: `.nii.gz`, `.nii` |
| **Kendiliğinden** | Önceki tur geçmişi (siz göndermeden yönetilir) |
| **Boyut** | Dosya başına en fazla 500 MB |

Ekleri 📎 düğmesiyle, cevabınızın ilk cümlesinden önce ekleyebilirsiniz; model ekleri ve soruyu
birlikte görür. 2D görüntüler olduğu gibi iletilir. 3D BT eklediğinizde model görüntünün tamamını
değil, **temsili 4 dilimi** ve hacim verisini birlikte değerlendirir — bu yüzden belli bir bölge
hakkında soru soruyorsanız, o bölgeyi 2D görüntü olarak kesip eklemek daha isabetli sonuç verir.

## Ne üretir?

| | |
|---|---|
| **Cevap** | Serbest metin — düz metin, madde listesi, tablo ve kod bloğu desteklenir |
| **Ek listesi** | Gönderdiğiniz her mesajın altında, o mesajda kullandığınız eklerin adları |
| **Kopyalama** | Her cevabın başlığında `Copy` düğmesi |
| **Kesildiğinde uyarı** | Cevap 16.384 tokenlık sınıra dayanırsa sonuna "uzunluk sınırına ulaştı" notu düşer |

Cevap kendiliğinden durur; bir cevabın uzunluğu sorunuzun gerektirdiği kadardır. Kısa soruya
kısa cevap gelmesi bir hata değildir.

## Nasıl kullanılır?

1. Üstteki **ClinFusion** sekmesine geçin.
2. Görüntü kullanacaksanız `📎` ile ekleyin — eklendiğini sağdaki **Attachments** panelinde
   önizlemeyle görürsünüz.
3. Sorunuzu yazın ve **Enter** ile gönderin. (`Shift+Enter` satır başına geçer.)
4. Cevap gelirken "thinking" göstergesi, ardından cevap metni görünür.
5. Cevabı `Copy` ile alabilir, panelden eklere göz atabilirsiniz.

**Sohbet akışı:** Aynı ekler mesaj gönderildikten sonra listede kalır ve `✓ sent` işareti alır.
Yeni bir soruya tekrar ekleyip kullanabilir, `×` ile listeden çıkarabilirsiniz.

**Daha iyi cevap için:** konuyu bölerek sorun. Uzun ve yapılandırılmış bir yanıt istiyorsanız
bunu açıkça söyleyin — örneğin "20 maddelik bir kontrol listesi yaz, her maddeyi iki cümleyle
açıkla". Böyle bir istek, kısa bir soruya verilecek kısa cevaptan çok daha iyidir.

**Sohbeti sıfırlamak:** `+ New chat` düğmesi geçmişi ve ekleri temizler, yeni sohbete başlar.

## Sınırlar

- Bir cevap en fazla **30 dakika** sürebilir (çok uzun cevaplarda geçerlidir).
- Cevap bütçesi **16.384 token** ile sınırlıdır. Not düşerse cevap tamamlanmamıştır; aynı konuyu
  sorarak kaldığınız yerden devam edebilirsiniz.
- İlk cevap **birkaç dakika** sürebilir: 32B modeldir ve büyük cevaplarda üretim yavaşlar.
- Bir iş sırasında başka bir iş gönderirseniz kuyruğa alınır.