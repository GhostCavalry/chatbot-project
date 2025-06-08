Chatbot Projesi - Açık Artırma Platformu Asistanı 🤖
MTH kodlu "Üretken Yapay Zeka Yardımı ile Chatbot Geliştirme Temelleri" dersi ödevi için hazırlanmıştır


📌 Proje Özeti

Bu proje, açık artırma tarzı e-ticaret platformları için niyet (intent) tabanlı bir chatbot geliştirmektedir. 5 temel etkileşim kategorisine odaklanmıştır:

Selamlama
Reddetme
Vedalaşma
Satın Alma/Satma İşlemleri
Güvenlik


🛠️ Teknik Uygulama

Veri Hazırlığı

1.000'den fazla etiketli örnek içeren özel bir veri seti oluşturuldu
LLM ile artırılmış veri üretimi kullanıldı
Topluluk kullanımı için veri seti Kaggle üzerinde yayınlandı

![image](https://github.com/user-attachments/assets/a7151bb5-ae66-437f-936a-aa8d4c21d153)


![image](https://github.com/user-attachments/assets/b0224fb6-7ca2-4a5f-a7be-58dfa7c44d6b)


🚀 Streamlit Arayüz Özellikleri

Gerçek zamanlı sohbet ile mesaj geçmişi
Niyet görselleştirme (sınıflandırma güven skoru)
Performans metrikleri gösterge paneli
Mobil uyumlu tasarım

![image](https://github.com/user-attachments/assets/7ef92c02-3fda-46a4-b303-280508ddaf71)

![image](https://github.com/user-attachments/assets/84d438d1-7b35-4c9e-b504-b16dc8541c90)

**Streamlit Arayüzü Özeti**  

Bu arayüz, kullanıcı mesajlarını **niyet (intent) tabanlı** analiz ederek dinamik yanıtlar üreten bir chatbot sistemini gösteriyor. Kullanıcı mesajı girildiğinde:  
1. **Niyet Tahmini**: Model (Gemini), mesajın içeriğini analiz ederek en olası intent'i belirliyor (örn. *"Satın Alma/Satma İşlemleri"*).  
2. **Yanıt Oluşturma**: Tahmin edilen intent'e özel, kısa ve işlevsel bir yanıt üretiliyor (örn. teklif verme talimatları).  
3. **Şeffaf Karar Süreci**: Altta yer alan **"Model Nasıl Karar Verdi?"** bölümünde, modelin benzerlik skorlarıyla (örn. *0.61*) hangi eğitim verilerine dayanarak karar aldığı gösteriliyor. Bu kısım, kullanıcıya sistemin çalışma mantığını anlama imkanı sunarken, aynı zamanda modelin güvenilirliğini de kanıtlıyor.  

**Öne Çıkan Özellikler**:  
- 🎯 **Niyet odaklı** dinamik yanıtlar  
- 🔍 **Şeffaf karar mekanizması** (benzerlik skorları ve referans alınan örnekler)  
- 💬 **Kullanıcı dostu** sohbet arayüzü  

*Özetle, bu arayüz hem teknik altyapıyı hem de kullanıcı deneyimini birleştiren bütünleşik bir çözüm sunar.*


❓ Neden Gemini ve ChatGPT Kullandım?

Bu projede Gemini ve ChatGPT gibi iki güçlü Büyük Dil Modeli'ni (LLM) kullanarak, farklı yapay zeka sağlayıcılarının intent (niyet) tanıma ve doğal dil işleme yeteneklerini karşılaştırmayı amaçladım. Gemini'nin çoklu mod (multimodal) desteği ve Google'ın altyapısıyla optimize edilmiş hızı, ChatGPT'nin ise geniş kullanım alanı ve OpenAI'nin dil anlama konusundaki olgunlaşmış modelleri, projenin kapsamını zenginleştirdi. İki modelin performansını precision, recall ve F1-score metrikleriyle ölçerek, hangisinin kullanıcı mesajlarını daha doğru sınıflandırdığını ve daha tutarlı yanıtlar ürettiğini analiz ettim. Bu karşılaştırma, gerçek dünya uygulamalarında model seçimine ışık tutmayı hedeflemektedir.

(Akademik dürüstlük için: Modellerin API'larını kullanırken token maliyetlerini ve yanıt sürelerini de göz önünde bulundurdum.)

📊 Sonuçlar

![image](https://github.com/user-attachments/assets/50b03dd7-5664-449e-be1b-c2404a38df40)

Model Performans Yorumları

🔍 1. Karışıklık Matrisleri (Confusion Matrix)

Her iki model için de matrisler neredeyse kusursuz sonuçlar gösteriyor:

![image](https://github.com/user-attachments/assets/05ce412e-2183-4b7b-93d0-5266bd081e33)

Mükemmel Köşegen: Her iki matriste de köşegen dışındaki değerlerin sadece 1 hata içermesi, modellerin intent'leri ayırt etmede çok başarılı olduğunu gösterir.
Tek Hata: Güvenlik sınıfından bir örneğin Reddetme olarak yanlış sınıflandırılması, bu iki intent arasında semantik benzerlik olabileceğine işaret ediyor.

⚖️ 2. Metriklerin Karşılaştırması

![image](https://github.com/user-attachments/assets/d331c7e7-c66c-43b2-b18b-d065407a39e5)

✅ 3. Sonuçların İstatistiksel Anlamı

%99.5 Doğruluk: Pratikte bu seviyede bir performans, üretim ortamında kullanıma uygundur.
Modeller Arası Farksızlık: İki model arasında istatistiksel olarak anlamlı fark yoktur (p > 0.05).
Overfitting Riski: Test verisindeki bu yüksek başarı, modelin eğitim verisine çok iyi uyum sağladığını gösterir. Gerçek dünya verisiyle test önerilir.







