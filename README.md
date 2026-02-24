# MNIST El Yazısı Rakam Tanıma (Handwritten Digit Recognition)

Bu proje, derin öğrenmenin "Merhaba Dünya"sı olarak kabul edilen **MNIST** veri setini kullanarak, el yazısıyla yazılmış rakamları (0-9) tanıyan bir Yapay Sinir Ağı (ANN) modelidir.

## Proje Özeti
Bu çalışmada, 28x28 piksel boyutundaki gri tonlamalı rakam resimlerini analiz eden bir model oluşturulmuştur. Model, resimdeki pikselleri girdi olarak alır ve bu resmin hangi rakama ait olduğunu yüksek doğruluk oranıyla tahmin eder.

## Kullanılan Teknolojiler
* **Python 3.11**
* **TensorFlow & Keras:** Derin öğrenme modelinin oluşturulması ve eğitilmesi.
* **Matplotlib:** Eğitim sürecinin ve tahminlerin görselleştirilmesi.
* **NumPy:** Veri manipülasyonu.

## Model Mimarisi
Modelim şu katmanlardan oluşmaktadır:
1. **Flatten:** 28x28'lik kare matrisi 784 elemanlı tek bir vektöre dönüştürür.
2. **Dense (128 units, ReLU):** Örüntüleri öğrenen ana gizli katman.
3. **Dropout (0.2):** Aşırı öğrenmeyi (overfitting) engellemek için nöronların %20'sini rastgele devre dışı bırakır.
4. **Dense (10 units, Softmax):** 10 farklı rakam sınıfı için olasılık dağılımı üretir.

## Sonuçlar
Model, test verisi üzerinde yaklaşık **%98** doğruluk (accuracy) oranına ulaşmıştır. 

