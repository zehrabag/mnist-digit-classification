import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt # Görselleştirme için ekledik

# 1. Veriyi Yükle
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. Modeli Kur
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 3. Eğitimi bir değişkene (history) atayalım ki grafiğini çizelim
print("Eğitim başlıyor...")
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# --- GÖRSELLEŞTİRME BÖLÜMÜ ---

# A. Doğruluk (Accuracy) Grafiği
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Test Doğruluğu')
plt.title('Model Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# B. Rastgele Tahminleri Gösterme
plt.subplot(1, 2, 2)
predictions = model.predict(x_test[:1]) # İlk test verisini tahmin et
plt.imshow(x_test[0], cmap='gray')
plt.title(f"Gerçek: {y_test[0]}, Tahmin: {predictions.argmax()}")
plt.axis('off')

plt.tight_layout()
plt.show() # Grafiği ekranda gösterir
