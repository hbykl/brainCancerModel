from data_loader import load_data
from model_builder import build_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import os

# Veriyi yükle (3-way split)
X_train, X_val, X_test, y_train, y_val, y_test, class_names = load_data()

# Modeli oluştur
model = build_model((128,128,1), len(class_names))

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=12, 
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, 
                      patience=5, min_lr=1e-6, verbose=1)
]

# Eğitim
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    shuffle=True,
    verbose=1
)

# Değerlendirme
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ Test Doğruluğu: {test_acc*100:.2f}%")
print(f"✅ Test Kaybı: {test_loss:.4f}")

# MODELİ HEMEN KAYDET - ÖNCELİKLİ İŞLEM
timestamp = time.strftime("%Y%m%d-%H%M%S")
model_name = f"brain_tumor_model_{test_acc*100:.1f}acc_{timestamp}.keras"
model.save(model_name)
print(f"\n🔥 MODEL KESİNLİKLE KAYDEDİLDİ: {model_name}")
print(f"📂 Dosya boyutu: {os.path.getsize(model_name)/1024/1024:.2f} MB")

# Alternatif olarak .h5 formatında da kaydedelim
h5_model_name = f"brain_tumor_model_{test_acc*100:.1f}acc_{timestamp}.h5"
model.save(h5_model_name)
print(f"🔁 Alternatif H5 formatında kaydedildi: {h5_model_name}")

# Grafik oluşturma
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim')
plt.plot(history.history['val_accuracy'], label='Doğrulama')
plt.title(f'Model Doğruluğu ({test_acc*100:.1f}%)')
plt.ylabel('Doğruluk')
plt.xlabel('Epok')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim')
plt.plot(history.history['val_loss'], label='Doğrulama')
plt.title('Model Kaybı')
plt.ylabel('Kayıp')
plt.xlabel('Epok')
plt.legend()
plt.tight_layout()

# Grafiği dosyaya kaydet (göstermeden)
plot_filename = f"training_plot_{test_acc*100:.1f}acc_{timestamp}.png"
plt.savefig(plot_filename, dpi=150)
print(f"\n📊 Grafik kaydedildi: {plot_filename}")

# İsteğe bağlı: Grafikleri göster (yorum satırından çıkarabilirsiniz)
# print("\n⚠️ Grafik penceresi açıldı. Kapatmak için lütfen pencereyi kapatın...")
# plt.show()

print("\n✨ Tüm işlemler başarıyla tamamlandı! ✨")