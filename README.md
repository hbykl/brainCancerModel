# Brain Tumor Classification Projesi

Bu proje, beyin tümörü MRI görüntülerinden üç sınıflı (`brain_glioma`, `brain_menin`, `brain_tumor`) bir sınıflandırma modeli eğitmek için TensorFlow/Keras tabanlı basit bir CNN akışı sağlar.

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)

## İçindekiler
- [Özellikler](#özellikler)
- [Teknolojiler ve Gereksinimler](#teknolojiler-ve-gereksinimler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Dizin Yapısı](#dizin-yapısı)
- [Model Mimarisi](#model-mimarisi)
- [Konfigürasyon](#konfigürasyon)
- [Testler](#testler)
- [Lisans](#lisans)

## Özellikler
- Üç sınıflı beyin tümörü sınıflandırması: `brain_glioma`, `brain_menin`, `brain_tumor`
- TensorFlow/Keras tabanlı basit bir CNN mimarisi
- Veri yükleme ve train/validation/test bölme script’i
- Eğitim sırasında eğitim/doğrulama kayıpları ve doğruluk grafiklerinin oluşturulması
- Kaydedilmiş model dosyalarının tekrar değerlendirilmesi için test script’i

## Teknolojiler ve Gereksinimler
- Python 3.8+
- TensorFlow
- Keras
- NumPy
- scikit-learn
- Matplotlib

```bash
pip install -r requirements.txt
```

## Kurulum

```bash
git clone https://github.com/hbykl/brainCancerModel.git
cd brainCancerModel

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Kullanım

### Veri Hazırlama

```bash
python data_loader.py
```

### Modeli Eğitme

```bash
python main.py
```

- Model: `brain_tumor_model_<acc>_<tarih>.keras` ve `.h5`
- Grafik: `training_plot_<acc>_<tarih>.png`

### Modeli Test Etme

```bash
python test.py --model_path path/to/model.keras
```

- Test çıktısı olarak doğruluk ve tahminler konsola yazdırılır.

## Dizin Yapısı

```text
brainCancerModel/
├── data/
│   ├── brain_glioma/
│   ├── brain_menin/
│   └── brain_tumor/
├── data_loader.py
├── model_builder.py
├── main.py
├── test.py
├── training_plot_*.png
├── brain_tumor_model_*.keras
├── brain_tumor_model_*.h5
├── requirements.txt
└── README.md
```

## Model Mimarisi

- **Giriş Katmanı**: `Input(shape=(128, 128, 1))`
- **Evrişim Katmanları**:
  - `Conv2D(32)` → `MaxPooling2D`
  - `Conv2D(64)` → `MaxPooling2D`
  - `Conv2D(128)` → `MaxPooling2D`
- **Dropout**: Her blok sonrası `Dropout(0.25)`
- **Fully Connected**:
  - `Flatten()`
  - `Dense(128, activation='relu')` → `Dropout(0.5)`
  - `Dense(3, activation='softmax')`
- **Derleyici**:
  - `Adam(learning_rate=0.001)`
  - `categorical_crossentropy`
  - `accuracy`

## Konfigürasyon

```python
import random, numpy as np, tensorflow as tf
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
```

## Testler

Henüz birim test tanımlı değil. `pytest` veya `unittest` ile test yapısı eklenebilir.

## Lisans

MIT Lisansı ile lisanslanmıştır. Ayrıntılar için `LICENSE` dosyasına bakınız.
