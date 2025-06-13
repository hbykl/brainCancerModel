import os
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

def load_data(data_path="data", img_size=(128, 128)):
    classes = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    X, y = [], []
    
    for idx, label in enumerate(classes):
        folder = os.path.join(data_path, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                arr = img_to_array(img) / 255.0
                X.append(arr)
                y.append(idx)
            except Exception as e:
                print(f"Hata: {img_path} → {e}")

    X = np.array(X, dtype="float32")
    y = to_categorical(np.array(y), num_classes=len(classes))

    # 3-way split: train/val/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=np.argmax(y, axis=1), random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=np.argmax(y_train, axis=1), random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, classes