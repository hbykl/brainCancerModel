import os
import time
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_test_data(data_path="data", img_size=(128, 128)):
    classes = sorted(
        [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    )
    X, y = [], []

    for idx, label in enumerate(classes):
        folder = os.path.join(data_path, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            try:
                # Gri-ton okuma
                img = load_img(img_path, target_size=img_size, color_mode="grayscale")
                arr = img_to_array(img) / 255.0          # şekil: (128,128,1)
                X.append(arr)
                y.append(idx)
            except Exception as e:
                print(f"Hata: {img_path} → {e}")

    X = np.array(X, dtype="float32")
    y = to_categorical(np.array(y), num_classes=len(classes))

   
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=np.argmax(y, axis=1), random_state=42
    )
    return X_test, y_test, classes

X_test, y_test, class_names = load_test_data(data_path="data", img_size=(128, 128))

model_path = "brain_tumor_model_94.7acc_20250613-144738.keras"
model = load_model(model_path)

REQUESTED_SAMPLES = 100
num_samples = min(REQUESTED_SAMPLES, len(X_test))

predictions = model.predict(X_test[:num_samples], verbose=0)
correct_count = 0

for i, pred in enumerate(predictions):
    predicted_class = class_names[np.argmax(pred)]
    true_class      = class_names[np.argmax(y_test[i])]
    is_correct      = predicted_class == true_class
    if is_correct:
        correct_count += 1
    print(
        f"Sample {i+1:03d}: Predicted = {predicted_class:8s} | "
        f"Actual = {true_class:8s} | {'Correct' if is_correct else 'Wrong'}"
    )
    time.sleep(0.1)   # 0.1 s gecikme
    
accuracy = (correct_count / num_samples) * 100
print(f"\nToplam {num_samples} örnek üzerindeki doğruluk: {accuracy:5.2f}%")
