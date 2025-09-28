E-Motor Assembly Quality Control using Machine Learning and Image Classification

This repository contains a Machine Learning project for **inspection of EMotor parts** using image classification. The model classifies images into categories such as **Complete**, **Missing Cover**, and **Missing Screw**.

---

## 📂 Dataset

- Images are organized in folders for different views (e.g., top, side).  
- Image identifiers:
  - `_C_` → Complete  
  - `_MC_` → Missing Cover  
  - `_MS_` → Missing Screw  

---

## ⚙️ Requirements
- Python 3.x  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas, Matplotlib, scikit-learn  

Install dependencies with:

```bash
pip install -r requirements.txt

---

## 🖥️ Usage
Load Data
from scripts import load_features_labels

features, labels = load_features_labels("./data/top", size=(128,128), color=True, flatten=True)

Train Model
model.fit(datagen.flow(X_train, y_train, batch_size=8),
          validation_data=(X_validation, y_validation),
          epochs=50)

Predict New Images
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img = load_img("new_image.jpg", target_size=(512,512))
img_array = img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
predicted_class = np.argmax(prediction, axis=1)[0]

##📊 Results

Training and validation accuracy/loss are tracked during model training.

Confusion matrices and classification reports are available.

##⚖️ License

This project is licensed under the MIT License.

##📧 Contact

For questions or suggestions, contact Your Name at alireza.mirbagheri@gmail.com


---
