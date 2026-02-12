import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load model once
model = MobileNetV2(weights="imagenet", include_top=True)
img_path = r"C:\Users\Omkar\Downloads\CNNs\dog.jpg" # put your image here

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    decoded = decode_predictions(preds, top=5)[0]

    print(f"\nPredictions for: {img_path}")
    for _, name, prob in decoded:
        print(f"{name:20s} : {prob:.4f}")


# ---- run on any image you want ----

predict_image(r"C:\Users\Omkar\Downloads\CNNs\dog.jpg")

