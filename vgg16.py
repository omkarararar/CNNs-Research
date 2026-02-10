import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# ------------------------------------------------
# 1. Load pre-trained VGG16 (ImageNet weights)
# ------------------------------------------------
model = VGG16(weights="imagenet", include_top=True)

# ------------------------------------------------
# 2. Load your image
# ------------------------------------------------
img_path = r"C:\Users\Omkar\Downloads\CNNs\dog.jpg"   # put your image here

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# ------------------------------------------------
# 3. Preprocess exactly as VGG16 expects
# ------------------------------------------------
x = preprocess_input(x)

# ------------------------------------------------
# 4. Predict
# ------------------------------------------------
preds = model.predict(x)

# ------------------------------------------------
# 5. Decode ImageNet predictions
# ------------------------------------------------
results = decode_predictions(preds, top=5)[0]

for cls_id, name, score in results:
    print(f"{name:20s} : {score:.4f}")
