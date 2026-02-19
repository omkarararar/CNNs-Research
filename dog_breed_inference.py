import numpy as np
import tensorflow as tf
import json
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image

# ------------------------------------------------
# 1. Load trained model & breed labels
# ------------------------------------------------
model = tf.keras.models.load_model("vgg16_dog_finetuned.h5")

with open("breed_labels.json", "r") as f:
    breed_names = json.load(f)

# ------------------------------------------------
# 2. Load & preprocess image
# ------------------------------------------------
img_path = r"C:\Users\Omkar\Downloads\CNNs\dog.jpg"   # put your dog image here

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# ------------------------------------------------
# 3. Predict breed
# ------------------------------------------------
preds = model.predict(x)[0]
top5_idx = np.argsort(preds)[-5:][::-1]

print(f"\nPredictions for: {img_path}")
print("-" * 40)
for i, idx in enumerate(top5_idx):
    breed = breed_names[idx].replace("_", " ").title()
    confidence = preds[idx] * 100
    print(f"  {i+1}. {breed:30s} : {confidence:.2f}%")

# ------------------------------------------------
# 4. Display image with prediction
# ------------------------------------------------
plt.imshow(image.load_img(img_path))
top_breed = breed_names[top5_idx[0]].replace("_", " ").title()
plt.title(f"Predicted: {top_breed} ({preds[top5_idx[0]]*100:.1f}%)")
plt.axis("off")
plt.show()
