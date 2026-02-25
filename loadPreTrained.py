import numpy as np
import tensorflow as tf
import json
import h5py
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models

# architecture and weights according to the trained model and vgg16
base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(120, activation="softmax")
])

# Build model so weights are initialized
model.build((None, 224, 224, 3))

# Load fine-tuned weights from .h5
with h5py.File("vgg16_dog_finetuned.h5", "r") as f:
    wg = f["model_weights"]

    # Load VGG16 conv layer weights (kernel first, then bias)
    for conv_layer in base_model.layers:
        if conv_layer.name in wg["vgg16"]:
            lg = wg["vgg16"][conv_layer.name]
            weights = []
            if "kernel" in lg:
                weights.append(np.array(lg["kernel"]))
            if "bias" in lg:
                weights.append(np.array(lg["bias"]))
            if weights:
                conv_layer.set_weights(weights)

    # Load head layers (dense, dense_1)
    # Weights are stored at: layer_name/sequential/layer_name/kernel|bias
    for head_layer in model.layers[1:]:  # skip base_model
        if head_layer.name in wg:
            lg = wg[head_layer.name]
            # Navigate through the nested group structure
            if "sequential" in lg:
                lg = lg["sequential"]
            if head_layer.name in lg:
                lg = lg[head_layer.name]
            weights = []
            if "kernel" in lg:
                weights.append(np.array(lg["kernel"]))
            if "bias" in lg:
                weights.append(np.array(lg["bias"]))
            if weights:
                head_layer.set_weights(weights)

print("Model loaded successfully!")

# breed labels -> around 120 with 20580 images in total -> around 160-150 per image
with open("breed_labels.json", "r") as f:
    breed_names = json.load(f)

print(f"Can classify {len(breed_names)} dog breeds.")

# preprocessing the image to this target size of the imagenet dataset
img_path = r"C:\Users\Omkar\Downloads\CNNs\dog2.jpg"

img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# predict the breed of the current dog
preds = model.predict(x)[0]
top5_idx = np.argsort(preds)[-5:][::-1]

print(f"\nPredictions for: {img_path}")
print("-" * 40)
for i, idx in enumerate(top5_idx):
    breed = breed_names[idx].replace("_", " ").title()
    confidence = preds[idx] * 100
    print(f"  {i+1}. {breed:30s} : {confidence:.2f}%")

#display the top 5 choices for the breeds
plt.imshow(image.load_img(img_path))
top_breed = breed_names[top5_idx[0]].replace("_", " ").title()
plt.title(f"Predicted: {top_breed} ({preds[top5_idx[0]]*100:.1f}%)")
plt.axis("off")
plt.show()