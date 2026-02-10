import kagglehub
import keras
import tensorflow as tf
import numpy as np
import os

# Download model
path = kagglehub.model_download(
    "google/mobilenet-v2/tensorFlow2/035-128-classification"
)

print("Path to model files:", path)

# IMPORTANT: the real SavedModel is inside version folder
# Usually it is "2"
model_path = os.path.join(path, "2")

# Wrap SavedModel as a Keras layer
mobilenet_layer = keras.layers.TFSMLayer(
    model_path,
    call_endpoint="serving_default"
)

# Build inference model
model = keras.Sequential([mobilenet_layer])

print(model)

# Load image
img = tf.keras.utils.load_img(
    r"C:\Users\Omkar\Downloads\CNNs\flower1par.jpg",
    target_size=(128, 128)
)

x = tf.keras.utils.img_to_array(img)
x = x / 255.0
x = np.expand_dims(x, axis=0)

# Inference
pred = model(x)

print("Output shape:", pred.shape)
print("Top class index:", np.argmax(pred[0]))
print("Top probability:", np.max(pred[0]))
