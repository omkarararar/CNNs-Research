# ============================================================
# VGG16 Dog Breed Classification - Google Colab (T4 GPU)
# ============================================================
# Copy-paste this entire file into a single Colab cell,
# or split at the comment markers into separate cells.
# After training, download vgg16_dog_finetuned.h5 and
# breed_labels.json to your local project.
# ============================================================

# ---- CELL 1: Install & Imports ----
!pip install tensorflow-datasets -q

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import numpy as np
import json
import matplotlib.pyplot as plt

print("GPU available:", tf.config.list_physical_devices('GPU'))

# ---- CELL 2: Parameters ----
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FEATURE = 10
EPOCHS_FINE = 5

# ---- CELL 3: Load Stanford Dogs dataset (120 breeds) ----
print("Loading Stanford Dogs dataset...")
(train_ds, val_ds), info = tfds.load(
    "stanford_dogs",
    split=["train", "test"],
    as_supervised=True,
    with_info=True
)

num_classes = info.features["label"].num_classes
breed_names = info.features["label"].names
print(f"Found {num_classes} breeds, {info.splits['train'].num_examples} train / {info.splits['test'].num_examples} test images")

with open("breed_labels.json", "w") as f:
    json.dump(breed_names, f)

# ---- CELL 4: Preprocess & Augment ----
def preprocess(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = tf.keras.applications.vgg16.preprocess_input(image)
    label = tf.one_hot(label, num_classes)
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    return image, label

train_data = (train_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_data = (val_ds
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ---- CELL 5: Build Model ----
base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# ---- CELL 6: Phase 1 - Train classifier head ----
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nPhase 1: Training classifier head...")
history1 = model.fit(train_data, epochs=EPOCHS_FEATURE, validation_data=val_data)

# ---- CELL 7: Phase 2 - Fine-tune last 4 conv layers ----
print("\nPhase 2: Fine-tuning last conv layers...")
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(train_data, epochs=EPOCHS_FINE, validation_data=val_data)

# ---- CELL 8: Save Model ----
model.save("vgg16_dog_finetuned.h5")
print("Model saved!")

# ---- CELL 9: Plot Training History ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

all_acc = history1.history['accuracy'] + history2.history['accuracy']
all_val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
all_loss = history1.history['loss'] + history2.history['loss']
all_val_loss = history1.history['val_loss'] + history2.history['val_loss']

ax1.plot(all_acc, label='Train Accuracy')
ax1.plot(all_val_acc, label='Val Accuracy')
ax1.axvline(x=EPOCHS_FEATURE-1, color='r', linestyle='--', label='Fine-tune start')
ax1.set_title('Accuracy')
ax1.legend()

ax2.plot(all_loss, label='Train Loss')
ax2.plot(all_val_loss, label='Val Loss')
ax2.axvline(x=EPOCHS_FEATURE-1, color='r', linestyle='--', label='Fine-tune start')
ax2.set_title('Loss')
ax2.legend()

plt.tight_layout()
plt.show()

# ---- CELL 10: Upload & test your own dog image ----
from google.colab import files
from tensorflow.keras.preprocessing import image

uploaded = files.upload()  # click "Choose Files" and pick a dog image
img_path = list(uploaded.keys())[0]

img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = tf.keras.applications.vgg16.preprocess_input(x)

preds = model.predict(x)[0]
top5_idx = np.argsort(preds)[-5:][::-1]

print(f"\nPredictions for: {img_path}")
print("-" * 40)
for i, idx in enumerate(top5_idx):
    breed = breed_names[idx].replace("_", " ").title()
    print(f"  {i+1}. {breed:30s} : {preds[idx]*100:.2f}%")

plt.imshow(image.load_img(img_path))
pred_breed = breed_names[top5_idx[0]].replace("_", " ").title()
plt.title(f"Predicted: {pred_breed} ({preds[top5_idx[0]]*100:.1f}%)")
plt.axis("off")
plt.show()

# ---- CELL 11: Download model files to your local machine ----
files.download("vgg16_dog_finetuned.h5")
files.download("breed_labels.json")
