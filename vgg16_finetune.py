import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
import json

# parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_FEATURE = 3
EPOCHS_FINE = 3

# ------------------------------------------------
# 1. Load Stanford Dogs dataset (120 breeds)
# ------------------------------------------------
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

# save breed labels for inference later
with open("breed_labels.json", "w") as f:
    json.dump(breed_names, f)

# ------------------------------------------------
# 2. Preprocess & augment
# ------------------------------------------------
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

# ------------------------------------------------
# 3. Load pretrained VGG16 (freeze base)
# ------------------------------------------------
base_model = tf.keras.applications.VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

# ------------------------------------------------
# 4. Build model
# ------------------------------------------------
model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

# ------------------------------------------------
# 5. Phase 1 - train classifier head
# ------------------------------------------------
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nPhase 1: Training classifier head...")
model.fit(train_data, epochs=EPOCHS_FEATURE, validation_data=val_data)

# ------------------------------------------------
# 6. Phase 2 - fine-tune last 4 conv layers
# ------------------------------------------------
print("\nPhase 2: Fine-tuning last conv layers...")
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(train_data, epochs=EPOCHS_FINE, validation_data=val_data)

# ------------------------------------------------
# 7. Save model
# ------------------------------------------------
model.save("vgg16_dog_finetuned.h5")
print("\nModel saved to vgg16_dog_finetuned.h5")
print("Breed labels saved to breed_labels.json")
print("Training complete!")
