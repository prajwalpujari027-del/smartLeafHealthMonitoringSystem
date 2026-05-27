import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

print("GPU Available:", tf.config.list_physical_devices('GPU'))

# -----------------------------
# DATASET PATH (ONLY TRAIN FOLDER)
# -----------------------------
dataset_path = "/content/train/train"

IMG_SIZE = (150, 150)
BATCH_SIZE = 32

# -----------------------------
# IMAGE PREPROCESSING (FIXED)
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_path,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -----------------------------
# CLASS NAMES (NOW INCLUDES RANDOM CLASS)
# -----------------------------
class_names = list(train_data.class_indices.keys())
print("Classes:", class_names)
print(f"Total Classes: {len(class_names)}")

# Check if Tomato__Random is included
if "Tomato__Random" in class_names:
    print("✅ Tomato__Random class detected!")
else:
    print("⚠️  Warning: Tomato__Random class not found. Check folder structure.")

# -----------------------------
# MODEL
# -----------------------------
model = models.Sequential([
    tf.keras.Input(shape=(150,150,3)),

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n📊 Model Summary:")
model.summary()

# -----------------------------
# CALLBACKS
# -----------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

# -----------------------------
# TRAIN
# -----------------------------
print("\n🚀 Starting Training...")
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)

# Print training results
print("\n✅ Training Complete!")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")

# -----------------------------
# SAVE MODEL
# -----------------------------
model.save("/content/drive/MyDrive/plant_disease_model.keras")
print("✅ Model saved to /content/drive/MyDrive/plant_disease_model.keras")

with open("/content/drive/MyDrive/class_names.json", "w") as f:
    json.dump(class_names, f, indent=2)
print("✅ Class names saved to /content/drive/MyDrive/class_names.json")

print("\n🎉 Model training and saving complete!")
