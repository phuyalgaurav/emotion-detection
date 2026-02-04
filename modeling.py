import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

# --- CONFIGURATION ---
# IMPORTANT: Make sure this points to your actual dataset folder
DATASET_PATH = "dataset"
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 2


# --- PART 1: DATA LOADING (PHASE 2) ---
def load_data_from_folders(base_path):
    X_data = []
    y_data = []
    LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

    subsets = ["train", "test"]
    print("Loading images from folders...")

    for subset in subsets:
        subset_path = os.path.join(base_path, subset)
        for label_index, emotion in enumerate(LABELS):
            emotion_path = os.path.join(subset_path, emotion)
            if not os.path.exists(emotion_path):
                continue

            files = os.listdir(emotion_path)
            for img_name in tqdm(files, desc=f"{subset}/{emotion}", leave=False):
                try:
                    img_path = os.path.join(emotion_path, img_name)
                    img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is not None:
                        img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                        X_data.append(img_resized)
                        y_data.append(label_index)
                except Exception:
                    pass

    return np.array(X_data), np.array(y_data)


# Execute Data Loading
if not os.path.exists(DATASET_PATH):
    print(
        f"❌ ERROR: Dataset not found at '{DATASET_PATH}'. Please check the folder name."
    )
    exit()

X, y_int = load_data_from_folders(DATASET_PATH)

# Preprocessing
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1).astype("float32") / 255.0
y = to_categorical(y_int, num_classes=7)

# Splitting (70:15:15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y_int
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(
    f"\nData Ready: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}"
)


# --- PART 2: MODEL DEFINITION (PHASE 3) ---
def build_model(input_shape=(48, 48, 1), num_classes=7):
    model = Sequential(
        [
            # Block 1
            Conv2D(
                64, (3, 3), activation="relu", padding="same", input_shape=input_shape
            ),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Block 2
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Block 3
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            # Fully Connected
            Flatten(),
            Dense(256, activation="relu"),
            BatchNormalization(),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    opt = Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


model = build_model()

# --- PART 3: TRAINING ---
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

callbacks = [
    ModelCheckpoint(
        "best_emotion_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001, verbose=1
    ),
]

print("\nStarting Training on M4 Mac (Check usage in Activity Monitor!)...")
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
)

# --- PART 4: PLOTTING ---
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.title("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.legend()
plt.show()

print("✅ Training finished successfully!")
