import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# CONFIGURATION
# 1. Put the path to your photo here
IMAGE_PATH = "image.png"

# 2. Path to your trained model
MODEL_PATH = "best_emotion_model.keras"

# Emotion labels (Must match your training order)
EMOTIONS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def test_on_photo(img_path):
    # Load the Model
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except:
        print("❌ Could not load model. Check the path!")
        return

    # Load the Image
    original_img = cv2.imread(img_path)
    if original_img is None:
        print("❌ Could not open image. Check the path!")
        return

    # Convert to Grayscale (Model expects gray)
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)

    # Load Face Detector (Haar Cascade)
    # This comes built-in with OpenCV
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    if len(faces) == 0:
        print(
            "⚠️ No face detected! Attempting to run on the whole image (result might be inaccurate)."
        )
        # If no face found, just resize the whole image
        face_roi = gray_img
        x, y, w, h = 0, 0, original_img.shape[1], original_img.shape[0]
        detected = False
    else:
        print(f"✅ Found {len(faces)} face(s). Using the largest one.")
        # Pick the largest face if multiple are found
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        (x, y, w, h) = faces[0]
        face_roi = gray_img[y : y + h, x : x + w]
        detected = True

    # --- PRE-PROCESSING (Crucial Step) ---
    # 1. Resize to 48x48
    final_image = cv2.resize(face_roi, (48, 48))

    # 2. Normalize (0-1)
    final_image = final_image.astype("float32") / 255.0

    # 3. Reshape (Add batch and channel dimensions: 1, 48, 48, 1)
    final_image = np.expand_dims(final_image, axis=0)  # Batch dim
    final_image = np.expand_dims(final_image, axis=-1)  # Channel dim

    # --- PREDICTION ---
    prediction = model.predict(final_image)
    max_index = np.argmax(prediction)
    predicted_emotion = EMOTIONS[max_index]
    confidence = prediction[0][max_index] * 100

    # --- VISUALIZATION ---
    # Draw a box around the face on the original color image
    if detected:
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert BGR to RGB for Matplotlib
    original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 5))
    plt.imshow(original_img_rgb)
    plt.title(
        f"Prediction: {predicted_emotion} ({confidence:.1f}%)",
        color="green",
        fontsize=15,
    )
    plt.axis("off")
    plt.show()

    print(f"\n🧠 Result: {predicted_emotion}")
    print("📊 Confidence Scores:")
    for i, score in enumerate(prediction[0]):
        print(f"  {EMOTIONS[i]}: {score*100:.1f}%")


# Run the function
test_on_photo(IMAGE_PATH)
