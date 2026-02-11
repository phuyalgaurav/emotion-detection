import cv2
import numpy as np
from tensorflow.keras.models import load_model

# CONFIGURATION
# Path to your trained model
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


def run_realtime_detection():
    """Run emotion detection in realtime using webcam."""
    # Load the Model
    try:
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Could not load model. Check the path! Error: {e}")
        return

    # Load Face Detector (Haar Cascade)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # Open webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open webcam. Check your camera connection!")
        return

    print("✅ Webcam opened successfully.")
    print("📹 Starting realtime emotion detection...")
    print("Press 'q' or ESC to quit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Failed to grab frame.")
            break

        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = gray[y : y + h, x : x + w]

            # --- PRE-PROCESSING (Crucial Step) ---
            # 1. Resize to 48x48
            final_image = cv2.resize(face_roi, (48, 48))

            # 2. Normalize (0-1)
            final_image = final_image.astype("float32") / 255.0

            # 3. Reshape (Add batch and channel dimensions: 1, 48, 48, 1)
            final_image = np.expand_dims(final_image, axis=0)  # Batch dim
            final_image = np.expand_dims(final_image, axis=-1)  # Channel dim

            # --- PREDICTION ---
            # Use direct model call for faster inference in realtime
            prediction = model(final_image, training=False).numpy()
            max_index = np.argmax(prediction)
            predicted_emotion = EMOTIONS[max_index]
            confidence = prediction[0][max_index] * 100

            # --- VISUALIZATION ---
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Prepare text with emotion and confidence
            text = f"{predicted_emotion} ({confidence:.1f}%)"
            
            # Draw text background for better visibility
            (text_width, text_height), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame,
                (x, y - text_height - 10),
                (x + text_width, y),
                (0, 255, 0),
                -1,
            )
            
            # Draw emotion text
            cv2.putText(
                frame,
                text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                2,
            )

        # Display the resulting frame
        cv2.imshow("Realtime Emotion Detection - Press 'q' to quit", frame)

        # Break loop on 'q' key or ESC
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 27 is ESC key
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Realtime detection stopped.")


if __name__ == "__main__":
    run_realtime_detection()
