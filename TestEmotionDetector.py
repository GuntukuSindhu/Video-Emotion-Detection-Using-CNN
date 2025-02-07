import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define constants
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FACE_DETECTION_MODEL = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
EMOTION_MODEL_JSON = 'C:\\Users\\sindh\\OneDrive\\Desktop\\mini project\\model\\emotion_model.json'
EMOTION_MODEL_WEIGHTS = 'C:\\Users\\sindh\\OneDrive\\Desktop\\mini project\\model\\emotion_model.h5'
EMOTION_DICT = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def load_emotion_model():
    """Load the emotion detection model from disk."""
    try:
        with open(EMOTION_MODEL_JSON, 'r') as json_file:
            loaded_model_json = json_file.read()
        emotion_model = model_from_json(loaded_model_json)
        emotion_model.load_weights(EMOTION_MODEL_WEIGHTS)
        print("Loaded model from disk")
        return emotion_model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def capture_video():
    """Capture video from the webcam."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error opening webcam")
        return None
    return cap

def detect_faces(frame):
    """Detect faces in the given frame."""
    face_detector = cv2.CascadeClassifier(FACE_DETECTION_MODEL)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    return num_faces

def predict_emotion(emotion_model, face):
    """Predict the emotion of the given face."""
    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)
    emotion_prediction = emotion_model.predict(cropped_img)
    maxindex = int(np.argmax(emotion_prediction))
    return EMOTION_DICT[maxindex]

def main():
    emotion_model = load_emotion_model()
    if emotion_model is None:
        return

    cap = capture_video()
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        num_faces = detect_faces(frame)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            roi_gray_frame = roi_gray_frame[y:y + h, x:x + w]
            emotion = predict_emotion(emotion_model, roi_gray_frame)
            cv2.putText(frame, emotion, (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
