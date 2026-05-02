# predict.py

import cv2
import mediapipe as mp
import pickle
import time

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def get_prediction():
    cap = None
    try:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            return "No gesture"

        # Warm-up camera
        for _ in range(8):
            ret, _ = cap.read()
            if not ret:
                return "No gesture"
            time.sleep(0.03)

        # Capture one frame
        ret, frame = cap.read()
        if not ret:
            return "No gesture"

        # Process frame
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(image)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                data = []
                for lm in hand_landmarks.landmark:
                    data.append(lm.x)
                    data.append(lm.y)

                prediction = model.predict([data])[0]
                return str(prediction)

        return "No gesture"

    except Exception:
        return "No gesture"

    finally:
        if cap is not None:
            cap.release()