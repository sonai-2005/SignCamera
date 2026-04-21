import cv2
import mediapipe as mp
import pickle
import time
import csv
import os
from datetime import datetime
import win32com.client

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Windows voice
speaker = win32com.client.Dispatch("SAPI.SpVoice")

# Create CSV file if not exists
if not os.path.exists("history.csv"):
    with open("history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "raw_word", "corrected_word", "status"])

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Variables
word = ""
last_prediction = ""
last_add_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image)

    prediction = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Collect points
            data = []
            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

            # Predict
            prediction = model.predict([data])[0]

            # Show detected symbol
            cv2.putText(
                frame,
                f"Detected: {prediction}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Auto add letter if stable for 1 sec
    if prediction != "" and prediction == last_prediction:
        if time.time() - last_add_time > 1:
            word += str(prediction)
            last_add_time = time.time()

    last_prediction = prediction

    # Show current word
    cv2.putText(
        frame,
        f"Word: {word}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    # Controls
    cv2.putText(
        frame,
        "C=Speak+Save  D=Delete  R=Reset  ESC=Exit",
        (30, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.imshow("Sign Language System", frame)

    key = cv2.waitKey(1) & 0xFF

    # Confirm word
    if key == ord('c'):
        if word != "":

            # Speak
            speaker.Speak(word)

            # Save in future-ready CSV
            with open("history.csv", "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    word,
                    "",
                    "pending"
                ])

            # Reset word
            word = ""

    # Delete last letter
    if key == ord('d'):
        word = word[:-1]

    # Reset full word
    if key == ord('r'):
        word = ""

    # Exit
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()