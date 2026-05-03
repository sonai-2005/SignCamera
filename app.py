from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import mediapipe as mp
import pickle
from google import genai
import time

# =========================
# CONFIG
# =========================

app = Flask(__name__)
CORS(app)

# 🔑 PUT YOUR NEW API KEY HERE
client = genai.Client(api_key="AIzaSyCVX-WiWf7b2dqpshf-9O8Gn9M5kggIfHY")

# Rate limit protection
last_call_time = 0

# =========================
# LOAD MODEL
# =========================

model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# =========================
# ROUTE 1: PREDICT (CAMERA)
# =========================

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if 'image' not in data:
        return jsonify({"error": "No image provided"})

    try:
        # Decode base64 image
        image_data = base64.b64decode(data['image'].split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            data_aux = []

            for lm in landmarks.landmark:
                data_aux.append(lm.x)
                data_aux.append(lm.y)

            prediction = model.predict([data_aux])[0]

            return jsonify({"gesture": str(prediction)})

        return jsonify({"gesture": "No gesture"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# ROUTE 2: GEMINI CORRECTION
# =========================

@app.route('/correct', methods=['POST'])
def correct_text():
    global last_call_time

    # Prevent rapid requests
    if time.time() - last_call_time < 5:
        return jsonify({"corrected": "Please wait a few seconds..."})

    last_call_time = time.time()

    data = request.json
    text = data.get("text", "")

    if not text.strip():
        return jsonify({"corrected": ""})

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=f"Correct this sentence properly: {text}"
        )

        corrected = response.text

        return jsonify({"corrected": corrected})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================

if __name__ == '__main__':
    app.run(port=5001, debug=True)