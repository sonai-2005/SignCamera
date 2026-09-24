from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import mediapipe as mp
import pickle
from google import genai
import time
import os

# =========================
# CONFIG
# =========================

from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# =========================
# GEMINI
# =========================

client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)

# Rate limit protection
last_call_time = 0

# =========================
# LOAD MODEL
# =========================

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# =========================
# MEDIAPIPE
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1
)

# =========================
# ROUTE 1: PREDICT
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    if not data or "image" not in data:
        return jsonify({
            "error": "No image provided"
        }), 400

    try:

        # =========================
        # 1. DECODE BASE64 IMAGE
        # =========================

        image_string = data["image"]

        # Supports:
        # data:image/jpeg;base64,XXXX
        # OR
        # XXXX

        if "," in image_string:
            image_string = image_string.split(",", 1)[1]

        image_data = base64.b64decode(image_string)

        np_arr = np.frombuffer(
            image_data,
            np.uint8
        )

        frame = cv2.imdecode(
            np_arr,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "error": "Could not decode image"
            }), 400

        print(
            "Image decoded:",
            frame.shape
        )

        # =========================
        # 2. MEDIAPIPE
        # =========================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        print(
            "Hand detected:",
            bool(results.multi_hand_landmarks)
        )

        # =========================
        # 3. EXTRACT LANDMARKS
        # =========================

        if results.multi_hand_landmarks:

            landmarks = results.multi_hand_landmarks[0]

            data_aux = []

            for lm in landmarks.landmark:

                data_aux.append(lm.x)
                data_aux.append(lm.y)

            print(
                "Number of features:",
                len(data_aux)
            )

            # =========================
            # 4. MODEL PREDICTION
            # =========================

            prediction = model.predict(
                [data_aux]
            )[0]

            print(
                "Prediction:",
                prediction
            )

            return jsonify({
                "gesture": str(prediction)
            })

        return jsonify({
            "gesture": "No gesture"
        })

    except Exception as e:

        import traceback

        print("\n")
        print("========== PREDICT ERROR ==========")

        traceback.print_exc()

        print("====================================")
        print("\n")

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# ROUTE 2: GEMINI CORRECTION
# =========================

@app.route("/correct", methods=["POST"])
def correct_text():

    global last_call_time

    # =========================
    # RATE LIMIT
    # =========================

    if time.time() - last_call_time < 5:

        return jsonify({
            "corrected": "Please wait a few seconds..."
        })

    last_call_time = time.time()

    # =========================
    # GET TEXT
    # =========================

    data = request.json

    text = data.get(
        "text",
        ""
    )

    if not text.strip():

        return jsonify({
            "corrected": ""
        })

    try:

        # =========================
        # GEMINI REQUEST
        # =========================

        response = client.models.generate_content(

            model="gemini-2.0-flash",

            contents=f"Correct this sentence properly: {text}"

        )

        corrected = response.text

        return jsonify({
            "corrected": corrected
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# =========================
# RUN SERVER
# =========================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5001
            )
        )
    )

