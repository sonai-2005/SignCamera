from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import mediapipe as mp
import pickle

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("model.pkl", "rb"))

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if 'image' not in data:
        return jsonify({"error": "No image provided"})

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

if __name__ == '__main__':
    app.run(port=5001, debug=True)