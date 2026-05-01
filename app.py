from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/gesture')
def get_gesture():
    return jsonify({
"gesture": "A"
})

if __name__ == "__main__":
    app.run(port=5001, debug=True)
