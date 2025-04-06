from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load model once at startup
model = load_model("wand_model.h5")

@app.route("/", methods=["GET"])
def home():
    return "Wand Gesture API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json.get("data")
        input_array = np.array(data)
        prediction = model.predict(input_array)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)