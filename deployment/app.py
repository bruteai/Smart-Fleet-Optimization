# deployment/app.py
from flask import Flask, request, jsonify
from deployment.inference import InferenceEngine

app = Flask(__name__)

# Initialize inference engines for each model
fuel_engine = InferenceEngine("models/fuel_model.onnx")
delay_engine = InferenceEngine("models/delay_model.onnx")
maintenance_engine = InferenceEngine("models/maintenance_model.onnx")

@app.route('/predict', methods=['POST'])
def predict():
    req = request.get_json()
    model_type = req.get("model_type")
    inputs = req.get("input_data")
    
    if model_type == "fuel":
        pred = fuel_engine.predict(inputs)
    elif model_type == "delay":
        pred = delay_engine.predict(inputs)
    elif model_type == "maintenance":
        pred = maintenance_engine.predict(inputs)
    else:
        return jsonify({"error": "Invalid model_type"}), 400

    return jsonify({"prediction": pred[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
