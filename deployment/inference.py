# deployment/inference.py
import onnxruntime as ort
import numpy as np

class InferenceEngine:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path)
    
    def predict(self, inputs):
        arr = np.array(inputs, dtype=np.float32)
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: arr.reshape(1, -1)})
        return result

if __name__ == "__main__":
    engine = InferenceEngine("models/fuel_model.onnx")
    print(engine.predict([0.5, 1.2, 3.4]))
