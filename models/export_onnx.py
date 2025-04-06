# models/export_onnx.py
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

def export_to_onnx(model_file, output_file, sample_input):
    """Convert a saved scikit-learn model to ONNX format."""
    model = joblib.load(model_file)
    initial_type = [('float_input', FloatTensorType([None, len(sample_input)]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open(output_file, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Model exported to {output_file}")

if __name__ == "__main__":
    sample_fuel = [0.0, 0.0, 0.0]
    sample_delay = [0.0, 0.0]
    sample_maint = [0.0, 0.0, 0.0]
    export_to_onnx("models/fuel_model.pkl", "models/fuel_model.onnx", sample_fuel)
    export_to_onnx("models/delay_model.pkl", "models/delay_model.onnx", sample_delay)
    export_to_onnx("models/maintenance_model.pkl", "models/maintenance_model.onnx", sample_maint)
