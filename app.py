"""
Flask Web Application - Grape Disease Prediction System
=======================================================
This Flask app provides a web interface for:
1. Inputting environmental and crop data
2. Predicting downy mildew disease risk
3. Getting actionable recommendations for farmers

Routes:
    GET  /  - Display the prediction form
    POST /predict - Process form data and return prediction

Usage:
    python app.py
    Then open: http://localhost:5000
"""

import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model and associated objects
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")

# Recommendation rules for each risk level
RECOMMENDATIONS = {
    "Low": {
        "action": "Normal Care",
        "details": [
            "Continue regular monitoring of the crop",
            "Maintain standard irrigation schedule",
            "Ensure proper drainage in the field",
            "Remove any infected leaves if spotted",
            "Next inspection in 7-10 days"
        ],
        "color": "#2ecc71"
    },
    "Medium": {
        "action": "Preventive Spray + Monitoring",
        "details": [
            "Apply preventive fungicide spray immediately",
            "Increase monitoring frequency to every 3-4 days",
            "Ensure adequate air circulation between vines",
            "Remove weeds that may harbor disease spores",
            "Consider copper-based fungicide application",
            "Record weather conditions daily"
        ],
        "color": "#f39c12"
    },
    "High": {
        "action": "Immediate Fungicide + Action Required",
        "details": [
            "URGENT: Apply systemic fungicide immediately",
            "Contact local agricultural extension officer",
            "Isolate affected areas if possible",
            "Remove and destroy severely infected plants",
            "Reduce irrigation to minimize moisture",
            "Apply treatment every 5-7 days until risk decreases",
            "Document all actions taken for future reference"
        ],
        "color": "#e74c3c"
    }
}

# Global variables for loaded model components
model_data = None


def load_model():
    """Load the trained model and preprocessing objects from pickle file."""
    global model_data
    try:
        with open(MODEL_PATH, "rb") as f:
            model_data = pickle.load(f)
        print(f"Model loaded successfully: {model_data['model_name']}")
        return True
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print("Please run train_model.py first to train and save the model.")
        return False
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False


@app.route("/")
def home():
    """Render the home page with the prediction form."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests.
    
    Expected form data:
        - rainfall: Rainfall in mm (0-150)
        - humidity: Humidity percentage (40-100)
        - temperature: Temperature in Celsius (10-40)
        - crop_stage: Crop stage (1=Early, 2=Mid, 3=Late)
    
    Returns:
        - prediction: Risk level (Low, Medium, High)
        - recommendation: Actionable advice
        - probabilities: Confidence scores for each class
    """
    try:
        # Validate model is loaded
        if model_data is None:
            return jsonify({
                "error": "Model not loaded. Please train the model first."
            }), 500

        # Extract and validate inputs
        rainfall = request.form.get("rainfall", type=float)
        humidity = request.form.get("humidity", type=float)
        temperature = request.form.get("temperature", type=float)
        crop_stage = request.form.get("crop_stage", type=float)

        # Input validation
        errors = []
        if rainfall is None or not (0 <= rainfall <= 200):
            errors.append("Rainfall must be between 0 and 200 mm")
        if humidity is None or not (0 <= humidity <= 100):
            errors.append("Humidity must be between 0 and 100%")
        if temperature is None or not (-10 <= temperature <= 50):
            errors.append("Temperature must be between -10 and 50°C")
        if crop_stage is None or crop_stage not in [1, 2, 3]:
            errors.append("Crop stage must be 1 (Early), 2 (Mid), or 3 (Late)")

        if errors:
            return jsonify({"errors": errors}), 400

        # Prepare features for prediction
        features = pd.DataFrame([[rainfall, humidity, temperature, crop_stage]], 
                                columns=model_data["feature_columns"])
        
        # Handle missing values using the trained imputer
        features_imputed = model_data["imputer"].transform(features)
        
        # Scale features using the trained scaler
        features_scaled = model_data["scaler"].transform(features_imputed)
        
        # Make prediction
        prediction_encoded = model_data["model"].predict(features_scaled)[0]
        prediction = model_data["label_encoder"].inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        probabilities = None
        if hasattr(model_data["model"], "predict_proba"):
            proba = model_data["model"].predict_proba(features_scaled)[0]
            classes = model_data["label_encoder"].classes_
            probabilities = {
                cls: round(float(p) * 100, 1)
                for cls, p in zip(classes, proba)
            }

        # Get recommendation
        recommendation = RECOMMENDATIONS.get(prediction, RECOMMENDATIONS["Low"])

        return jsonify({
            "prediction": prediction,
            "recommendation": recommendation,
            "probabilities": probabilities,
            "inputs": {
                "rainfall": rainfall,
                "humidity": humidity,
                "temperature": temperature,
                "crop_stage": int(crop_stage)
            }
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    API endpoint for programmatic access.
    Accepts JSON input and returns JSON output.
    
    Example request:
        POST /api/predict
        {
            "rainfall": 50,
            "humidity": 80,
            "temperature": 25,
            "crop_stage": 2
        }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Extract inputs
        rainfall = data.get("rainfall")
        humidity = data.get("humidity")
        temperature = data.get("temperature")
        crop_stage = data.get("crop_stage")

        # Validate inputs
        errors = []
        if rainfall is None or not (0 <= rainfall <= 200):
            errors.append("Rainfall must be between 0 and 200")
        if humidity is None or not (0 <= humidity <= 100):
            errors.append("Humidity must be between 0 and 100")
        if temperature is None or not (-10 <= temperature <= 50):
            errors.append("Temperature must be between -10 and 50")
        if crop_stage is None or crop_stage not in [1, 2, 3]:
            errors.append("Crop stage must be 1, 2, or 3")

        if errors:
            return jsonify({"errors": errors}), 400

        # Prepare and predict (same logic as /predict route)
        features = pd.DataFrame([[rainfall, humidity, temperature, crop_stage]],
                                columns=model_data["feature_columns"])
        features_imputed = model_data["imputer"].transform(features)
        features_scaled = model_data["scaler"].transform(features_imputed)
        
        prediction_encoded = model_data["model"].predict(features_scaled)[0]
        prediction = model_data["label_encoder"].inverse_transform([prediction_encoded])[0]
        recommendation = RECOMMENDATIONS.get(prediction, RECOMMENDATIONS["Low"])

        return jsonify({
            "prediction": prediction,
            "recommendation": recommendation["action"],
            "details": recommendation["details"]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 50)
    print("  GRAPE DISEASE PREDICTION SYSTEM")
    print("=" * 50)
    
    if load_model():
        print("\nStarting Flask server...")
        print("Open your browser and go to: http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        app.run(debug=True, host="0.0.0.0", port=5000)
    else:
        print("\nFailed to load model. Exiting.")
