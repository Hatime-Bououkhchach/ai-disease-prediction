from flask import Flask, request, jsonify
import numpy as np
import pickle
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Charger les modèles une seule fois
heart_model = pickle.load(open("model.pkl", "rb"))
heart_scaler = pickle.load(open("scaler.pkl", "rb"))

diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
diabetes_scaler = pickle.load(open("diabetes_scaler.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        disease = data.get("disease")
        features = data.get("features")

        if disease == "heart":
            model = heart_model
            scaler = heart_scaler
            feature_list = [features[key] for key in ["age", "sex", "cp", "trestbps", "chol", "fbs", 
                                                     "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
        elif disease == "diabetes":
            model = diabetes_model
            scaler = diabetes_scaler
            feature_list = [features[key] for key in ["pregnancies", "glucose", "bloodpressure", 
                                                     "skinthickness", "insulin", "bmi", "dpf", "age"]]
        else:
            return jsonify({"error": "Maladie non reconnue. Utilisez 'heart' ou 'diabetes'"}), 400

        input_array = np.array([feature_list])
        input_scaled = scaler.transform(input_array)
        proba = model.predict_proba(input_scaled)[0][1] * 100

        if proba > 50:
            recommendation = "Risque élevé - Consultez un médecin immédiatement"
            status = "Élevé"
        else:
            recommendation = "Risque faible - Suivi régulier recommandé"
            status = "Faible"

        response = {
            "disease": disease,
            "risk_percentage": round(proba, 2),
            "risk_status": status,
            "recommendation": recommendation,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("🚀 API Flask démarrée sur http://127.0.0.1:5000")
    print("Testez avec Postman → POST /predict")
    app.run(debug=True, port=5000)