import pickle
import numpy as np

# Charger le modèle
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

print("Modèle chargé avec succès\n")

# Test avec valeurs à très haut risque
high_risk = np.array([[65, 1, 0, 160, 300, 1, 2, 110, 1, 4.0, 0, 3, 3]])

scaled = scaler.transform(high_risk)
proba = model.predict_proba(scaled)[0][1] * 100

print(f"Risque prédit pour ce patient à haut risque : {proba:.1f}%")
print("Classe prédite :", model.predict(scaled)[0])