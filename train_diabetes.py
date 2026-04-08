import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Chargement des données
df = pd.read_csv("heart.csv")

# Features et Target
X = df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
        "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
y = df["target"]

# Important : Si dans ton dataset 0 = maladie, on inverse
# Décommente la ligne ci-dessous si nécessaire :
# y = 1 - df["target"]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Meilleur modèle
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Évaluation
score = model.score(X_test, y_test)
print(f"Accuracy du modèle : {score:.4f} ({score*100:.1f}%)")

# Sauvegarde
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Nouveau modèle Random Forest créé avec succès !")
print("Fichiers mis à jour : model.pkl + scaler.pkl")