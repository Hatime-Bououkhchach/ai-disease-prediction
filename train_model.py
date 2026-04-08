import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

print("=== Ré-entraînement avec Target inversé ===\n")

df = pd.read_csv("heart.csv")

features = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

X = df[features]
y = 1 - df["target"]   # ←←← ON INVERSE ICI (le plus important)

print(f"Malades après inversion : {(y==1).sum()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"\n✅ Accuracy après inversion : {accuracy:.4f} ({accuracy*100:.1f}%)")

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("✅ Modèle sauvegardé avec succès !")