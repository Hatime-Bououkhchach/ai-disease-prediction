import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF

st.set_page_config(page_title="AI Early Disease Prediction", layout="wide")

st.title("🧬 AI-Based Early Disease Prediction System")
st.markdown("**Système intelligent de prédiction précoce de maladies** - PFE 2026")

API_URL = "http://127.0.0.1:5000/predict"

def create_pdf(disease, inputs_dict, risk, recommendation):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Rapport de Prédiction - AI Early Disease Prediction System", ln=1, align="C")
    pdf.ln(10)
    
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Maladie : {disease}", ln=1)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=1)
    pdf.ln(5)
    
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Données saisies :", ln=1)
    pdf.set_font("Arial", "", 11)
    for key, value in inputs_dict.items():
        pdf.cell(0, 8, f"- {key} : {value}", ln=1)
    
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Risque calculé : {risk:.1f}%", ln=1)
    
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, recommendation)
    
    pdf.ln(10)
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, "Modèle utilisé : Random Forest (via API Flask)", ln=1, align="C")
    
    return pdf.output(dest="S").encode("latin-1", errors="replace")

tab1, tab2, tab3 = st.tabs(["🔮 Prédiction", "📊 Modèles & Performances", "📈 EDA"])

with tab1:
    disease_choice = st.selectbox("**Choisissez la maladie à prédire** :", 
                                  ["❤️ Maladie Cardiaque", "🩸 Diabète"])

    if disease_choice == "❤️ Maladie Cardiaque":
        st.subheader("❤️ Prédiction Maladie Cardiaque")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Âge", 20, 100, 50)
            sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme")
            cp = st.number_input("Type de douleur thoracique (0-3)", 0, 3, 0)
            trestbps = st.number_input("Tension artérielle", 90, 200, 120)
            chol = st.number_input("Cholestérol", 100, 600, 200)
            fbs = st.number_input("Glycémie à jeun (1=oui)", 0, 1, 0)
        with col2:
            restecg = st.number_input("ECG au repos (0-2)", 0, 2, 0)
            thalach = st.number_input("Fréquence cardiaque max", 60, 220, 150)
            exang = st.number_input("Angine d'effort (0/1)", 0, 1, 0)
            oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
            slope = st.number_input("Slope (0-2)", 0, 2, 1)
            ca = st.number_input("CA (0-4)", 0, 4, 0)
            thal = st.number_input("Thal (1-3)", 1, 3, 3)

        if st.button("🔍 Prédire risque cardiaque", type="primary"):
            features = {
                "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
                "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
                "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
            }
            
            try:
                response = requests.post(API_URL, json={"disease": "heart", "features": features}, timeout=10)
                response.raise_for_status()
                result = response.json()
                proba = result["risk_percentage"]
                
                st.progress(int(proba))
                st.write(f"**Risque de maladie cardiaque : {proba:.1f}%**")
                
                if proba > 50:
                    st.error("⚠️ RISQUE ÉLEVÉ - Consultez un médecin")
                    reco = "Recommandation : Consultez un cardiologue rapidement. Contrôlez tension, cholestérol et pratiquez une activité physique régulière."
                else:
                    st.success("✅ Risque faible")
                    reco = "Recommandation : Continuez une bonne hygiène de vie. Suivi annuel recommandé."

                inputs = {"Âge": age, "Sexe": "Homme" if sex == 1 else "Femme", "Tension": trestbps, 
                          "Cholestérol": chol}
                
                pdf_bytes = create_pdf("Maladie Cardiaque", inputs, proba, reco)
                st.download_button("📄 Télécharger le rapport PDF", data=pdf_bytes, 
                                 file_name="Rapport_Cardiaque.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {str(e)}")
                st.info("Vérifiez que api.py est bien lancé.")

    else:  # Diabète
        st.subheader("🩸 Prédiction Diabète")
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Nombre de grossesses", 0, 20, 2)
            glucose = st.number_input("Glucose (mg/dl)", 0, 200, 120)
            bloodpressure = st.number_input("Tension", 0, 150, 70)
            skinthickness = st.number_input("Épaisseur peau (mm)", 0, 100, 20)
        with col2:
            insulin = st.number_input("Insuline", 0, 900, 80)
            bmi = st.number_input("IMC (BMI)", 0.0, 70.0, 28.0)
            dpf = st.number_input("Fonction pedigree", 0.0, 3.0, 0.6)
            age = st.number_input("Âge", 20, 100, 40)

        if st.button("🔍 Prédire risque de diabète", type="primary"):
            features = {
                "pregnancies": pregnancies, "glucose": glucose, "bloodpressure": bloodpressure,
                "skinthickness": skinthickness, "insulin": insulin, "bmi": bmi, "dpf": dpf, "age": age
            }
            
            try:
                response = requests.post(API_URL, json={"disease": "diabetes", "features": features}, timeout=10)
                response.raise_for_status()
                result = response.json()
                proba = result["risk_percentage"]
                
                st.progress(int(proba))
                st.write(f"**Risque de diabète : {proba:.1f}%**")
                
                if proba > 50:
                    st.error("⚠️ RISQUE ÉLEVÉ")
                    reco = "Recommandation : Consultez un endocrinologue."
                else:
                    st.success("✅ Risque faible")
                    reco = "Recommandation : Maintenez une bonne hygiène de vie."

                inputs = {"Âge": age, "Glucose": glucose, "IMC": bmi}
                pdf_bytes = create_pdf("Diabète", inputs, proba, reco)
                st.download_button("📄 Télécharger le rapport PDF", data=pdf_bytes, 
                                 file_name="Rapport_Diabete.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Erreur de connexion à l'API : {str(e)}")

with tab2:
    st.subheader("📊 Comparaison des modèles")
    st.success("✅ Meilleur modèle : Random Forest (via API Flask)")

with tab3:
    st.subheader("📈 EDA")
    option = st.radio("Dataset :", ["Maladie Cardiaque", "Diabète"])
    if option == "Maladie Cardiaque":
        df = pd.read_csv("heart.csv")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

st.caption("PFE 2026 - Frontend Streamlit + Backend Flask")
