import streamlit as st
import pandas as pd
import requests
import os
from datetime import datetime

# Titre de l'application
st.title("Analyse de Données d'Actions avec DeepSeek")

# Upload du fichier (CSV ou Excel)
uploaded_file = st.file_uploader("Téléchargez votre fichier (CSV ou Excel)", type=["csv", "xlsx"])

# Fonction pour interagir avec l'API DeepSeek
def analyze_with_deepseek(data):
    api_url = "https://api.deepseek.com/v1/analyze"  # Remplacez par l'URL réelle de l'API
    headers = {
        "Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}",  # Utilisez la clé API depuis les variables d'environnement
        "Content-Type": "application/json"
    }

    # Convertir les colonnes de type Timestamp en chaînes de caractères
    for col in data.columns:
        if pd.api.types.is_datetime64_any_dtype(data[col]):
            data[col] = data[col].astype(str)

    payload = {
        "data": data.to_dict(orient='records')  # Convertir les données en format JSON
    }
    response = requests.post(api_url, json=payload, headers=headers)
    return response.json()

if uploaded_file is not None:
    # Lire le fichier en fonction de son type
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    # Afficher les données brutes
    st.write("Données brutes :")
    st.write(df)

    # Vérifier les données manquantes
    if df.isnull().sum().any():
        st.write("Données manquantes détectées. Interpolation linéaire en cours...")
        df = df.interpolate(method='linear')  # Interpolation linéaire
        st.write("Données après interpolation :")
        st.write(df)

    # Trier les données par date (si une colonne "Date" existe)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])  # Convertir en format datetime
        df = df.sort_values(by='Date')  # Trier par date
        st.write("Données organisées par date :")
        st.write(df)
    else:
        st.warning("Aucune colonne 'Date' trouvée. Les données ne seront pas triées par date.")

    # Bouton pour analyser avec DeepSeek
    if st.button("Analyser avec DeepSeek"):
        try:
            result = analyze_with_deepseek(df)
            st.write("Résultats de l'analyse DeepSeek :")
            st.write(result)
        except Exception as e:
            st.error(f"Une erreur s'est produite lors de l'analyse : {e}")

    # Télécharger les données traitées
    st.download_button(
        label="Télécharger les données traitées",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="donnees_traitees.csv",
        mime="text/csv"
    )
else:
    st.info("Veuillez télécharger un fichier CSV ou Excel pour commencer.")
