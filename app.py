import streamlit as st
import pandas as pd
import requests

st.title("Analyse de Données d'Actions avec DeepSeek")

# Upload du fichier Excel
uploaded_file = st.file_uploader("Téléchargez votre fichier Excel", type=["xlsx"])

if uploaded_file is not None:
    # Lire et traiter les données
    df = pd.read_excel(uploaded_file)
    st.write("Données brutes :")
    st.write(df)

    # Interpolation linéaire pour les données manquantes
    if df.isnull().sum().any():
        st.write("Données manquantes détectées. Interpolation en cours...")
        df = df.interpolate(method='linear')
        st.write("Données après interpolation :")
        st.write(df)

    # Trier par date
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    st.write("Données organisées par date :")
    st.write(df)

    # Bouton pour analyser avec DeepSeek
    if st.button("Analyser avec DeepSeek"):
        def analyze_with_deepseek(data):
            api_url = "https://api.deepseek.com/v1/analyze"
            headers = {
                "Authorization": "Bearer VOTRE_CLE_API",
                "Content-Type": "application/json"
            }
            payload = {
                "data": data.to_dict(orient='records')
            }
            response = requests.post(api_url, json=payload, headers=headers)
            return response.json()

        result = analyze_with_deepseek(df)
        st.write("Résultats de l'analyse DeepSeek :")
        st.write(result)

    # Télécharger les données traitées
    st.download_button(
        label="Télécharger les données traitées",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="donnees_traitees.csv",
        mime="text/csv"
    )
