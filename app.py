import streamlit as st
import pandas as pd
import os

# Configuration de la page
st.set_page_config(page_title="Backtesting BRVM", layout="wide")

st.title("📈 Backtesting sur NTLC - BRVM")

# Upload du fichier CSV ou Excel
uploaded_file = st.file_uploader("Upload ton fichier de données (CSV ou Excel)", type=["csv", "xls", "xlsx"])

if uploaded_file is not None:
    # Extraction de l'extension du fichier
    filename = uploaded_file.name.lower()
    file_extension = os.path.splitext(filename)[1]

    try:
        # Lecture du fichier selon son format
        if file_extension == ".csv":
            data = pd.read_csv(uploaded_file)
        elif file_extension in [".xls", ".xlsx"]:
            data = pd.read_excel(uploaded_file)
        else:
            st.error("❌ Format de fichier non supporté. Veuillez uploader un fichier CSV ou Excel.")
            st.stop()

        # Vérification des colonnes
        expected_columns = ["Date", "Ouverture", "Clôture", "Volume"]
        missing_columns = [col for col in expected_columns if col not in data.columns]

        if missing_columns:
            st.error(f"❌ Colonnes manquantes : {missing_columns}. Vérifie le format du fichier.")
            st.stop()

        # Conversion de la colonne Date
        data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

        # Vérification des valeurs NaN après conversion
        if data["Date"].isna().sum() > 0:
            st.warning("⚠️ Certaines dates n'ont pas pu être converties. Vérifie le format de la colonne Date.")

        # Affichage des premières lignes
        st.subheader("📊 Aperçu des données chargées")
        st.write(data.head())

        # Graphique simple de l'évolution des prix de clôture
        st.subheader("📉 Évolution des prix de clôture")
        st.line_chart(data.set_index("Date")["Clôture"])

    except Exception as e:
        st.error(f"❌ Erreur lors du traitement du fichier : {e}")

else:
    st.info("📤 Charge un fichier CSV ou Excel pour commencer l'analyse.")