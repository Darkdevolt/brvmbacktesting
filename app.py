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
            try:
                data = pd.read_csv(uploaded_file, encoding="utf-8", delimiter=";", error_bad_lines=False)
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier CSV : {e}")
                st.stop()
        
        elif file_extension in [".xls", ".xlsx"]:
            try:
                data = pd.read_excel(uploaded_file, engine="openpyxl")
            except Exception as e:
                st.error(f"❌ Erreur lors de la lecture du fichier Excel : {e}")
                st.stop()

        else:
            st.error("❌ Format de fichier non supporté. Veuillez uploader un fichier CSV ou Excel.")
            st.stop()

        # Affichage des premières lignes pour vérifier les colonnes
        st.subheader("📊 Aperçu des données chargées")
        st.write(data.head())

        # Vérification automatique des colonnes
        st.subheader("✅ Vérification des colonnes")
        st.write("Colonnes détectées :", list(data.columns))

        # Tentative d'auto-détection de la colonne de date
        date_columns = [col for col in data.columns if "date" in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            try:
                data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
                st.success(f"📅 Colonne date détectée : **{date_col}** et convertie en format datetime.")
            except Exception as e:
                st.error(f"❌ Erreur de conversion de la colonne Date : {e}")
                st.stop()
        else:
            st.error("❌ Aucune colonne Date trouvée. Vérifiez votre fichier.")
            st.stop()

        # Vérification et correction des types de colonnes
        for col in data.columns:
            if data[col].dtype == "object":
                try:
                    data[col] = pd.to_numeric(data[col].str.replace(",", ".").str.replace(" ", ""), errors="coerce")
                except:
                    pass

        # Détection de la colonne "Clôture" ou "Close" pour le graphique
        possible_price_cols = [col for col in data.columns if "clôture" in col.lower() or "close" in col.lower()]
        if possible_price_cols:
            price_col = possible_price_cols[0]
            st.subheader("📉 Évolution des prix de clôture")
            st.line_chart(data.set_index(date_col)[price_col])
        else:
            st.error("❌ Impossible de détecter une colonne de prix de clôture. Vérifiez votre fichier.")

    except Exception as e:
        st.error(f"❌ Erreur inattendue : {e}")

else:
    st.info("📤 Charge un fichier CSV ou Excel pour commencer l'analyse.")