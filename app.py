import streamlit as st
import pandas as pd
import io

# Fonction de nettoyage des données
def nettoyer_fichier(uploaded_file):
    try:
        # Détection du format du fichier
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, delimiter=",", encoding="utf-8")
        else:
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

        # Renommage des colonnes pour correspondre aux données attendues
        df.columns = ["Date", "Dernier", "Ouverture", "Plus Haut", "Plus Bas", "Volume", "Variation %"][:df.shape[1]]

        # Conversion des valeurs en nombres et multiplication par 1000
        cols_a_corriger = ["Dernier", "Ouverture", "Plus Haut", "Plus Bas"]
        for col in cols_a_corriger:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float) * 1000  # Correction de la séparation décimale

        # Correction des dates
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None

# Interface Streamlit
st.title("Nettoyage des Données de Bourse")

uploaded_file = st.file_uploader("Uploader un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    df_propre = nettoyer_fichier(uploaded_file)

    if df_propre is not None:
        st.success("Fichier traité avec succès !")
        st.write("### Données Nettoyées :")
        st.dataframe(df_propre)

        # Télécharger le fichier nettoyé
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_propre.to_excel(writer, index=False, sheet_name="Données Nettoyées")
        processed_data = output.getvalue()

        st.download_button(
            label="Télécharger le fichier nettoyé",
            data=processed_data,
            file_name="data_propre.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )