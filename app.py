import streamlit as st
import pandas as pd
import io

# Fonction de nettoyage des données avec correction automatique
def nettoyer_fichier(uploaded_file):
    try:
        # Chargement du fichier CSV ou Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, delimiter=",", encoding="utf-8")
        else:
            df = pd.read_excel(uploaded_file)

        # Renommage des colonnes
        df.columns = ["Date", "Dernier", "Ouverture", "Plus Haut", "Plus Bas", "Volume", "Variation %"][:df.shape[1]]

        # Conversion des nombres (gestion des virgules)
        cols_a_corriger = ["Dernier", "Ouverture", "Plus Haut", "Plus Bas"]
        for col in cols_a_corriger:
            df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

        # Détection des valeurs mal enregistrées
        for col in cols_a_corriger:
            max_val = df[col].max()
            
            if max_val < 100:  # Si toutes les valeurs sont inférieures à 100, elles sont sûrement en "petit format"
                df[col] = df[col] * 1000  # Correction automatique

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