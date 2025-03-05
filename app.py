import streamlit as st
import pandas as pd
import io

# Fonction de correction des prix
def corriger_prix(val):
    try:
        val = str(val).replace(".", "")  # Supprime le point
        return int(val)  # Convertit en entier
    except:
        return val  # Si erreur, on garde la valeur originale

# Fonction de nettoyage des données
def nettoyer_fichier(uploaded_file):
    try:
        # Chargement du fichier CSV
        df = pd.read_csv(uploaded_file, delimiter=",", encoding="utf-8")

        # Renommage des colonnes (ajuster selon le fichier réel)
        df.columns = ["Date", "Dernier", "Ouverture", "Plus Haut", "Plus Bas", "Volume", "Variation %"][:df.shape[1]]

        # Correction des prix sur les bonnes colonnes
        colonnes_prix = ["Dernier", "Ouverture", "Plus Haut", "Plus Bas"]
        for col in colonnes_prix:
            df[col] = df[col].apply(corriger_prix)

        # Correction des dates
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

        return df
    except Exception as e:
        st.error(f"Erreur : {e}")
        return None

# Interface Streamlit
st.title("Nettoyage des Données de Bourse")

uploaded_file = st.file_uploader("Uploader un fichier CSV", type=["csv"])

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