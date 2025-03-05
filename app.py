import streamlit as st
import pandas as pd
import io

# Fonction de nettoyage des données
def nettoyer_fichier(uploaded_file):
    try:
        # Détecter si c'est un CSV ou un Excel
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, delimiter=",", encoding="utf-8")
        else:
            xls = pd.ExcelFile(uploaded_file)
            df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

        # Vérifier si les données sont mal stockées en une seule colonne
        if df.shape[1] == 1:
            df = df.iloc[:, 0].str.split(',', expand=True)

        # Renommer les colonnes correctement
        df.columns = ["Date", "Dernier", "Ouverture", "Plus Haut", "Plus Bas", "Volume", "Variation %"] + \
                     [f"Colonne_{i}" for i in range(7, df.shape[1])]

        # Supprimer les colonnes inutiles si elles existent
        df = df[["Date", "Dernier", "Ouverture", "Plus Haut", "Plus Bas", "Volume", "Variation %"]]

        # Fonction pour nettoyer les valeurs numériques
        def clean_numeric(value):
            if isinstance(value, str):
                value = value.replace('"', '').replace(' ', '').replace('K', '000').replace('%', '')
                try:
                    return float(value) / 100 if '%' in value else float(value)
                except ValueError:
                    return None  # Si la valeur est incorrecte
            return value

        # Appliquer la correction sur toutes les colonnes sauf la date
        for col in ["Dernier", "Ouverture", "Plus Haut", "Plus Bas", "Volume", "Variation %"]:
            df[col] = df[col].apply(clean_numeric)

        # Correction des prix (ex: 9.3 → 9300)
        for col in ["Dernier", "Ouverture", "Plus Haut", "Plus Bas"]:
            df[col] = df[col] * 1000

        # Convertir la colonne Date en format datetime
        df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors='coerce')

        return df
    except Exception as e:
        st.error(f"Erreur lors du nettoyage du fichier : {e}")
        return None

# Interface Streamlit
st.title("Nettoyage et Visualisation des Données CSV/Excel")

# Upload du fichier
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
        