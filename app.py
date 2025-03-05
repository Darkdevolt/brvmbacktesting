import streamlit as st
import pandas as pd
import numpy as np

st.title("Traitement de fichiers CSV")

# Upload du fichier CSV
uploaded_file = st.file_uploader("Choisissez un fichier CSV", type="csv")

if uploaded_file is not None:
    # Lire le fichier CSV
    df = pd.read_csv(uploaded_file)

    # Afficher les données brutes
    st.write("Données brutes")
    st.write(df)

    # Traitement des données (exemple : trier par une colonne)
    st.write("Données triées par la première colonne")
    sorted_df = df.sort_values(by=df.columns[0])
    st.write(sorted_df)

    # Exemple de traitement supplémentaire (calcul de la moyenne d'une colonne)
    if df.select_dtypes(include=[np.number]).columns.any():
        numeric_column = st.selectbox("Sélectionnez une colonne numérique pour calculer la moyenne", df.select_dtypes(include=[np.number]).columns)
        mean_value = df[numeric_column].mean()
        st.write(f"La moyenne de la colonne {numeric_column} est : {mean_value}")