import streamlit as st
import pandas as pd

def clean_data(df):
    # Traitement des colonnes numériques (suppression du séparateur de milliers)
    for col in ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]:
        # Suppression du point pour convertir "9.300" en 9300
        df[col] = df[col].str.replace('.', '', regex=False).astype(int)
    
    # Traitement de la colonne "Vol."
    df["Vol."] = df["Vol."].str.replace(',', '.', regex=False)  # Convertir la virgule en point
    df["Vol."] = df["Vol."].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x)).astype(int)
    return df

st.title("Traitement de fichier CSV")

uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
if uploaded_file is not None:
    # Lecture du fichier
    df = pd.read_csv(uploaded_file, quotechar='"')
    st.write("Données brutes", df.head())
    
    # Nettoyage des données
    df_clean = clean_data(df)
    st.write("Données nettoyées", df_clean.head())