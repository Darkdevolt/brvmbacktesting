import streamlit as st
import pandas as pd

def process_data(file):
    # Lecture du CSV avec reconnaissance automatique du séparateur de milliers
    df = pd.read_csv(file, quotechar='"', sep=',', thousands='.')
    
    # Nettoyage des noms de colonnes : suppression d'espaces inutiles et du BOM
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    st.write("Colonnes du fichier :", df.columns.tolist())
    
    # Conversion des colonnes numériques déjà lues comme nombres (float) en entier
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        df[col] = df[col].fillna(0).astype(int)
    
    # Traitement de la colonne "Vol."
    # Les valeurs de Vol. sont au format "3,13K" (virgule pour la décimale)
    df["Vol."] = df["Vol."].astype(str).str.replace(',', '.', regex=False)
    
    def convert_vol(x):
        if 'K' in x:
            return float(x.replace('K','')) * 1000
        else:
            return float(x)
    
    df["Vol."] = df["Vol."].apply(convert_vol)
    df["Vol."] = df["Vol."].fillna(0).astype(int)
    
    return df

def main():
    st.title("Traitement de CSV - Application BRVM")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df_processed = process_data(uploaded_file)
            st.write("Données traitées :", df_processed)
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")

if __name__ == '__main__':
    main()