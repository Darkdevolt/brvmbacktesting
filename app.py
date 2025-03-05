import streamlit as st
import pandas as pd

def process_data(file):
    # Lecture du CSV en spécifiant le séparateur et le quotechar
    df = pd.read_csv(file, quotechar='"', sep=',')
    
    # Nettoyage des noms de colonnes : suppression d'espaces inutiles et du BOM
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    st.write("Colonnes du fichier :", df.columns.tolist())
    
    # Liste des colonnes numériques à traiter (le point est un séparateur de milliers)
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        # On s'assure que la colonne est traitée comme chaîne de caractères
        df[col] = df[col].astype(str)
        # Suppression du point servant de séparateur de milliers (ex: "9.300" -> "9300")
        df[col] = df[col].str.replace('.', '', regex=False)
        # Conversion en nombre entier
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Traitement de la colonne "Vol." :
    # 1. Convertir la colonne en chaîne de caractères
    df["Vol."] = df["Vol."].astype(str)
    # 2. Remplacer la virgule par un point pour gérer la partie décimale
    df["Vol."] = df["Vol."].str.replace(',', '.', regex=False)
    # 3. Conversion : si la valeur contient 'K', multiplier par 1000
    def convert_vol(x):
        if 'K' in x:
            # Enlever le 'K' et multiplier par 1000
            return float(x.replace('K', '')) * 1000
        else:
            return float(x)
    
    df["Vol."] = df["Vol."].apply(convert_vol).astype(int)
    
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