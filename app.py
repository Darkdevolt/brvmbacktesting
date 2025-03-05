import streamlit as st
import pandas as pd

def process_data(file):
    # Lecture du CSV en spécifiant le séparateur et le quotechar
    df = pd.read_csv(file, quotechar='"', sep=',')
    
    # Nettoyage des noms de colonnes : suppression d'espaces inutiles et du BOM
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    st.write("Colonnes du fichier :", df.columns.tolist())
    
    # Traitement des colonnes numériques (le point est un séparateur de milliers)
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.replace('.', '', regex=False)
        # Conversion en nombre et remplacement des valeurs non convertibles par 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Traitement de la colonne "Vol."
    df["Vol."] = df["Vol."].astype(str)
    df["Vol."] = df["Vol."].str.replace(',', '.', regex=False)
    
    def convert_vol(x):
        try:
            if 'K' in x:
                return float(x.replace('K', '')) * 1000
            else:
                return float(x)
        except:
            return None

    df["Vol."] = df["Vol."].apply(convert_vol)
    # Remplacement des éventuels NaN par 0, puis conversion en entier
    df["Vol."] = pd.to_numeric(df["Vol."], errors='coerce').fillna(0).astype(int)
    
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