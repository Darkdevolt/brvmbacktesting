import streamlit as st
import pandas as pd

def process_data(file):
    # Lecture du fichier CSV en précisant le quotechar
    df = pd.read_csv(file, sep=',', quotechar='"', dtype=str)
    
    # Nettoyage des noms de colonnes
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    
    # Vérification des colonnes
    st.write("Colonnes du fichier :", df.columns.tolist())

    # Colonnes contenant des nombres avec un séparateur de milliers '.'
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].str.replace('.', ',', regex=False)  # Remplacer le point par une virgule

    # Traitement de la colonne "Vol."
    if "Vol." in df.columns:
        df["Vol."] = df["Vol."].str.replace(',', '.', regex=False)  # Remplacer ',' par '.'
        
        def convert_vol(val):
            try:
                if 'K' in val:
                    return int(float(val.replace('K', '')) * 1000)  # Supprimer 'K' et multiplier
                return int(float(val))  # Convertir normalement
            except:
                return 0  # En cas d'erreur

        df["Vol."] = df["Vol."].apply(convert_vol)

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