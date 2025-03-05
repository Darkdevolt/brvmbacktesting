import streamlit as st
import pandas as pd

def process_data(file):
    # Lecture du CSV en spécifiant le séparateur et le quotechar
    df = pd.read_csv(file, quotechar='"', sep=',')
    
    # Nettoyage des noms de colonnes : suppression d'espaces inutiles et du BOM
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    
    # Affichage des noms de colonnes pour vérifier qu'ils correspondent
    st.write("Colonnes du fichier :", df.columns.tolist())
    
    # Liste des colonnes numériques à traiter (séparateur de milliers : le point)
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        # Suppression du point servant de séparateur de milliers
        # Attention : "9.300" doit devenir 9300 et non 9.3
        df[col] = df[col].str.replace('.', '', regex=False).astype(int)
    
    # Traitement de la colonne "Vol." : conversion de la virgule en point et multiplication si 'K' est présent
    df["Vol."] = df["Vol."].str.replace(',', '.', regex=False)
    df["Vol."] = df["Vol."].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in x else float(x)).astype(int)
    
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