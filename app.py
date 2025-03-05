import streamlit as st
import pandas as pd

def process_data(file):
    # Lecture du CSV en spécifiant simplement le séparateur et le quotechar
    df = pd.read_csv(file, sep=',', quotechar='"')
    
    # Nettoyage des noms de colonnes : suppression d'espaces inutiles et du BOM
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    
    # Pour vérifier les noms de colonnes détectés
    st.write("Colonnes du fichier :", df.columns.tolist())
    
    # Colonnes numériques à convertir (où le point est un séparateur de milliers)
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        # 1) Convertir en chaîne (pour être sûr de pouvoir remplacer)
        df[col] = df[col].astype(str)
        # 2) Supprimer le point (séparateur de milliers), ex: "9.300" -> "9300"
        df[col] = df[col].str.replace('.', '', regex=False)
        # 3) Convertir en entier. Ainsi "9300" devient l’entier 9300 (et non 93).
        #    Les valeurs non convertibles deviennent NaN, qu'on remplace par 0 si besoin.
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Traitement de la colonne "Vol."
    # Exemple : "3,13K" -> remplacer ',' par '.' -> "3.13K" -> multiplier par 1000 -> 3130
    df["Vol."] = df["Vol."].astype(str)
    df["Vol."] = df["Vol."].str.replace(',', '.', regex=False)
    
    def convert_vol(val):
        if 'K' in val:
            # Enlever 'K' et multiplier par 1000
            return float(val.replace('K', '')) * 1000
        else:
            return float(val)

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