import pandas as pd
import streamlit as st

def clean_data(df):
    # Inverser l'ordre des dates (du plus ancien au plus récent)
    df = df.iloc[::-1].reset_index(drop=True)
    
    # Convertir les colonnes numériques
    df['Dernier'] = df['Dernier'].str.replace(',', '.').astype(float)
    df['Ouv.'] = df['Ouv.'].str.replace(',', '.').astype(float)
    df['Plus Haut'] = df['Plus Haut'].str.replace(',', '.').astype(float)
    df['Plus Bas'] = df['Plus Bas'].str.replace(',', '.').astype(float)
    
    # Convertir la colonne 'Vol.' en valeurs numériques
    df['Vol.'] = df['Vol.'].str.replace('K', '').str.replace(',', '.').astype(float) * 1000
    
    # Convertir la colonne 'Variation %' en valeurs numériques
    df['Variation %'] = df['Variation %'].str.replace('%', '').str.replace(',', '.').astype(float) / 100
    
    return df

def main():
    st.title("Nestlé Côte d'Ivoire (BRVM) - Données Historiques Corrigées")
    
    # Charger le fichier CSV
    uploaded_file = st.file_uploader("Téléversez votre fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Lire le fichier CSV
        df = pd.read_csv(uploaded_file, parse_dates=["Date"])
        
        # Nettoyer les données
        df_cleaned = clean_data(df)
        
        # Afficher les données corrigées
        st.write("Données corrigées :")
        st.dataframe(df_cleaned)
        
        # Télécharger les données corrigées
        st.download_button(
            label="Télécharger les données corrigées",
            data=df_cleaned.to_csv(index=False).encode('utf-8'),
            file_name="NTLC_Donnees_Historiques_Corrige.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()