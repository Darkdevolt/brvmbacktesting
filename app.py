import pandas as pd
import streamlit as st

def process_data(uploaded_file):
    # Lire le CSV avec gestion des guillemets et séparateurs
    df = pd.read_csv(uploaded_file, dtype=str, quotechar='"', thousands='.', decimal=',')

    # Traitement spécifique pour chaque colonne
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    
    # Conversion des nombres avec séparateur de milliers
    for col in numeric_cols:
        df[col] = df[col].str.replace('.', '', regex=False).astype(float)
    
    # Traitement de la colonne Vol.
    if 'Vol.' in df.columns:
        df['Vol.'] = (
            df['Vol.']
            .str.replace('K', '', regex=False)
            .str.replace(',', '.')
            .astype(float)
            * 1000
        )
    
    # Traitement de la colonne Variation %
    if 'Variation %' in df.columns:
        df['Variation %'] = (
            df['Variation %']
            .str.replace('%', '', regex=False)
            .str.replace(',', '.')
            .astype(float)
        )
    
    return df

def main():
    st.title("📊 Traitement de Données Financières")
    
    uploaded_file = st.file_uploader("Déposez votre fichier CSV", type=["csv"])
    
    if uploaded_file:
        df = process_data(uploaded_file)
        
        st.success("Traitement réussi !")
        st.dataframe(df.style.format({
            'Vol.': '{:,.0f}',
            'Variation %': '{:.2f}%'
        }))
        
        csv = df.to_csv(index=False, sep=';', decimal=',')
        st.download_button(
            label="📥 Télécharger les données traitées",
            data=csv,
            file_name="donnees_traitees.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()