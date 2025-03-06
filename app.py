import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime

# Fonction pour calculer les indicateurs techniques
def calculate_technical_indicators(df):
    # Moyenne mobile sur 20 jours
    df['MA_20'] = df['Dernier'].rolling(window=20).mean()
    
    # Moyenne mobile sur 50 jours
    df['MA_50'] = df['Dernier'].rolling(window=50).mean()
    
    # Rendement quotidien
    df['Rendement_Quotidien'] = df['Dernier'].pct_change()
    
    # Rendement annuel
    df['Année'] = df['Date'].dt.year
    annual_returns = df.groupby('Année')['Rendement_Quotidien'].apply(lambda x: (1 + x).prod() - 1)
    
    return df, annual_returns

# Fonction pour traiter le fichier CSV
def process_and_analyze_file(file):
    # Lecture du fichier CSV
    df = pd.read_csv(file, delimiter=',', parse_dates=['Date'], dayfirst=True)
    
    # Conversion des nombres (formats français)
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        df[col] = df[col].str.replace('.', '').str.replace(',', '.').astype(float)
    
    # Conversion des volumes (K et M)
    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].replace({'K': '*1e3', 'M': '*1e6'}, regex=True).map(pd.eval).astype(int)
    
    # Calcul des indicateurs techniques
    df, annual_returns = calculate_technical_indicators(df)
    
    return df, annual_returns

# Interface Streamlit
def main():
    st.set_page_config(page_title="BRVM Backtester", layout="wide")
    
    st.title("🚀 Backtesting Automatique BRVM")
    st.markdown("""
    **Visualisez les performances de vos stratégies sur les données BRVM.**
    - Moyennes mobiles (20 et 50 jours)
    - Rendements annuels
    - Analyse technique simplifiée
    """)

    # Upload du fichier CSV
    uploaded_file = st.file_uploader("Importer votre fichier CSV Historique", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Traitement et analyse du fichier
            df, annual_returns = process_and_analyze_file(uploaded_file)
            
            # Affichage des résultats
            st.success("✅ Fichier analysé avec succès !")
            
            # Section 1 : Graphique des prix et moyennes mobiles
            st.subheader("Graphique des prix et moyennes mobiles")
            st.line_chart(df.set_index('Date')[['Dernier', 'MA_20', 'MA_50']])
            
            # Section 2 : Rendements annuels
            st.subheader("Rendements annuels")
            st.bar_chart(annual_returns)
            
            # Section 3 : Tableau de bord des indicateurs
            st.subheader("Indicateurs clés")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rendement total", f"{df['Rendement_Quotidien'].sum() * 100:.2f}%")
            with col2:
                st.metric("Volatilité annuelle", f"{df['Rendement_Quotidien'].std() * np.sqrt(252) * 100:.2f}%")
            with col3:
                st.metric("Ratio de Sharpe", f"{(df['Rendement_Quotidien'].mean() / df['Rendement_Quotidien'].std()) * np.sqrt(252):.2f}")
            
            # Section 4 : Téléchargement des résultats
            st.subheader("Télécharger les résultats")
            output = io.StringIO()
            df.to_csv(output, index=False)
            st.download_button(
                label="📥 Télécharger les données analysées",
                data=output.getvalue(),
                file_name="BRVM_Analyse.csv",
                mime="text/csv"
            )

        except Exception as e:
            st.error(f"❌ Erreur lors de l'analyse : {str(e)}")

if __name__ == '__main__':
    main()