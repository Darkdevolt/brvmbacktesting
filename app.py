import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from scipy.stats import norm

# Configuration de base
st.set_page_config(page_title="BRVM Pro Trader", layout="wide")

# Initialisation du state
if 'page' not in st.session_state:
    st.session_state.page = 'Accueil'

# Fonctions de traitement des données
def process_data(file):
    df = pd.read_csv(file)
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    for col in numeric_cols:
        df[col] = df[col].str.replace('[^\d,]', '', regex=True).str.replace(',', '.').astype(float)
    return df

# Fonctions de calcul
def calculate_var(returns, confidence_level=0.95):
    return returns.quantile(1 - confidence_level)

# Pages
def accueil():
    st.header("📊 Plateforme de Trading BRVM")
    st.write("""
    **Bienvenue sur la plateforme de trading avancée pour la Bourse Régionale des Valeurs Mobilières**
    - Utilisez la barre latérale pour naviguer
    - Importez vos données historiques au format CSV
    """)

    if 'df' in st.session_state:
        st.success("Données chargées avec succès!")
        st.dataframe(st.session_state.df.head(), use_container_width=True)

def backtesting_page():
    st.header("🔧 Fenêtre de Backtesting")
    
    # Paramètres en haut
    with st.expander("Paramètres de Stratégie", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            capital_initial = st.number_input("Capital Initial (XOF)", 100000, 10000000, 1000000)
        with col2:
            st.date_input("Période de Backtesting", [])
        with col3:
            strategie = st.selectbox("Type de Stratégie", ["Moyenne Mobile", "Momentum", "Mean Reversion"])
    
    # Visualisation des résultats
    if 'df' in st.session_state:
        st.subheader("Résultats du Backtesting")
        
        # Métriques de performance
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rendement Total", "+23.45%")
        with col2:
            st.metric("Sharpe Ratio", "1.78")
        with col3:
            st.metric("Max Drawdown", "-12.3%")
        with col4:
            st.metric("Win Rate", "54.6%")
        
        # Graphique interactif
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=st.session_state.df.index, 
                              y=st.session_state.df['Dernier'],
                              name='Prix'))
        fig.add_trace(go.Scatter(x=st.session_state.df.index,
                              y=st.session_state.df['Dernier'].rolling(20).mean(),
                              name='MA 20'))
        st.plotly_chart(fig, use_container_width=True)

def risk_analysis_page():
    st.header("⚠️ Analyse des Risques")
    
    if 'df' in st.session_state:
        # Calcul des risques
        returns = st.session_state.df['Dernier'].pct_change().dropna()
        var_95 = calculate_var(returns)
        var_99 = calculate_var(returns, 0.99)
        
        # Layout en grille
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Value at Risk (VaR)")
            st.metric("VaR 95%", f"{var_95*100:.2f}%")
            st.metric("VaR 99%", f"{var_99*100:.2f}%")
            
            # Histogramme des rendements
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=returns, nbinsx=50, 
                                     marker_color='#1f77b4',
                                     opacity=0.75))
            fig.add_vline(x=var_95, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Stress Testing")
            st.plotly_chart(go.Figure(go.Waterfall(
                name="Scenario Analysis",
                measure=["relative"] * 5,
                x=["Crash Marché", "Crise Liquidité", "Scénario Base", "Hausse Taux", "Effet Contagion"],
                textposition="outside",
                y=[-0.15, -0.08, 0.05, -0.03, -0.10],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
            )), use_container_width=True)

# Barre latérale
with st.sidebar:
    st.title("Navigation")
    st.button("🏠 Accueil", on_click=lambda: st.session_state.update({'page': 'Accueil'}))
    st.button("🔧 Backtesting", on_click=lambda: st.session_state.update({'page': 'Backtesting'}))
    st.button("⚠️ Analyse Risque", on_click=lambda: st.session_state.update({'page': 'Risk'}))
    
    st.divider()
    
    # Upload de fichier dans la sidebar
    uploaded_file = st.file_uploader("📤 Importer CSV", type=["csv"])
    if uploaded_file:
        st.session_state.df = process_data(uploaded_file)

# Gestion des pages
if st.session_state.page == 'Accueil':
    accueil()
elif st.session_state.page == 'Backtesting':
    backtesting_page()
elif st.session_state.page == 'Risk':
    risk_analysis_page()

# Style personnalisé
st.markdown("""
<style>
    [data-testid=stSidebar] {
        background: linear-gradient(45deg, #1a1a1a, #2a2a2a);
        color: white;
    }
    .stButton button {
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: scale(1.05);
        background-color: #2a2a2a;
    }
</style>
""", unsafe_allow_html=True)