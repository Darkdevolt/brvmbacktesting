import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from io import BytesIO
import base64

# Configuration de la page
st.set_page_config(
    page_title="Performance CAC 40",
    layout="wide",
    page_icon="📊"
)

# Style CSS
st.markdown("""
<style>
    .main {max-width: 1200px;}
    .metric-box {border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px 0;}
    .positive {color: #2ecc71;}
    .negative {color: #e74c3c;}
    .header {font-size: 1.5em; font-weight: bold; margin: 10px 0;}
    .ticker-header {background-color: #f0f2f6; padding: 10px; border-radius: 5px;}
</style>
""", unsafe_allow_html=True)

# Titre
st.title('📊 Performances des actions du CAC 40')

# Liste des composants du CAC 40 avec leurs tickers Yahoo Finance
CAC40_TICKERS = {
    'AC.PA': 'Accor',
    'AI.PA': 'Air Liquide',
    'AIR.PA': 'Airbus',
    'MT.PA': 'ArcelorMittal',
    'ATO.PA': 'Atos',
    'CS.PA': 'AXA',
    'BNP.PA': 'BNP Paribas',
    'EN.PA': 'Bouygues',
    'CAP.PA': 'Capgemini',
    'CA.PA': 'Carrefour',
    'ACA.PA': 'Crédit Agricole',
    'BN.PA': 'Danone',
    'DSY.PA': 'Dassault Systèmes',
    'ENGI.PA': 'Engie',
    'EL.PA': 'EssilorLuxottica',
    'RMS.PA': 'Hermès',
    'KER.PA': 'Kering',
    'LR.PA': 'Legrand',
    'OR.PA': 'L\'Oréal',
    'MC.PA': 'LVMH',
    'ML.PA': 'Michelin',
    'ORA.PA': 'Orange',
    'RI.PA': 'Pernod Ricard',
    'PUB.PA': 'Publicis',
    'RNO.PA': 'Renault',
    'SAF.PA': 'Safran',
    'SGO.PA': 'Saint-Gobain',
    'SAN.PA': 'Sanofi',
    'SU.PA': 'Schneider Electric',
    'GLE.PA': 'Société Générale',
    'STLA.PA': 'Stellantis',
    'STM.PA': 'STMicroelectronics',
    'TEP.PA': 'Teleperformance',
    'HO.PA': 'Thales',
    'TTE.PA': 'TotalEnergies',
    'VIE.PA': 'Veolia',
    'DG.PA': 'Vinci',
    'VIV.PA': 'Vivendi'
}

# Sidebar
with st.sidebar:
    st.header('Paramètres')
    period = st.selectbox('Période', ['1j', '1sem', '1mo', '3mo', '6mo', '1an', 'YTD'], index=5)
    benchmark = st.selectbox('Indice de référence', ['^FCHI', '^GSPC (S&P 500)', '^IXIC (NASDAQ)'], index=0)
    st.markdown("---")
    st.markdown("**Mise à jour:** " + datetime.now().strftime("%d/%m/%Y %H:%M"))
    st.markdown("---")
    st.markdown("""
    **Sources:**
    - Yahoo Finance
    - Wikipédia CAC 40
    """)

# Fonction pour récupérer les données
@st.cache_data(ttl=3600)  # Cache pour 1 heure
def get_stock_data(tickers, period):
    end_date = datetime.now()
    
    if period == '1j':
        start_date = end_date - timedelta(days=1)
    elif period == '1sem':
        start_date = end_date - timedelta(weeks=1)
    elif period == '1mo':
        start_date = end_date - timedelta(days=30)
    elif period == '3mo':
        start_date = end_date - timedelta(days=90)
    elif period == '6mo':
        start_date = end_date - timedelta(days=180)
    elif period == '1an':
        start_date = end_date - timedelta(days=365)
    elif period == 'YTD':
        start_date = datetime(end_date.year, 1, 1)
    
    data = yf.download(list(tickers.keys()), start=start_date, end=end_date, group_by='ticker')
    return data

# Récupération des données
try:
    data = get_stock_data(CAC40_TICKERS, period)
    if data.empty:
        st.error("Erreur lors de la récupération des données. Veuillez réessayer plus tard.")
        st.stop()
except Exception as e:
    st.error(f"Erreur: {str(e)}")
    st.stop()

# Calcul des performances
performance_data = []
for ticker in CAC40_TICKERS:
    try:
        if ticker in data:
            close_prices = data[ticker]['Close']
            if len(close_prices) > 1:
                start_price = close_prices[0]
                end_price = close_prices[-1]
                change = ((end_price - start_price) / start_price) * 100
                performance_data.append({
                    'Ticker': ticker,
                    'Société': CAC40_TICKERS[ticker],
                    'Prix': end_price,
                    'Variation (%)': change,
                    'Performance': 'positive' if change >= 0 else 'negative'
                })
    except:
        continue

performance_df = pd.DataFrame(performance_data).sort_values('Variation (%)', ascending=False)

# Affichage des métriques globales
st.subheader('Performance globale du CAC 40')
avg_performance = performance_df['Variation (%)'].mean()
best_stock = performance_df.iloc[0]
worst_stock = performance_df.iloc[-1]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Performance moyenne", f"{avg_performance:.2f}%", delta=f"{avg_performance:.2f}%")
with col2:
    st.metric("Meilleure performance", 
              f"{best_stock['Société']} ({best_stock['Ticker']})", 
              f"{best_stock['Variation (%)']:.2f}%")
with col3:
    st.metric("Pire performance", 
              f"{worst_stock['Société']} ({worst_stock['Ticker']})", 
              f"{worst_stock['Variation (%)']:.2f}%")

# Tableau des performances
st.subheader('Détail par action')

# Fonction pour colorer les valeurs
def color_perf(val):
    color = 'green' if val >= 0 else 'red'
    return f'color: {color}'

styled_df = performance_df.style.format({
    'Prix': '{:.2f} €',
    'Variation (%)': '{:.2f}%'
}).applymap(color_perf, subset=['Variation (%)'])

st.dataframe(styled_df, height=800, use_container_width=True)

# Graphique des performances
st.subheader('Visualisation des performances')
fig = px.bar(performance_df, 
             x='Société', 
             y='Variation (%)',
             color='Variation (%)',
             color_continuous_scale=['red', 'green'],
             labels={'Variation (%)': 'Variation (%)'},
             height=600)

fig.update_layout(xaxis_title='Société',
                 yaxis_title='Variation (%)',
                 coloraxis_showscale=False)

st.plotly_chart(fig, use_container_width=True)

# Téléchargement des données
csv = performance_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Télécharger les données",
    data=csv,
    file_name=f"cac40_performance_{datetime.now().strftime('%Y%m%d')}.csv",
    mime='text/csv'
)

# Pied de page
st.markdown("---")
st.caption("""
**Disclaimer:** Les données financières sont fournies à titre informatif uniquement. 
Les performances passées ne préjugent pas des performances futures.
""")
