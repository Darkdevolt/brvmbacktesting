import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Analyse CAC 40",
    layout="wide",
    page_icon="📈"
)

# Style CSS
st.markdown("""
<style>
    .main {max-width: 1200px;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .metric-box {border: 1px solid #ccc; border-radius: 5px; padding: 10px; margin: 5px 0;}
    .positive {color: #2ecc71;}
    .negative {color: #e74c3c;}
    .tabs .stTab {font-size: 16px; padding: 10px;}
</style>
""", unsafe_allow_html=True)

# Liste des composants du CAC 40
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

# Navigation dans la sidebar
st.sidebar.title('Navigation')
app_mode = st.sidebar.radio("",
    ['Analyse Générale', 'Analyse Technique Avancée'])

# Fonction pour l'analyse générale
def general_analysis():
    st.title('📊 Analyse Générale du CAC 40')
    
    # Paramètres
    col1, col2 = st.columns(2)
    with col1:
        period = st.selectbox('Période', ['1j', '1sem', '1mo', '3mo', '6mo', '1an', 'YTD'], index=5)
    with col2:
        benchmark = st.selectbox('Comparer avec', ['^FCHI', '^GSPC (S&P 500)', '^IXIC (NASDAQ)'], index=0)
    
    # Récupération des données
    @st.cache_data(ttl=3600)
    def get_stock_data(tickers, period):
        end_date = datetime.now()
        periods = {
            '1j': timedelta(days=1),
            '1sem': timedelta(weeks=1),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1an': timedelta(days=365),
            'YTD': datetime(end_date.year, 1, 1)
        }
        start_date = end_date - periods[period] if period != 'YTD' else periods[period]
        
        data = yf.download(list(tickers.keys()), start=start_date, end=end_date, group_by='ticker')
        return data
    
    try:
        data = get_stock_data(CAC40_TICKERS, period)
        if data.empty:
            st.error("Erreur lors de la récupération des données.")
            return
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return
    
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
    
    # Métriques globales
    st.subheader('Performance Globale')
    avg_performance = performance_df['Variation (%)'].mean()
    best_stock = performance_df.iloc[0]
    worst_stock = performance_df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Performance moyenne", f"{avg_performance:.2f}%")
    col2.metric("Meilleure performance", f"{best_stock['Société']}", f"{best_stock['Variation (%)']:.2f}%")
    col3.metric("Pire performance", f"{worst_stock['Société']}", f"{worst_stock['Variation (%)']:.2f}%")
    
    # Visualisation
    st.subheader('Détail des Performances')
    fig = px.bar(performance_df, 
                 x='Société', 
                 y='Variation (%)',
                 color='Variation (%)',
                 color_continuous_scale=['red', 'green'],
                 height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Tableau détaillé
    st.dataframe(performance_df.style.format({
        'Prix': '{:.2f} €',
        'Variation (%)': '{:.2f}%'
    }), height=600)

# Fonction pour l'analyse technique
def technical_analysis():
    st.title('📈 Analyse Technique Avancée')
    
    # Sélection de l'action
    selected_ticker = st.selectbox('Sélectionnez une action', list(CAC40_TICKERS.keys()), 
                                 format_func=lambda x: f"{x} - {CAC40_TICKERS[x]}")
    
    # Paramètres techniques
    st.sidebar.header('Paramètres Techniques')
    sma_short = st.sidebar.slider('Moyenne Mobile Courte', 5, 50, 20)
    sma_long = st.sidebar.slider('Moyenne Mobile Longue', 50, 200, 50)
    rsi_period = st.sidebar.slider('Période RSI', 5, 30, 14)
    rsi_overbought = st.sidebar.slider('Seuil RSI Surachat', 50, 90, 70)
    rsi_oversold = st.sidebar.slider('Seuil RSI Survendu', 10, 50, 30)
    
    # Récupération des données historiques (10 ans)
    @st.cache_data(ttl=3600)
    def get_historical_data(ticker):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        data = yf.download(ticker, start=start_date, end=end_date)
        return data
    
    try:
        data = get_historical_data(selected_ticker)
        if data.empty:
            st.error("Données non disponibles pour cette action.")
            return
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return
    
    # Calcul des indicateurs
    data['SMA_short'] = SMAIndicator(data['Close'], window=sma_short).sma_indicator()
    data['SMA_long'] = SMAIndicator(data['Close'], window=sma_long).sma_indicator()
    data['RSI'] = RSIIndicator(data['Close'], window=rsi_period).rsi()
    
    # Stratégie de trading
    class SMACrossRSIStrategy(Strategy):
        def init(self):
            self.sma_short = self.I(SMAIndicator, self.data.Close, window=sma_short)
            self.sma_long = self.I(SMAIndicator, self.data.Close, window=sma_long)
            self.rsi = self.I(RSIIndicator, self.data.Close, window=rsi_period)
        
        def next(self):
            if (crossover(self.sma_short, self.sma_long)) and (self.rsi < rsi_overbought):
                self.buy()
            elif (crossover(self.sma_long, self.sma_short)) and (self.rsi > rsi_oversold):
                self.sell()
    
    # Backtesting
    st.subheader('Backtesting sur 10 ans')
    if st.button('Lancer le Backtest'):
        with st.spinner('Calcul en cours...'):
            # Préparation des données
            data_bt = data.copy()
            data_bt.columns = [col.lower() for col in data_bt.columns]
            
            # Exécution du backtest
            bt = Backtest(data_bt, SMACrossRSIStrategy, commission=.002, cash=10000)
            results = bt.run()
            
            # Affichage des résultats
            st.success("Backtest terminé !")
            
            # Métriques clés
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Return [%]", f"{results['Return [%]']:.2f}")
            col2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
            col3.metric("Win Rate [%]", f"{results['Win Rate [%]']:.2f}")
            col4.metric("# Trades", results['# Trades'])
            
            # Graphique des performances
            st.subheader('Performance de la Stratégie')
            st.pyplot(bt.plot())
            
            # Détails des trades
            st.subheader('Détail des Trades')
            st.dataframe(results['_trades'])
            
            # Statistiques complètes
            st.subheader('Statistiques Complètes')
            stats_df = pd.DataFrame(results).drop(['_trades', '_equity_curve', '_strategy'], axis=1)
            st.dataframe(stats_df)
    
    # Visualisation des indicateurs
    st.subheader('Indicateurs Techniques')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Prix et SMA
    ax1.plot(data.index, data['Close'], label='Prix', color='black')
    ax1.plot(data.index, data['SMA_short'], label=f'SMA {sma_short}', color='blue')
    ax1.plot(data.index, data['SMA_long'], label=f'SMA {sma_long}', color='red')
    ax1.set_title(f'{CAC40_TICKERS[selected_ticker]} - Prix et Moyennes Mobiles')
    ax1.legend()
    ax1.grid(True)
    
    # RSI
    ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
    ax2.axhline(rsi_overbought, color='red', linestyle='--')
    ax2.axhline(rsi_oversold, color='green', linestyle='--')
    ax2.set_title('RSI')
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True)
    
    st.pyplot(fig)

# Navigation
if app_mode == 'Analyse Générale':
    general_analysis()
elif app_mode == 'Analyse Technique Avancée':
    technical_analysis()

# Pied de page
st.sidebar.markdown("---")
st.sidebar.markdown("""
**À propos:**
- Données: Yahoo Finance
- Mise à jour: {}
""".format(datetime.now().strftime("%d/%m/%Y %H:%M")))
