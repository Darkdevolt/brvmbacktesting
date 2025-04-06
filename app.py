import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration pour éviter les conflits de packages
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError:
    st.error("Erreur d'importation du module backtesting. Vérifiez les dépendances.")
    st.stop()

# Importer pandas_ta après backtesting pour éviter les conflits
try:
    import pandas_ta as ta
except ImportError:
    st.error("Erreur d'importation du module pandas_ta. Vérifiez les dépendances.")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="Backtesting Trading App",
    layout="wide"
)

# Titre de l'application
st.title('📈 Application de Backtesting Trading')

# Sidebar pour les paramètres
with st.sidebar:
    st.header('⚙️ Paramètres')
    ticker = st.text_input('Symbole (ex: AAPL)', 'AAPL').upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Début', pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input('Fin', pd.to_datetime('2023-12-31'))
    
    st.markdown("---")
    st.header('📊 Indicateurs')
    sma_short = st.slider('SMA courte', 10, 50, 20)
    sma_long = st.slider('SMA longue', 50, 200, 50)
    rsi_period = st.slider('Période RSI', 5, 30, 14)

# Chargement des données
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

data = load_data(ticker, start_date, end_date)

if data is None:
    st.error("Données non disponibles. Vérifiez le symbole.")
    st.stop()

# Calcul des indicateurs (version simplifiée)
def calculate_indicators(df):
    # Moyennes mobiles simples
    df['SMA_short'] = df['Close'].rolling(sma_short).mean()
    df['SMA_long'] = df['Close'].rolling(sma_long).mean()
    
    # RSI manuel
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

data = calculate_indicators(data)

# Affichage des données
st.subheader('Données du marché')
st.dataframe(data.tail())

# Stratégie de trading simplifiée
class SMACrossStrategy(Strategy):
    def init(self):
        self.sma_short = self.I(lambda x: x.rolling(sma_short).mean(), self.data.Close)
        self.sma_long = self.I(lambda x: x.rolling(sma_long).mean(), self.data.Close)
    
    def next(self):
        if crossover(self.sma_short, self.sma_long):
            self.buy()
        elif crossover(self.sma_long, self.sma_short):
            self.sell()

# Backtesting
if st.button('Lancer le Backtest'):
    st.info("Calcul en cours...")
    
    # Préparation des données
    data_bt = data.copy()
    data_bt.columns = [col.lower() for col in data_bt.columns]
    
    # Exécution du backtest
    bt = Backtest(data_bt, SMACrossStrategy, commission=.002)
    results = bt.run()
    
    # Affichage des résultats
    st.success("Backtest terminé!")
    st.write(results)
    
    # Graphique
    st.pyplot(bt.plot())
