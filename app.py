import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration pour éviter les conflits d'imports
try:
    import pandas_ta as ta
except ImportError:
    st.error("Erreur d'installation de pandas_ta. Vérifiez que la version 0.3.14b0 est installée.")
    st.stop()

try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
except ImportError:
    st.error("Erreur d'installation de backtesting. Vérifiez que la version 0.3.3 est installée.")
    st.stop()

# Configuration de la page
st.set_page_config(
    page_title="Backtesting Trading App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS
st.markdown("""
<style>
    .main {max-width: 1200px;}
    .stButton>button {background-color: #4CAF50; color: white;}
    .stAlert {padding: 20px; border-radius: 5px;}
    .st-bb {background-color: #f0f2f6;}
    .st-at {background-color: #f0f2f6;}
    .css-18e3th9 {padding: 2rem 1rem;}
</style>
""", unsafe_allow_html=True)

# Titre et description
st.title('📈 Application de Backtesting Trading')
st.markdown("""
Cette application permet de tester des stratégies de trading basées sur des indicateurs techniques.
""")

# Sidebar avec paramètres
with st.sidebar:
    st.header('⚙️ Paramètres')
    ticker = st.text_input('Symbole boursier (ex: AAPL)', 'AAPL').upper()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Date de début', pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input('Date de fin', pd.to_datetime('2023-12-31'))
    
    st.markdown("---")
    st.header('📊 Indicateurs techniques')
    
    sma_short = st.slider('Moyenne mobile courte', 5, 50, 20)
    sma_long = st.slider('Moyenne mobile longue', 50, 200, 50)
    rsi_period = st.slider('Période RSI', 5, 30, 14)
    rsi_overbought = st.slider('Seuil RSI surachat', 50, 90, 70)
    rsi_oversold = st.slider('Seuil RSI survente', 10, 50, 30)
    
    st.markdown("---")
    st.header('💰 Paramètres trading')
    commission = st.number_input('Commission (%)', min_value=0.0, max_value=5.0, value=0.1, step=0.05) / 100

# Fonction pour charger les données
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            st.error("Aucune donnée trouvée pour ce symbole.")
            return None
        return data
    except Exception as e:
        st.error(f"Erreur de téléchargement: {str(e)}")
        return None

# Chargement des données
data = load_data(ticker, start_date, end_date)
if data is None:
    st.stop()

# Calcul des indicateurs techniques
def calculate_indicators(df):
    # Utilisation de pandas_ta pour les indicateurs
    df.ta.sma(length=sma_short, append=True, col_names=('SMA_short',))
    df.ta.sma(length=sma_long, append=True, col_names=('SMA_long',))
    df.ta.rsi(length=rsi_period, append=True, col_names=('RSI',))
    df.ta.macd(append=True, col_names=('MACD', 'MACD_signal', 'MACD_hist'))
    return df

data = calculate_indicators(data)

# Affichage des données
st.subheader('📋 Données du marché')
st.dataframe(data.tail().style.format("{:.2f}"))

# Graphiques
st.subheader('📈 Visualisation des indicateurs')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})

# Prix et SMA
ax1.plot(data.index, data['Close'], label='Prix', color='black', alpha=0.7)
ax1.plot(data.index, data['SMA_short'], label=f'SMA {sma_short}', color='blue')
ax1.plot(data.index, data['SMA_long'], label=f'SMA {sma_long}', color='red')
ax1.set_title('Prix et Moyennes Mobiles')
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

plt.tight_layout()
st.pyplot(fig)

# Stratégie de trading
class SMACrossRSIStrategy(Strategy):
    def init(self):
        self.sma_short = self.I(ta.sma, self.data.Close, sma_short)
        self.sma_long = self.I(ta.sma, self.data.Close, sma_long)
        self.rsi = self.I(ta.rsi, self.data.Close, rsi_period)
    
    def next(self):
        if (crossover(self.sma_short, self.sma_long) and (self.rsi < rsi_overbought):
            self.buy()
        elif (crossover(self.sma_long, self.sma_short)) and (self.rsi > rsi_oversold):
            self.sell()

# Backtesting
if st.button('🚀 Lancer le Backtest', type='primary'):
    st.info("Calcul en cours...")
    
    # Préparation des données
    data_bt = data.copy()
    data_bt.columns = [col.lower() for col in data_bt.columns]
    
    # Exécution du backtest
    bt = Backtest(data_bt, SMACrossRSIStrategy, commission=commission, cash=10000)
    results = bt.run()
    
    # Affichage des résultats
    st.success("Backtest terminé!")
    
    # Métriques principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Rendement (%)", f"{results['Return [%]']:.2f}")
    col2.metric("Ratio de Sharpe", f"{results['Sharpe Ratio']:.2f}")
    col3.metric("Max Drawdown (%)", f"{results['Max. Drawdown [%]']:.2f}")
    
    # Graphique des résultats
    st.pyplot(bt.plot())
    
    # Détails des trades
    st.subheader('📝 Historique des trades')
    st.dataframe(results['_trades'])

# Pied de page
st.markdown("---")
st.caption("""
⚠️ **Note:** Cette application est à but éducatif seulement. 
Les performances passées ne garantissent pas les résultats futurs.
Le trading comporte des risques de perte en capital.
""")
