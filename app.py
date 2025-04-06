import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Backtesting Trading App",
    page_icon="📈",
    layout="wide"
)

# Style CSS personnalisé
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
        font-weight: bold;
    }
    .stAlert {
        padding: 20px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Titre de l'application
st.title('📈 Application de Backtesting Trading')
st.markdown("""
Cette application permet de tester des stratégies de trading basées sur des indicateurs techniques.
""")

# Sidebar pour les paramètres
with st.sidebar:
    st.header('⚙️ Paramètres du Backtest')
    
    # Sélection du ticker et période
    ticker = st.text_input('Symbole boursier (ex: AAPL, MSFT)', 'AAPL').upper()
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Date de début', pd.to_datetime('2020-01-01'))
    with col2:
        end_date = st.date_input('Date de fin', pd.to_datetime('2023-12-31'))
    
    st.markdown("---")
    st.header('📊 Paramètres des indicateurs')
    
    # Paramètres des indicateurs
    sma_short = st.slider('Moyenne mobile courte (SMA)', 10, 100, 50)
    sma_long = st.slider('Moyenne mobile longue (SMA)', 50, 300, 200)
    rsi_period = st.slider('Période RSI', 5, 30, 14)
    rsi_overbought = st.slider('Seuil RSI surachat', 50, 90, 70)
    rsi_oversold = st.slider('Seuil RSI survente', 10, 50, 30)
    
    st.markdown("---")
    st.header('💰 Paramètres de trading')
    commission = st.number_input('Commission par trade (%)', min_value=0.0, max_value=5.0, value=0.1, step=0.05) / 100

# Fonction pour charger les données
@st.cache_data
def load_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end, progress=False)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données: {e}")
        return None

# Chargement des données
data = load_data(ticker, start_date, end_date)

if data is None:
    st.error("Impossible de charger les données. Vérifiez le symbole et la période.")
    st.stop()

# Calcul des indicateurs techniques
def calculate_indicators(df):
    # Moyennes mobiles
    df['SMA_short'] = ta.sma(df['Close'], length=sma_short)
    df['SMA_long'] = ta.sma(df['Close'], length=sma_long)
    
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=rsi_period)
    
    # MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    return df

data = calculate_indicators(data)

# Affichage des données
st.subheader('📋 Données du marché')
st.write(f"Données pour {ticker} du {start_date} au {end_date}")

# Sélection des colonnes à afficher
default_cols = ['Close', f'SMA_{sma_short}', f'SMA_{sma_long}', 'RSI', 'MACD_12_26_9']
selected_cols = st.multiselect(
    'Colonnes à afficher',
    options=data.columns,
    default=default_cols
)

st.dataframe(data[selected_cols].tail(10).style.format("{:.2f}"))

# Graphique des prix et indicateurs
st.subheader('📈 Graphique des prix et indicateurs')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

# Prix et moyennes mobiles
ax1.plot(data.index, data['Close'], label='Prix de clôture', color='black', alpha=0.7)
ax1.plot(data.index, data[f'SMA_{sma_short}'], label=f'SMA {sma_short}', color='blue')
ax1.plot(data.index, data[f'SMA_{sma_long}'], label=f'SMA {sma_long}', color='red')
ax1.set_title('Prix et Moyennes Mobiles')
ax1.legend()
ax1.grid(True)

# RSI
ax2.plot(data.index, data['RSI'], label='RSI', color='purple')
ax2.axhline(y=rsi_overbought, color='red', linestyle='--')
ax2.axhline(y=rsi_oversold, color='green', linestyle='--')
ax2.set_title('RSI')
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True)

# MACD
ax3.plot(data.index, data['MACD_12_26_9'], label='MACD', color='blue')
ax3.plot(data.index, data['MACDs_12_26_9'], label='Signal', color='orange')
ax3.bar(data.index, data['MACDh_12_26_9'], label='Histogramme', color=np.where(data['MACDh_12_26_9'] > 0, 'green', 'red'))
ax3.set_title('MACD')
ax3.legend()
ax3.grid(True)

plt.tight_layout()
st.pyplot(fig)

# Définition de la stratégie
class SMACrossWithRSIStrategy(Strategy):
    def init(self):
        close = self.data.Close
        self.sma_short = self.I(ta.sma, close, sma_short)
        self.sma_long = self.I(ta.sma, close, sma_long)
        self.rsi = self.I(ta.rsi, close, rsi_period)
        self.macd = self.I(ta.macd, close)
    
    def next(self):
        # Conditions d'achat
        if (crossover(self.sma_short, self.sma_long) and 
            (self.rsi < rsi_overbought) and 
            (self.macd > 0)):
            self.buy()
        
        # Conditions de vente
        elif (crossover(self.sma_long, self.sma_short) and 
              (self.rsi > rsi_oversold) and 
              (self.macd < 0)):
            self.sell()

# Section de backtesting
st.subheader('🔍 Backtesting de la stratégie')

if st.button('🚀 Lancer le Backtest', use_container_width=True):
    st.info("Calcul en cours... Cette opération peut prendre quelques instants.")
    
    # Préparation des données pour backtesting.py
    data_bt = data.copy()
    data_bt.columns = [col.lower() for col in data_bt.columns]
    
    # Exécution du backtest
    bt = Backtest(data_bt, SMACrossWithRSIStrategy, commission=commission, cash=10000)
    results = bt.run()
    
    # Affichage des résultats
    st.success("Backtest terminé !")
    
    # Métriques clés
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Return [%]", f"{results['Return [%]']:.2f}")
    col2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
    col3.metric("Max. Drawdown [%]", f"{results['Max. Drawdown [%]']:.2f}")
    col4.metric("# Trades", results['# Trades'])
    
    # Graphique des résultats
    st.subheader('📊 Performance de la stratégie')
    fig = bt.plot(resample=False)
    st.pyplot(fig)
    
    # Détails des trades
    st.subheader('📝 Détails des trades')
    st.dataframe(results['_trades'])
    
    # Statistiques complètes
    st.subheader('📊 Statistiques complètes')
    stats_df = pd.DataFrame(results).drop(['_trades', '_equity_curve', '_strategy'], axis=1)
    st.dataframe(stats_df)
    
    # Téléchargement des résultats
    csv = stats_df.to_csv().encode('utf-8')
    st.download_button(
        label="📥 Télécharger les résultats",
        data=csv,
        file_name=f"backtest_results_{ticker}.csv",
        mime='text/csv',
        use_container_width=True
    )

# Pied de page
st.markdown("---")
st.markdown("""
**📌 Remarques:**
- Cette application est à but éducatif seulement
- Les performances passées ne garantissent pas les résultats futurs
- Le trading comporte des risques de perte en capital
""")
