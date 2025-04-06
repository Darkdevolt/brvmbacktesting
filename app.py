import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from datetime import datetime, timedelta

# Configuration de la page
st.set_page_config(
    page_title="Analyse CAC 40",
    layout="wide",
    page_icon="📈"
)

# Liste des composants du CAC 40
CAC40_TICKERS = {
    'AC.PA': 'Accor', 'AI.PA': 'Air Liquide', 'AIR.PA': 'Airbus',
    'MT.PA': 'ArcelorMittal', 'ATO.PA': 'Atos', 'CS.PA': 'AXA',
    'BNP.PA': 'BNP Paribas', 'EN.PA': 'Bouygues', 'CAP.PA': 'Capgemini',
    'CA.PA': 'Carrefour', 'ACA.PA': 'Crédit Agricole', 'BN.PA': 'Danone',
    'DSY.PA': 'Dassault Systèmes', 'ENGI.PA': 'Engie', 'EL.PA': 'EssilorLuxottica',
    'RMS.PA': 'Hermès', 'KER.PA': 'Kering', 'LR.PA': 'Legrand',
    'OR.PA': 'L\'Oréal', 'MC.PA': 'LVMH', 'ML.PA': 'Michelin',
    'ORA.PA': 'Orange', 'RI.PA': 'Pernod Ricard', 'PUB.PA': 'Publicis',
    'RNO.PA': 'Renault', 'SAF.PA': 'Safran', 'SGO.PA': 'Saint-Gobain',
    'SAN.PA': 'Sanofi', 'SU.PA': 'Schneider Electric', 'GLE.PA': 'Société Générale',
    'STLA.PA': 'Stellantis', 'STM.PA': 'STMicroelectronics', 'TEP.PA': 'Teleperformance',
    'HO.PA': 'Thales', 'TTE.PA': 'TotalEnergies', 'VIE.PA': 'Veolia',
    'DG.PA': 'Vinci', 'VIV.PA': 'Vivendi'
}

# Navigation
st.sidebar.title('Navigation')
app_mode = st.sidebar.radio("", ['Analyse Générale', 'Analyse Technique Avancée'])

def calculate_indicators(df, sma_short, sma_long, rsi_period):
    # Calcul des moyennes mobiles
    df['SMA_short'] = df['Close'].rolling(window=sma_short).mean()
    df['SMA_long'] = df['Close'].rolling(window=sma_long).mean()
    
    # Calcul du RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss if avg_loss != 0 else 1
    df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

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
    commission = st.sidebar.number_input('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
    initial_cash = st.sidebar.number_input('Capital initial (€)', min_value=1000, max_value=100000, value=10000)

    # Récupération des données
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
    data = calculate_indicators(data, sma_short, sma_long, rsi_period)

    # Stratégie de trading
class SMACrossRSIStrategy(Strategy):
    def init(self):
        self.sma_short = self.I(lambda x: x.rolling(sma_short).mean(), self.data.Close)
        self.sma_long = self.I(lambda x: x.rolling(sma_long).mean(), self.data.Close)
        self.rsi = self.I(lambda x: 100 - (100 / (1 + (x.diff().where(x.diff() > 0, 0).rolling(rsi_period).mean() / 
                              -x.diff().where(x.diff() < 0, 0).rolling(rsi_period).mean()))), self.data.Close)
        
        def next(self):
            if crossover(self.sma_short, self.sma_long) and (self.rsi < rsi_overbought):
                if not self.position:
                    self.buy()
            elif crossover(self.sma_long, self.sma_short) and (self.rsi > rsi_oversold):
                if self.position:
                    self.sell()

    # Backtesting
    st.subheader('Backtesting sur 10 ans')
    if st.button('Lancer le Backtest'):
        with st.spinner('Calcul en cours...'):
            try:
                # Préparation des données
                data_bt = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                
                # Exécution du backtest
                bt = Backtest(data_bt, SMACrossRSIStrategy, 
                            commission=commission, 
                            cash=initial_cash,
                            exclusive_orders=True)
                
                results = bt.run()
                
                # Affichage des résultats
                st.success("Backtest terminé !")
                
                # Métriques principales
                st.subheader('📊 Métriques Clés')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Return [%]", f"{results['Return [%]']:.2f}")
                col2.metric("Sharpe Ratio", f"{results['Sharpe Ratio']:.2f}")
                col3.metric("Max Drawdown [%]", f"{results['Max. Drawdown [%]']:.2f}")
                col4.metric("# Trades", results['# Trades'])
                
                # Calcul des métriques avancées
                trades = results['_trades'].dropna()
                if not trades.empty:
                    winning_trades = trades[trades['PnL'] > 0]
                    losing_trades = trades[trades['PnL'] < 0]
                    
                    win_rate = len(winning_trades) / len(trades) * 100
                    avg_win = winning_trades['PnL'].mean()
                    avg_loss = losing_trades['PnL'].mean()
                    profit_factor = -avg_win / avg_loss if avg_loss != 0 else np.inf
                else:
                    win_rate = avg_win = avg_loss = profit_factor = 0
                
                # Affichage des métriques avancées
                st.subheader('📈 Statistiques de Performance')
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Win Rate [%]", f"{win_rate:.2f}")
                col2.metric("Gain Moyen/Trade [%]", f"{avg_win:.2f}")
                col3.metric("Perte Moyenne/Trade [%]", f"{avg_loss:.2f}")
                col4.metric("Profit Factor", f"{profit_factor:.2f}" if profit_factor != np.inf else "∞")
                
                # Graphique des performances
                st.subheader('📉 Performance de la Stratégie')
                st.pyplot(bt.plot())
                
                # Détails des trades
                st.subheader('📝 Détail des Trades')
                st.dataframe(trades.style.format({
                    'EntryPrice': '{:.2f}',
                    'ExitPrice': '{:.2f}',
                    'PnL': '{:.2f}',
                    'ReturnPct': '{:.2f}%'
                }))
                
                # Téléchargement des résultats
                csv = trades.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les trades",
                    data=csv,
                    file_name=f"trades_{selected_ticker}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )
                
            except Exception as e:
                st.error(f"Erreur lors du backtest: {str(e)}")

    # Visualisation des indicateurs
    st.subheader('📊 Indicateurs Techniques')
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

if app_mode == 'Analyse Technique Avancée':
    technical_analysis()

# Pied de page
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Mise à jour:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
