import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Fonction pour calculer les indicateurs techniques
def calculate_indicators(data, short_window=20, long_window=50):
    data['MA_Short'] = data['close'].rolling(window=short_window, min_periods=1).mean()
    data['MA_Long'] = data['close'].rolling(window=long_window, min_periods=1).mean()
    data['RSI'] = calculate_rsi(data['close'], window=14)
    return data

# Fonction pour calculer le RSI
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Fonction de backtesting corrigée
def backtest_strategy(data, stop_loss_pct=2, take_profit_pct=4, montant_investi=100000.0):
    data = data.sort_index(ascending=True)  # Tri chronologique strict
    data['Signal'] = 0
    data['Trade_Result'] = 0.0
    data['Position'] = None
    data['Capital'] = montant_investi
    data['Actions_Detenues'] = 0
    
    capital = montant_investi
    actions_detenues = 0
    prix_moyen_achat = 0.0
    premier_achat_effectue = False  # Contrôle du premier achat

    for i in range(1, len(data)):
        # Logique d'achat
        if (data['MA_Short'].iloc[i] > data['MA_Long'].iloc[i] and 
            data['MA_Short'].iloc[i-1] <= data['MA_Long'].iloc[i-1]):
            
            if not premier_achat_effectue:  # Premier achat obligatoire
                prix_achat = data['close'].iloc[i]
                actions_possibles = int(capital // prix_achat)
                
                if actions_possibles > 0:
                    data.at[data.index[i], 'Signal'] = 1
                    data.at[data.index[i], 'Position'] = 'Buy'
                    actions_detenues = actions_possibles
                    capital -= actions_possibles * prix_achat
                    prix_moyen_achat = prix_achat
                    premier_achat_effectue = True
                    
            elif capital > 0:  # Renforcement de position
                prix_achat = data['close'].iloc[i]
                actions_possibles = int(capital // prix_achat)
                
                if actions_possibles > 0:
                    data.at[data.index[i], 'Signal'] = 1
                    data.at[data.index[i], 'Position'] = 'Buy'
                    new_avg_price = (prix_moyen_achat * actions_detenues + prix_achat * actions_possibles) / (actions_detenues + actions_possibles)
                    capital -= actions_possibles * prix_achat
                    actions_detenues += actions_possibles
                    prix_moyen_achat = new_avg_price

        # Logique de vente
        if actions_detenues > 0:
            current_price = data['close'].iloc[i]
            stop_loss = prix_moyen_achat * (1 - stop_loss_pct/100)
            take_profit = prix_moyen_achat * (1 + take_profit_pct/100)
            
            if current_price <= stop_loss or current_price >= take_profit:
                data.at[data.index[i], 'Signal'] = -1
                data.at[data.index[i], 'Position'] = 'Sell'
                capital += actions_detenues * current_price
                profit_pct = ((current_price - prix_moyen_achat)/prix_moyen_achat) * 100
                data.at[data.index[i], 'Trade_Result'] = profit_pct
                actions_detenues = 0
                prix_moyen_achat = 0.0

        # Mise à jour des métriques
        data.at[data.index[i], 'Capital'] = capital + (actions_detenues * data['close'].iloc[i])
        data.at[data.index[i], 'Actions_Detenues'] = actions_detenues

    return data

# Fonctions d'affichage (identique à précédemment)
def display_results(data, montant_investi):
    # ... (identique à votre version précédente)

def plot_results(data):
    # ... (identique à votre version précédente)

def display_trades_table(data):
    # ... (identique à votre version précédente)

def plot_capital_evolution(data):
    # ... (identique à votre version précédente)

# Interface Streamlit 
st.title("Backtesting BRVM - Version Finale")
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Importer CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    data = data.rename(columns=lambda x: x.strip().lower())
    
    # Paramètres
    short_window = st.sidebar.slider("Moyenne Courte", 5, 50, 20)
    long_window = st.sidebar.slider("Moyenne Longue", 50, 200, 50)
    stop_loss = st.sidebar.number_input("Stop-Loss (%)", 1.0, 10.0, 2.0)
    take_profit = st.sidebar.number_input("Take-Profit (%)", 1.0, 20.0, 4.0)
    capital = st.sidebar.number_input("Capital Initial (CFA)", 10000, 1000000, 100000)

    # Execution
    data = calculate_indicators(data, short_window, long_window)
    data = backtest_strategy(data, stop_loss, take_profit, capital)
    
    # Affichage
    display_results(data, capital)
    plot_results(data)
    display_trades_table(data)
    plot_capital_evolution(data)
else:
    st.info("Veuillez uploader un fichier CSV pour commencer.")
