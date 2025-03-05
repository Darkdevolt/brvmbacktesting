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

# Fonction pour simuler la stratégie
def backtest_strategy(data, stop_loss_pct=2, take_profit_pct=4):
    data['Signal'] = 0
    data['Trade_Result'] = 0.0
    data['Position'] = None
    in_position = False
    entry_price = 0.0

    for i in range(1, len(data)):
        if not in_position:
            # Condition d'entrée : Croisement de moyennes mobiles
            if data['MA_Short'].iloc[i] > data['MA_Long'].iloc[i] and data['MA_Short'].iloc[i-1] <= data['MA_Long'].iloc[i-1]:
                data.at[data.index[i], 'Signal'] = 1
                entry_price = data['close'].iloc[i]
                in_position = True
                data.at[data.index[i], 'Position'] = 'Buy'
        else:
            # Conditions de sortie : Stop-loss ou Take-profit
            current_price = data['close'].iloc[i]
            stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
            take_profit_price = entry_price * (1 + take_profit_pct / 100)

            if current_price <= stop_loss_price or current_price >= take_profit_price:
                data.at[data.index[i], 'Signal'] = -1
                trade_result = (current_price - entry_price) / entry_price * 100
                data.at[data.index[i], 'Trade_Result'] = trade_result
                in_position = False
                data.at[data.index[i], 'Position'] = 'Sell'

    return data

# Fonction pour afficher les résultats
def display_results(data):
    total_trades = data[data['Signal'] != 0].shape[0]
    winning_trades = data[data['Trade_Result'] > 0].shape[0]
    losing_trades = data[data['Trade_Result'] < 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    total_profit = data['Trade_Result'].sum()

    st.write(f"**Total Trades:** {total_trades}")
    st.write(f"**Winning Trades:** {winning_trades}")
    st.write(f"**Losing Trades:** {losing_trades}")
    st.write(f"**Win Rate:** {win_rate:.2f}%")
    st.write(f"**Total Profit:** {total_profit:.2f}%")

# Fonction pour afficher les graphiques
def plot_results(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Graphique des prix et des indicateurs
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Close Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Short'], name='MA Short'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Long'], name='MA Long'), row=1, col=1)

    # Graphique des trades
    buy_signals = data[data['Position'] == 'Buy']
    sell_signals = data[data['Position'] == 'Sell']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Buy', marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Sell', marker=dict(color='red', size=10)), row=1, col=1)

    # Graphique du RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=2, col=1)

    fig.update_layout(height=800, title_text="Backtesting Results")
    st.plotly_chart(fig)

# Interface Streamlit
st.title("Backtesting de Stratégie de Trading")
st.sidebar.header("Paramètres")

# Chargement des données
uploaded_file = st.sidebar.file_uploader("Téléchargez votre fichier CSV", type=["csv"])
if uploaded_file is not None:
    # Lire le fichier CSV
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    
    # Convertir la colonne 'Date' en format datetime avec le bon format
    data.index = pd.to_datetime(data.index, format='%m/%d/%y')
    
    # Normaliser les noms des colonnes (supprimer les espaces et convertir en minuscules)
    data.columns = data.columns.str.strip().str.lower()

    # Afficher les colonnes disponibles pour déboguer
    st.write("**Colonnes disponibles dans le fichier CSV :**")
    st.write(data.columns.tolist())

    # Renommer les colonnes si nécessaire
    data = data.rename(columns={
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    })

    # Paramètres personnalisables
    short_window = st.sidebar.number_input("Moyenne mobile courte (jours)", value=20)
    long_window = st.sidebar.number_input("Moyenne mobile longue (jours)", value=50)
    stop_loss_pct = st.sidebar.number_input("Stop-loss (%)", value=2.0)
    take_profit_pct = st.sidebar.number_input("Take-profit (%)", value=4.0)

    # Calcul des indicateurs et simulation de la stratégie
    data = calculate_indicators(data, short_window, long_window)
    data = backtest_strategy(data, stop_loss_pct, take_profit_pct)

    # Affichage des résultats
    st.write("**Résultats de la simulation :**")
    display_results(data)

    # Affichage des graphiques
    st.write("**Graphiques :**")
    plot_results(data)
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")
