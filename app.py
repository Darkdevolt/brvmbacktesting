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

# Fonction de backtesting corrigée (indentation fixée)
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

    return data  # <-- Indentation correcte ici

# Fonction pour afficher les résultats (indentation corrigée)
def display_results(data, montant_investi):
    total_trades = data[data['Signal'] != 0].shape[0]
    winning_trades = data[data['Trade_Result'] > 0].shape[0]
    losing_trades = data[data['Trade_Result'] < 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    total_profit = data['Trade_Result'].sum()
    resultat_final = data['Capital'].iloc[-1]

    st.write(f"**Total des Trades :** {total_trades}")
    st.write(f"**Trades Gagnants :** {winning_trades}")
    st.write(f"**Trades Perdants :** {losing_trades}")
    st.write(f"**Taux de Réussite :** {win_rate:.2f}%")
    st.write(f"**Profit Total :** {total_profit:.2f}%")
    st.write(f"**Montant Investi :** {montant_investi:.2f} CFA")
    st.write(f"**Résultat Final :** {resultat_final:.2f} CFA")

# Fonction pour afficher les graphiques (indentation corrigée)
def plot_results(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Prix de Clôture'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Short'], name='Moyenne Mobile Courte'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Long'], name='Moyenne Mobile Longue'), row=1, col=1)
    
    buy_signals = data[data['Position'] == 'Buy']
    sell_signals = data[data['Position'] == 'Sell']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Achat', marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Vente', marker=dict(color='red', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=2, col=1)
    fig.update_layout(height=800, title_text="Résultats du Backtesting")
    st.plotly_chart(fig)

# Fonction pour afficher les trades (indentation corrigée)
def display_trades_table(data):
    trades = data[data['Signal'] != 0][['Signal', 'Position', 'close', 'Trade_Result']]
    trades['Type'] = trades['Position'].apply(lambda x: 'Achat' if x == 'Buy' else 'Vente')
    trades['Résultat (%)'] = trades['Trade_Result']
    trades['Prix'] = trades['close']
    trades_table = trades[['Type', 'Prix', 'Résultat (%)']]
    trades_table.index.name = 'Date'
    
    st.write("**Détails des Trades :**")
    st.dataframe(trades_table.style.format({'Prix': '{:.2f}', 'Résultat (%)': '{:.2f}%'}))

# Fonction pour l'évolution du capital (indentation corrigée)
def plot_capital_evolution(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Capital'], name='Capital', mode='lines'))
    fig.update_layout(title="Évolution du Capital", xaxis_title="Date", yaxis_title="Capital (CFA)")
    st.plotly_chart(fig)

# Interface Streamlit
st.title("Backtesting BRVM - Version Finale")
st.sidebar.header("Configuration")

uploaded_file = st.sidebar.file_uploader("Importer CSV", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    data = data.rename(columns=lambda x: x.strip().lower())
    
    short_window = st.sidebar.slider("Moyenne Courte", 5, 50, 20)
    long_window = st.sidebar.slider("Moyenne Longue", 50, 200, 50)
    stop_loss = st.sidebar.number_input("Stop-Loss (%)", 1.0, 10.0, 2.0)
    take_profit = st.sidebar.number_input("Take-Profit (%)", 1.0, 20.0, 4.0)
    capital = st.sidebar.number_input("Capital Initial (CFA)", 10000, 1000000, 100000)

    data = calculate_indicators(data, short_window, long_window)
    data = backtest_strategy(data, stop_loss, take_profit, capital)
    
    display_results(data, capital)
    plot_results(data)
    display_trades_table(data)
    plot_capital_evolution(data)
else:
    st.info("Veuillez uploader un fichier CSV pour commencer.")
