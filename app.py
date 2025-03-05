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

# Fonction pour simuler la stratégie (sans short selling)
def backtest_strategy(data, stop_loss_pct=2, take_profit_pct=4, montant_investi=100000.0):
    data['Signal'] = 0  # 1 pour acheter, -1 pour vendre, 0 pour rien faire
    data['Trade_Result'] = 0.0  # Résultat de chaque trade
    data['Position'] = None  # Position actuelle (None, 'Buy', 'Sell')
    data['Capital'] = montant_investi  # Capital initial
    data['Actions_Detenues'] = 0  # Nombre d'actions détenues
    capital = montant_investi  # Capital courant
    actions_detenues = 0  # Nombre d'actions détenues
    prix_moyen_achat = 0.0  # Prix moyen d'achat des actions détenues

    for i in range(1, len(data)):
        # Condition d'achat : Croisement de moyennes mobiles
        if data['MA_Short'].iloc[i] > data['MA_Long'].iloc[i] and data['MA_Short'].iloc[i-1] <= data['MA_Long'].iloc[i-1]:
            if actions_detenues == 0:
                # Premier achat
                data.at[data.index[i], 'Signal'] = 1
                prix_achat = data['close'].iloc[i]
                actions_detenues = capital // prix_achat  # Acheter autant d'actions que possible
                capital -= actions_detenues * prix_achat  # Mettre à jour le capital
                prix_moyen_achat = prix_achat  # Initialiser le prix moyen d'achat
                data.at[data.index[i], 'Position'] = 'Buy'
                data.at[data.index[i], 'Actions_Detenues'] = actions_detenues
            else:
                # Renforcer l'achat (acheter davantage)
                data.at[data.index[i], 'Signal'] = 1
                prix_achat = data['close'].iloc[i]
                nouvelles_actions = capital // prix_achat  # Acheter autant d'actions que possible
                capital -= nouvelles_actions * prix_achat  # Mettre à jour le capital
                # Mettre à jour le prix moyen d'achat
                prix_moyen_achat = (prix_moyen_achat * actions_detenues + prix_achat * nouvelles_actions) / (actions_detenues + nouvelles_actions)
                actions_detenues += nouvelles_actions  # Mettre à jour le nombre d'actions détenues
                data.at[data.index[i], 'Position'] = 'Buy'
                data.at[data.index[i], 'Actions_Detenues'] = actions_detenues

        # Conditions de vente : Stop-loss ou Take-profit
        if actions_detenues > 0:
            current_price = data['close'].iloc[i]
            stop_loss_price = prix_moyen_achat * (1 - stop_loss_pct / 100)
            take_profit_price = prix_moyen_achat * (1 + take_profit_pct / 100)

            if current_price <= stop_loss_price or current_price >= take_profit_price:
                data.at[data.index[i], 'Signal'] = -1
                capital += actions_detenues * current_price  # Vendre toutes les actions
                trade_result = (current_price - prix_moyen_achat) / prix_moyen_achat * 100
                data.at[data.index[i], 'Trade_Result'] = trade_result
                actions_detenues = 0  # Plus d'actions détenues
                prix_moyen_achat = 0.0  # Réinitialiser le prix moyen d'achat
                data.at[data.index[i], 'Position'] = 'Sell'
                data.at[data.index[i], 'Actions_Detenues'] = 0

        # Mettre à jour le capital dans le DataFrame
        data.at[data.index[i], 'Capital'] = capital + (actions_detenues * data['close'].iloc[i])

    return data

# Fonction pour afficher les résultats
def display_results(data, montant_investi):
    total_trades = data[data['Signal'] != 0].shape[0]
    winning_trades = data[data['Trade_Result'] > 0].shape[0]
    losing_trades = data[data['Trade_Result'] < 0].shape[0]
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    total_profit = data['Trade_Result'].sum()
    
    # Calcul du résultat final en fonction du montant investi en CFA
    resultat_final = data['Capital'].iloc[-1]  # Capital final

    st.write(f"**Total des Trades :** {total_trades}")
    st.write(f"**Trades Gagnants :** {winning_trades}")
    st.write(f"**Trades Perdants :** {losing_trades}")
    st.write(f"**Taux de Réussite :** {win_rate:.2f}%")
    st.write(f"**Profit Total :** {total_profit:.2f}%")
    st.write(f"**Montant Investi :** {montant_investi:.2f} CFA")
    st.write(f"**Résultat Final :** {resultat_final:.2f} CFA")

# Fonction pour afficher les graphiques
def plot_results(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])

    # Graphique des prix et des indicateurs
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Prix de Clôture'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Short'], name='Moyenne Mobile Courte'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Long'], name='Moyenne Mobile Longue'), row=1, col=1)

    # Graphique des trades
    buy_signals = data[data['Position'] == 'Buy']
    sell_signals = data[data['Position'] == 'Sell']
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['close'], mode='markers', name='Achat', marker=dict(color='green', size=10)), row=1, col=1)
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['close'], mode='markers', name='Vente', marker=dict(color='red', size=10)), row=1, col=1)

    # Graphique du RSI
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=2, col=1)

    fig.update_layout(height=800, title_text="Résultats du Backtesting")
    st.plotly_chart(fig)

# Fonction pour afficher les trades dans un tableau
def display_trades_table(data):
    # Filtrer les trades (achats et ventes)
    trades = data[data['Signal'] != 0][['Signal', 'Position', 'close', 'Trade_Result']]
    trades['Type'] = trades['Position'].apply(lambda x: 'Achat' if x == 'Buy' else 'Vente')
    trades['Résultat (%)'] = trades['Trade_Result']
    trades['Prix'] = trades['close']
    
    # Sélectionner les colonnes à afficher
    trades_table = trades[['Type', 'Prix', 'Résultat (%)']]
    trades_table.index.name = 'Date'
    
    # Afficher le tableau
    st.write("**Détails des Trades :**")
    st.dataframe(trades_table.style.format({
        'Prix': '{:.2f}',
        'Résultat (%)': '{:.2f}%'
    }))

# Fonction pour afficher l'évolution du capital
def plot_capital_evolution(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Capital'], name='Capital', mode='lines'))
    fig.update_layout(title="Évolution du Capital", xaxis_title="Date", yaxis_title="Capital (CFA)")
    st.plotly_chart(fig)

# Interface Streamlit
st.title("Backtesting de Stratégie de Trading sur la BRVM (Sans Short Selling)")
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
    montant_investi = st.sidebar.number_input("Montant Investi (CFA)", value=100000.0)  # Montant en CFA

    # Calcul des indicateurs et simulation de la stratégie
    data = calculate_indicators(data, short_window, long_window)
    data = backtest_strategy(data, stop_loss_pct, take_profit_pct, montant_investi)

    # Affichage des résultats
    st.write("**Résultats de la simulation :**")
    display_results(data, montant_investi)

    # Affichage des graphiques
    st.write("**Graphiques :**")
    plot_results(data)

    # Affichage des trades dans un tableau
    display_trades_table(data)

    # Affichage de l'évolution du capital
    st.write("**Évolution du Capital :**")
    plot_capital_evolution(data)
else:
    st.write("Veuillez télécharger un fichier CSV pour commencer.")
