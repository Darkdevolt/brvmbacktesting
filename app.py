import streamlit as st
import pandas as pd
import numpy as np
import ta  # Utilisation de la bibliothèque ta
import plotly.graph_objects as go

# Titre de l'application
st.title("Application de Backtesting")

# Sidebar pour les paramètres
st.sidebar.header("Paramètres de la stratégie")

# Paramètres RSI
rsi_period = st.sidebar.slider("RSI Period", 1, 50, 14)
rsi_overbought = st.sidebar.slider("RSI Overbought", 70, 100, 70)
rsi_oversold = st.sidebar.slider("RSI Oversold", 0, 30, 30)

# Paramètres des moyennes mobiles
ma_period = st.sidebar.slider("Moyenne Mobile Period", 1, 200, 50)

# Paramètres des Bandes de Bollinger
bb_period = st.sidebar.slider("Bollinger Bands Period", 1, 50, 20)
bb_std = st.sidebar.slider("Bollinger Bands Std Dev", 1, 3, 2)

# Paramètres de gestion de risque
capital = st.sidebar.number_input("Capital de base", min_value=1000, value=10000)
risk_per_trade = st.sidebar.number_input("Risk par trade (%)", min_value=0.1, max_value=10.0, value=1.0)
risk_reward = st.sidebar.number_input("Risk/Reward", min_value=1.0, max_value=5.0, value=2.0)

# Upload des données historiques
uploaded_file = st.file_uploader("Uploader votre fichier de données historiques", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    # Assurez-vous que les colonnes sont correctement nommées
    data.rename(columns={
        'Dernier': 'Close',
        'Ouv.': 'Open',
        'Plus Haut': 'High',
        'Plus Bas': 'Low',
        'Vol.': 'Volume',
        'Variation %': 'Change'
    }, inplace=True)
    
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)

    # Calcul des indicateurs techniques avec ta
    data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_period)
    data['MA'] = ta.trend.sma_indicator(data['Close'], window=ma_period)
    bb = ta.volatility.BollingerBands(data['Close'], window=bb_period, window_dev=bb_std)
    data['Upper Band'] = bb.bollinger_hband()
    data['Middle Band'] = bb.bollinger_mavg()
    data['Lower Band'] = bb.bollinger_lband()

    # Génération des signaux
    data['Signal'] = 0
    data['Signal'][(data['RSI'] < rsi_oversold) & (data['Close'] < data['Lower Band'])] = 1  # Achat
    data['Signal'][(data['RSI'] > rsi_overbought) & (data['Close'] > data['Upper Band'])] = -1  # Vente

    # Calcul des transactions
    data['Position'] = data['Signal'].diff()
    transactions = data[data['Position'] != 0]

    # Affichage des transactions
    st.write("Transactions possibles :")
    st.write(transactions)

    # Calcul de la rentabilité
    initial_capital = capital
    risk_per_trade_amount = initial_capital * (risk_per_trade / 100)
    profit_loss = []

    for i, row in transactions.iterrows():
        if row['Position'] == 1:  # Achat
            entry_price = row['Close']
            stop_loss = entry_price * (1 - risk_per_trade / 100)
            take_profit = entry_price * (1 + (risk_per_trade / 100) * risk_reward)
        elif row['Position'] == -1:  # Vente
            entry_price = row['Close']
            stop_loss = entry_price * (1 + risk_per_trade / 100)
            take_profit = entry_price * (1 - (risk_per_trade / 100) * risk_reward)

        # Simuler le résultat du trade
        if row['Position'] == 1:
            if data['Close'].shift(-1).iloc[0] >= take_profit:
                profit_loss.append(risk_per_trade_amount * risk_reward)
            elif data['Close'].shift(-1).iloc[0] <= stop_loss:
                profit_loss.append(-risk_per_trade_amount)
        elif row['Position'] == -1:
            if data['Close'].shift(-1).iloc[0] <= take_profit:
                profit_loss.append(risk_per_trade_amount * risk_reward)
            elif data['Close'].shift(-1).iloc[0] >= stop_loss:
                profit_loss.append(-risk_per_trade_amount)

    # Calcul de la rentabilité totale
    total_profit_loss = sum(profit_loss)
    final_capital = initial_capital + total_profit_loss
    st.write(f"Capital final : {final_capital:.2f}")
    st.write(f"Rentabilité : {(final_capital - initial_capital) / initial_capital * 100:.2f}%")

    # Value at Risk (VaR)
    var_95 = np.percentile(profit_loss, 5)
    st.write(f"Value at Risk (95%) : {var_95:.2f}")

    # Conclusion
    if final_capital > initial_capital:
        st.success("La stratégie est rentable.")
    else:
        st.error("La stratégie n'est pas rentable.")

    # Visualisation des données
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MA'], mode='lines', name='MA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Upper Band'], mode='lines', name='Upper Band'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Lower Band'], mode='lines', name='Lower Band'))
    fig.add_trace(go.Scatter(x=transactions.index, y=transactions['Close'], mode='markers', name='Transactions', marker=dict(color='red', size=10)))
    st.plotly_chart(fig)

else:
    st.write("Veuillez uploader un fichier CSV ou Excel contenant les données historiques.")