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
    # Lecture des données
    data = pd.read_excel(uploaded_file, sheet_name="in")

    # Nettoyage des colonnes
    data.columns = data.iloc[0].str.replace('"', '').str.strip()  # Supprimer les guillemets et espaces
    data = data[1:].reset_index(drop=True)  # Supprimer la ligne des titres en double
    
    # Renommer les colonnes pour standardiser
    column_mapping = {
        'Date': 'Date',
        'Ouv.': 'Open',
        'Plus Haut': 'High',
        'Plus Bas': 'Low',
        'Dernier': 'Close',
        'Vol.': 'Volume',
        'Variation %': 'Change'
    }
    data.rename(columns=column_mapping, inplace=True)

    # Conversion des types de données
    data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')
    for col in ['Open', 'High', 'Low', 'Close']:
        data[col] = data[col].str.replace(' ', '').astype(float)

    # Conversion des volumes (remplacement des "K" par "000")
    data['Volume'] = data['Volume'].str.replace('K', '000').str.replace(' ', '').astype(float)

    # Nettoyage et conversion des variations
    data['Change'] = data['Change'].str.replace('%', '').str.replace(',', '.').astype(float) / 100

    # Mettre la colonne Date en index
    data.set_index('Date', inplace=True)

    # Calcul des indicateurs techniques avec ta
    data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_period)
    data['MA'] = ta.trend.sma_indicator(data['Close'], window=ma_period)
    bb = ta.volatility.BollingerBands(data['Close'], window=bb_period, window_dev=bb_std)
    data['Upper Band'] = bb.bollinger_hband()
    data['Middle Band'] = bb.bollinger_mavg()
    data['Lower Band'] = bb.bollinger_lband()

    # Génération des signaux d'achat et de vente
    data['Signal'] = 0
    data.loc[(data['RSI'] < rsi_oversold) & (data['Close'] < data['Lower Band']), 'Signal'] = 1  # Achat
    data.loc[(data['RSI'] > rsi_overbought) & (data['Close'] > data['Upper Band']), 'Signal'] = -1  # Vente

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

    for i in range(len(transactions) - 1):
        row = transactions.iloc[i]
        next_row = transactions.iloc[i + 1]  # Éviter les erreurs d'index

        entry_price = row['Close']
        if row['Position'] == 1:  # Achat
            stop_loss = entry_price * (1 - risk_per_trade / 100)
            take_profit = entry_price * (1 + (risk_per_trade / 100) * risk_reward)
            if next_row['Close'] >= take_profit:
                profit_loss.append(risk_per_trade_amount * risk_reward)
            elif next_row['Close'] <= stop_loss:
                profit_loss.append(-risk_per_trade_amount)

        elif row['Position'] == -1:  # Vente
            stop_loss = entry_price * (1 + risk_per_trade / 100)
            take_profit = entry_price * (1 - (risk_per_trade / 100) * risk_reward)
            if next_row['Close'] <= take_profit:
                profit_loss.append(risk_per_trade_amount * risk_reward)
            elif next_row['Close'] >= stop_loss:
                profit_loss.append(-risk_per_trade_amount)

    # Calcul de la rentabilité totale
    total_profit_loss = sum(profit_loss)
    final_capital = initial_capital + total_profit_loss
    st.write(f"Capital final : {final_capital:.2f}")
    st.write(f"Rentabilité : {(final_capital - initial_capital) / initial_capital * 100:.2f}%")

    # Value at Risk (VaR)
    if len(profit_loss) > 0:
        var_95 = np.percentile(profit_loss, 5)
        st.write(f"Value at Risk (95%) : {var_95:.2f}")

    # Affichage des résultats
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