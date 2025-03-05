import streamlit as st
import pandas as pd
import numpy as np
import ta  # Pour les indicateurs techniques
import plotly.graph_objects as go

# Titre de l'application
st.title("📈 Application de Backtesting - Nestlé CI (BRVM)")

# Sidebar pour les paramètres
st.sidebar.header("🔧 Paramètres de la stratégie")

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
capital = st.sidebar.number_input("💰 Capital de départ", min_value=1000, value=10000)
risk_per_trade = st.sidebar.number_input("⚠️ Risque par trade (%)", min_value=0.1, max_value=10.0, value=1.0)
risk_reward = st.sidebar.number_input("🎯 Risk/Reward", min_value=1.0, max_value=5.0, value=2.0)

# Upload des données historiques
uploaded_file = st.file_uploader("📂 Uploader votre fichier de données historiques", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Chargement des données
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)

    # Vérifier et afficher les colonnes pour éviter les erreurs
    st.write("📌 Colonnes détectées :", data.columns.tolist())

    # Vérification et correction des colonnes
    rename_dict = {
        'Date': 'Date',
        'Dernier': 'Close',
        'Ouv.': 'Open',
        'Plus Haut': 'High',
        'Plus Bas': 'Low',
        'Vol.': 'Volume',
        'Variation %': 'Change'
    }
    
    # Renommer les colonnes automatiquement si elles existent
    data.rename(columns={col: rename_dict[col] for col in data.columns if col in rename_dict}, inplace=True)

    # Vérifier que les colonnes essentielles sont bien présentes
    required_columns = ['Date', 'Close', 'Open', 'High', 'Low', 'Volume', 'Change']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        st.error(f"❌ Erreur : Colonnes manquantes dans le fichier : {missing_columns}")
        st.stop()

    # Convertir "Date" en format datetime
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')

    # Convertir "Vol." (Volume) en nombre en supprimant les K et M
    def convert_volume(vol):
        if isinstance(vol, str):
            vol = vol.replace('K', 'e3').replace('M', 'e6')
            try:
                return float(eval(vol))  # Convertit "313K" en 313000
            except:
                return np.nan
        return vol

    data['Volume'] = data['Volume'].apply(convert_volume)

    # Convertir "Variation %" en float en supprimant le "%"
    data['Change'] = data['Change'].str.replace('%', '').str.replace(',', '.').astype(float)

    # Trier les données par date
    data.set_index('Date', inplace=True)
    data = data.sort_index()

    # Calcul des indicateurs techniques avec ta
    data['RSI'] = ta.momentum.rsi(data['Close'], window=rsi_period)
    data['MA'] = ta.trend.sma_indicator(data['Close'], window=ma_period)
    bb = ta.volatility.BollingerBands(data['Close'], window=bb_period, window_dev=bb_std)
    data['Upper Band'] = bb.bollinger_hband()
    data['Middle Band'] = bb.bollinger_mavg()
    data['Lower Band'] = bb.bollinger_lband()

    # Génération des signaux
    data['Signal'] = 0
    data.loc[(data['RSI'] < rsi_oversold) & (data['Close'] < data['Lower Band']), 'Signal'] = 1  # Achat
    data.loc[(data['RSI'] > rsi_overbought) & (data['Close'] > data['Upper Band']), 'Signal'] = -1  # Vente

    # Calcul des transactions
    data['Position'] = data['Signal'].diff()
    transactions = data[data['Position'] != 0]

    # Affichage des transactions
    st.write("📊 Transactions possibles :")
    st.write(transactions)

    # Affichage du graphique interactif avec Plotly
    fig = go.Figure()

    # Ajouter les chandeliers
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Prix'
    ))

    # Ajouter la moyenne mobile
    fig.add_trace(go.Scatter(
        x=data.index, y=data['MA'],
        mode='lines', name='Moyenne Mobile',
        line=dict(color='blue', width=1)
    ))

    # Ajouter les bandes de Bollinger
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Upper Band'],
        mode='lines', name='Bollinger Haut',
        line=dict(color='red', width=1, dash='dot')
    ))
    fig.add_trace(go.Scatter(
        x=data.index, y=data['Lower Band'],
        mode='lines', name='Bollinger Bas',
        line=dict(color='green', width=1, dash='dot')
    ))

    # Ajouter les signaux d'achat et de vente
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]

    fig.add_trace(go.Scatter(
        x=buy_signals.index, y=buy_signals['Close'],
        mode='markers', name='Achat',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ))

    fig.add_trace(go.Scatter(
        x=sell_signals.index, y=sell_signals['Close'],
        mode='markers', name='Vente',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ))

    # Mise en page du graphique
    fig.update_layout(title="📈 Graphique des prix avec indicateurs techniques", xaxis_title="Date", yaxis_title="Prix", xaxis_rangeslider_visible=False)
    
    # Affichage du graphique
    st.plotly_chart(fig)