import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import datetime

# 1. Calcul des indicateurs techniques avec options
def calculate_indicators(data, short_window=20, long_window=50, 
                        show_rsi=True, show_bollinger=True, 
                        bollinger_window=20, bollinger_std=2):
    # Moyennes mobiles
    data['MA_Short'] = data['close'].rolling(window=short_window, min_periods=1).mean()
    data['MA_Long'] = data['close'].rolling(window=long_window, min_periods=1).mean()
    
    # RSI
    if show_rsi:
        data['RSI'] = calculate_rsi(data['close'])
    
    # Bandes de Bollinger
    if show_bollinger:
        data['Bollinger_Mid'] = data['close'].rolling(window=bollinger_window).mean()
        data['Bollinger_Std'] = data['close'].rolling(window=bollinger_window).std()
        data['Bollinger_Upper'] = data['Bollinger_Mid'] + (data['Bollinger_Std'] * bollinger_std)
        data['Bollinger_Lower'] = data['Bollinger_Mid'] - (data['Bollinger_Std'] * bollinger_std)
    
    return data

# 2. Calcul du RSI
def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# 3. Stratégie de trading
def backtest_strategy(data, stop_loss_pct=2, take_profit_pct=4, montant_investi=100000.0):
    data = data.sort_index(ascending=True)
    data['Signal'] = 0
    data['Trade_Result'] = 0.0
    data['Position'] = None
    data['Capital'] = montant_investi
    data['Actions_Detenues'] = 0
    
    capital = montant_investi
    actions_detenues = 0
    prix_moyen_achat = 0.0
    premier_achat_effectue = False

    for i in range(1, len(data)):
        # Logique d'achat
        if (data['MA_Short'].iloc[i] > data['MA_Long'].iloc[i] and data['MA_Short'].iloc[i-1] <= data['MA_Long'].iloc[i-1]) or \
           (data['RSI'].iloc[i] < 30 and data['close'].iloc[i] < data['Bollinger_Lower'].iloc[i]):
            if not premier_achat_effectue:
                prix_achat = data['close'].iloc[i]
                actions_possibles = int(capital // prix_achat)
                if actions_possibles > 0:
                    data.at[data.index[i], 'Signal'] = 1
                    data.at[data.index[i], 'Position'] = 'Buy'
                    actions_detenues = actions_possibles
                    capital -= actions_possibles * prix_achat
                    prix_moyen_achat = prix_achat
                    premier_achat_effectue = True
            elif capital > 0:
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
            
            if current_price <= stop_loss or current_price >= take_profit or \
               (data['RSI'].iloc[i] > 70 and data['close'].iloc[i] > data['Bollinger_Upper'].iloc[i]):
                data.at[data.index[i], 'Signal'] = -1
                data.at[data.index[i], 'Position'] = 'Sell'
                capital += actions_detenues * current_price
                profit_pct = ((current_price - prix_moyen_achat)/prix_moyen_achat) * 100
                data.at[data.index[i], 'Trade_Result'] = profit_pct
                actions_detenues = 0
                prix_moyen_achat = 0.0

        data.at[data.index[i], 'Capital'] = capital + (actions_detenues * data['close'].iloc[i])
        data.at[data.index[i], 'Actions_Detenues'] = actions_detenues

    return data

# 4. Synthèse annuelle
def display_yearly_summary(data):
    data['Year'] = data.index.year
    yearly_data = []
    
    for year, group in data.groupby('Year'):
        if group.empty: continue
            
        start_capital = group['Capital'].iloc[0]
        end_capital = group['Capital'].iloc[-1]
        annual_return = ((end_capital - start_capital)/start_capital * 100) if start_capital !=0 else 0
        
        trades = group[group['Signal'] !=0]
        n_trades = len(trades)
        winning_trades = len(trades[trades['Trade_Result'] >0])
        win_rate = (winning_trades/n_trades *100) if n_trades>0 else 0
        
        peak = group['Capital'].cummax()
        drawdown = (peak - group['Capital'])/peak *100
        max_drawdown = drawdown.max()
        
        daily_returns = group['Capital'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252)*100
        
        yearly_data.append({
            'Année': year,
            'Rentabilité (%)': annual_return,
            'Trades': n_trades,
            'Taux Réussite (%)': win_rate,
            'Drawdown Max (%)': max_drawdown,
            'Volatilité (%)': volatility,
            'Capital Final': f"{end_capital:,.2f} CFA"
        })
    
    yearly_df = pd.DataFrame(yearly_data).set_index('Année')
    st.dataframe(yearly_df.style.format({
        'Rentabilité (%)': '{:.2f}%',
        'Taux Réussite (%)': '{:.2f}%',
        'Drawdown Max (%)': '{:.2f}%',
        'Volatilité (%)': '{:.2f}%'
    }))

# 5. Affichage des résultats
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
    st.write(f"**Montant Investi :** {montant_investi:,.2f} CFA")
    st.write(f"**Résultat Final :** {resultat_final:,.2f} CFA")
    display_yearly_summary(data)

# 6. Visualisations
def plot_results(data, show_rsi=True, show_bollinger=True):
    rows = 1
    if show_rsi: rows += 1
    
    fig = make_subplots(
        rows=rows, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[{"type": "scatter"}]] + [[{"type": "scatter"}]] if show_rsi else []
    )
    
    # Graphique principal
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Prix'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Short'], name='MM Courte'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Long'], name='MM Longue'), row=1, col=1)
    
    # Bandes de Bollinger
    if show_bollinger and 'Bollinger_Upper' in data:
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Bollinger_Upper'], 
            name='Bollinger Upper',
            line=dict(color='rgba(200,200,200,0.5)')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data['Bollinger_Lower'], 
            name='Bollinger Lower',
            line=dict(color='rgba(200,200,200,0.5)'),
            fill='tonexty',
            fillcolor='rgba(200,200,200,0.2)'
        ), row=1, col=1)
    
    # Signaux de trading
    buys = data[data['Position'] == 'Buy']
    sells = data[data['Position'] == 'Sell']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['close'], mode='markers', 
                           marker=dict(color='green', size=10), name='Achat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['close'], mode='markers', 
                           marker=dict(color='red', size=10), name='Vente'), row=1, col=1)
    
    # RSI
    if show_rsi and 'RSI' in data:
        fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=2, col=1)
        fig.add_hline(y=30, line=dict(color='red', dash='dash'), row=2, col=1)
        fig.add_hline(y=70, line=dict(color='green', dash='dash'), row=2, col=1)
    
    fig.update_layout(height=800, title_text="Analyse Technique", showlegend=True)
    st.plotly_chart(fig)

def plot_capital_evolution(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index, 
        y=data['Capital'], 
        name="Capital",
        line=dict(color='#00CC96', width=3)
    ))
    fig.update_layout(
        title="Évolution du Capital",
        xaxis_title="Date",
        yaxis_title="Montant (CFA)",
        template='plotly_white',
        hovermode="x unified"
    )
    st.plotly_chart(fig)

# 7. Journal des transactions
def display_transactions_table(data):
    transactions = data[data['Signal'] != 0].copy().sort_index(ascending=True)
    
    transactions['Type'] = transactions['Position'].apply(lambda x: 'Achat' if x == 'Buy' else 'Vente')
    transactions['Quantité'] = transactions['Actions_Detenues'].diff().abs().fillna(transactions['Actions_Detenues'])
    transactions['Résultat (CFA)'] = transactions['Trade_Result']/100 * transactions['close'] * transactions['Quantité']
    
    formatted_table = transactions[[
        'Type', 'close', 'Quantité', 'Résultat (CFA)', 'Trade_Result'
    ]].rename(columns={
        'close': 'Prix',
        'Trade_Result': 'Résultat (%)'
    })
    
    st.write("**Journal Complet des Transactions**")
    st.dataframe(
        formatted_table.style.format({
            'Prix': '{:.2f} CFA',
            'Quantité': '{:.0f}',
            'Résultat (%)': '{:.2f}%',
            'Résultat (CFA)': '{:,.2f} CFA'
        }),
        height=600,
        column_config={
            "Résultat (%)": st.column_config.ProgressColumn(
                "Performance",
                format="%.2f%%",
                min_value=-100,
                max_value=100,
                width="medium"
            )
        }
    )

# 8. Interface Streamlit
st.title("📈 Backtesting BRVM - Édition Professionnelle")

# Barre de navigation pour la période de backtesting
st.write("### Sélectionnez la période de backtesting")
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input("Date de début", datetime(2020, 1, 1))
with col2:
    end_date = st.date_input("Date de fin", datetime.today())

st.sidebar.header("⚙️ Configuration Générale")

# Upload des données
uploaded_file = st.sidebar.file_uploader("📤 Importer CSV Historique", type=["csv"])

# Paramètres des indicateurs
with st.sidebar.expander("📊 Options des Indicateurs"):
    col1, col2 = st.columns(2)
    with col1:
        show_rsi = st.checkbox("Afficher RSI", value=True)
        show_bollinger = st.checkbox("Afficher Bollinger", value=True)
    with col2:
        if show_bollinger:
            bollinger_window = st.slider("Fenêtre Bollinger", 10, 50, 20)
            bollinger_std = st.slider("Écart-type", 1.0, 3.0, 2.0)

# Paramètres de stratégie
with st.sidebar.expander("🎯 Paramètres de Stratégie"):
    col1, col2 = st.columns(2)
    with col1:
        short_window = st.slider("MM Courte", 5, 50, 20)
        stop_loss = st.number_input("Stop-Loss (%)", 1.0, 20.0, 2.0)
    with col2:
        long_window = st.slider("MM Longue", 50, 200, 50)
        take_profit = st.number_input("Take-Profit (%)", 1.0, 30.0, 4.0)
    
    capital = st.number_input("Capital Initial (CFA)", 10000, 10000000, 100000)

# Traitement principal
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    data = data.rename(columns=lambda x: x.strip().lower())
    
    # Filtrage des données selon la période sélectionnée
    data = data.loc[start_date:end_date]
    
    data = calculate_indicators(
        data,
        short_window=short_window,
        long_window=long_window,
        show_rsi=show_rsi,
        show_bollinger=show_bollinger,
        bollinger_window=bollinger_window if show_bollinger else 20,
        bollinger_std=bollinger_std if show_bollinger else 2
    )
    
    data = backtest_strategy(data, stop_loss, take_profit, capital)
    
    display_results(data, capital)
    plot_results(data, show_rsi, show_bollinger)
    plot_capital_evolution(data)
    display_transactions_table(data)

else:
    st.info("ℹ️ Veuillez uploader un fichier CSV pour démarrer l'analyse.")

st.sidebar.markdown("---")
st.sidebar.markdown("Développé par [Votre Nom] • Version 1.2.0")
