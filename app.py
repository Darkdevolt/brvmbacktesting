import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# 1. Calcul des indicateurs techniques
def calculate_indicators(data, short_window=20, long_window=50):
    data['MA_Short'] = data['close'].rolling(window=short_window, min_periods=1).mean()
    data['MA_Long'] = data['close'].rolling(window=long_window, min_periods=1).mean()
    data['RSI'] = calculate_rsi(data['close'], window=14)
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
        if data['MA_Short'].iloc[i] > data['MA_Long'].iloc[i] and data['MA_Short'].iloc[i-1] <= data['MA_Long'].iloc[i-1]:
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
            
            if current_price <= stop_loss or current_price >= take_profit:
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
def plot_results(data):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Scatter(x=data.index, y=data['close'], name='Prix'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Short'], name='MM Courte'), row=1, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=data['MA_Long'], name='MM Longue'), row=1, col=1)
    
    buys = data[data['Position'] == 'Buy']
    sells = data[data['Position'] == 'Sell']
    fig.add_trace(go.Scatter(x=buys.index, y=buys['close'], mode='markers', 
                           marker=dict(color='green', size=10), name='Achat'), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells['close'], mode='markers', 
                           marker=dict(color='red', size=10), name='Vente'), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=data.index, y=data['RSI'], name='RSI'), row=2, col=1)
    
    fig.update_layout(height=800, title_text="Analyse Complète")
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

# 7. Tableau des transactions chronologique
def display_transactions_table(data):
    transactions = data[data['Signal'] != 0].copy()
    transactions = transactions.sort_index(ascending=True)  # Tri chronologique
    
    transactions['Type'] = transactions['Position'].apply(lambda x: 'Achat' if x == 'Buy' else 'Vente')
    transactions['Quantité'] = transactions['Actions_Detenues'].diff().abs().fillna(transactions['Actions_Detenues'])
    transactions['Résultat (CFA)'] = transactions['Trade_Result']/100 * transactions['close'] * transactions['Quantité']
    
    formatted_table = transactions[[
        'Type', 'close', 'Quantité', 'Résultat (CFA)', 'Trade_Result'
    ]].rename(columns={
        'close': 'Prix',
        'Trade_Result': 'Résultat (%)'
    })
    
    st.write("**Journal des Transactions Chronologique**")
    st.dataframe(
        formatted_table.style.format({
            'Prix': '{:.2f} CFA',
            'Quantité': '{:.0f}',
            'Résultat (%)': '{:.2f}%',
            'Résultat (CFA)': '{:,.2f} CFA'
        }),
        height=600,
        column_config={
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Prix": st.column_config.NumberColumn("Prix (CFA)", format="%.2f CFA"),
            "Quantité": st.column_config.NumberColumn("Quantité", format="%d"),
            "Résultat (%)": st.column_config.ProgressColumn(
                "Résultat (%)",
                format="%.2f%%",
                min_value=-100,
                max_value=100
            )
        }
    )

# 8. Interface Streamlit
st.title("📈 Backtesting BRVM Pro - Version Finale")
st.sidebar.header("⚙️ Configuration")

uploaded_file = st.sidebar.file_uploader("📤 Importer données historiques (CSV)", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
    data = data.rename(columns=lambda x: x.strip().lower())
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        short_window = st.slider("📏 Moyenne Courte", 5, 50, 20)
        stop_loss = st.number_input("⛔ Stop-Loss (%)", 1.0, 20.0, 2.0)
    with col2:
        long_window = st.slider("📐 Moyenne Longue", 50, 200, 50)
        take_profit = st.number_input("🎯 Take-Profit (%)", 1.0, 30.0, 4.0)
    
    capital = st.sidebar.number_input("💰 Capital Initial (CFA)", 10000, 10000000, 100000)

    data = calculate_indicators(data, short_window, long_window)
    data = backtest_strategy(data, stop_loss, take_profit, capital)
    
    display_results(data, capital)
    plot_results(data)
    plot_capital_evolution(data)
    display_transactions_table(data)  # Tableau ajouté ici

else:
    st.info("ℹ️ Veuillez uploader un fichier CSV pour commencer.")
