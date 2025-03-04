import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD
from ta.volatility import BollingerBands
from scipy.stats import norm

# Configuration de la page
st.set_page_config(page_title="BRVM Backtest", layout="wide")
plt.style.use('ggplot')

def calculate_risk_metrics(returns):
    """Calcule les métriques de risque"""
    var_95 = np.percentile(returns, 5)
    max_drawdown = (returns.cumsum().expanding().max() - returns.cumsum()).max()
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    return var_95, max_drawdown, sharpe_ratio

def backtest_strategy(data, initial_capital, stop_loss_pct, rsi_oversold, rsi_overbought):
    """Exécute le backtesting avec la stratégie"""
    capital = initial_capital
    position = 0
    portfolio_values = []
    trades = []
    stop_loss_price = None
    in_position = False
    
    for i in range(1, len(data)):
        current_price = data['Close'].iloc[i]
        prev_price = data['Close'].iloc[i-1]
        
        # Conditions d'entrée
        buy_condition = (
            (data['SMA_20'].iloc[i] > data['SMA_50'].iloc[i]) and
            (data['RSI'].iloc[i] < rsi_oversold) and
            (current_price < data['BB_Lower'].iloc[i])
        )
        
        # Conditions de sortie
        sell_condition = (
            (data['SMA_20'].iloc[i] < data['SMA_50'].iloc[i]) or
            (data['RSI'].iloc[i] > rsi_overbought) or
            (current_price > data['BB_Upper'].iloc[i])
        )
        
        # Gestion des positions
        if buy_condition and not in_position:
            position = capital // current_price
            if position > 0:
                capital -= position * current_price
                entry_price = current_price
                stop_loss_price = entry_price * (1 - stop_loss_pct)
                trades.append(('Achat', data.index[i], current_price, position))
                in_position = True
        
        elif in_position and (sell_condition or current_price < stop_loss_price):
            capital += position * current_price
            trades.append(('Vente', data.index[i], current_price, position))
            position = 0
            in_position = False
            stop_loss_price = None
        
        portfolio_values.append(capital + position * current_price)
    
    return pd.Series(portfolio_values, index=data.index[1:]), trades

def main():
    st.title('📈 Backtest Stratégie BRVM')
    st.markdown("""
    **Application de trading quantitatif pour la Bourse Régionale des Valeurs Mobilières (BRVM)**
    - Stratégie basée sur SMA + RSI + Bollinger Bands
    - Gestion du risque avec Stop-Loss et Value at Risk
    - Optimisation des paramètres
    """)
    
    # Upload de données
    with st.expander("📤 Importer les données historiques"):
        uploaded_file = st.file_uploader("Déposer un fichier CSV", type="csv")
        if uploaded_file:
            data = pd.read_csv(uploaded_file, parse_dates=['Date'], index_col='Date')
            data.sort_index(inplace=True)
            
            # Calcul des indicateurs
            data['SMA_20'] = SMAIndicator(data['Close'], 20).sma_indicator()
            data['SMA_50'] = SMAIndicator(data['Close'], 50).sma_indicator()
            data['RSI'] = RSIIndicator(data['Close'], 14).rsi()
            
            # Bollinger Bands
            bb = BollingerBands(data['Close'], 20, 2)
            data['BB_Upper'] = bb.bollinger_hband()
            data['BB_Lower'] = bb.bollinger_lband()
            
            # Aperçu des données
            st.write("**Aperçu des données:**")
            st.dataframe(data.tail(), use_container_width=True)
            
            # Graphique des prix
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Prix de clôture'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20'))
            fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], name='SMA 50'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Upper'], name='Bollinger Upper'))
            fig.add_trace(go.Scatter(x=data.index, y=data['BB_Lower'], name='Bollinger Lower'))
            fig.update_layout(title='Analyse Technique', height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Veuillez déposer un fichier CSV pour commencer l'analyse")
            return
    
    # Paramètres de trading
    with st.sidebar:
        st.header("⚙️ Paramètres de la stratégie")
        initial_capital = st.number_input("Capital initial (XOF)", 100000, 100000000, 1000000)
        stop_loss_pct = st.slider("Stop-Loss (%)", 1.0, 20.0, 5.0) / 100
        rsi_oversold = st.slider("RSI Survente", 0, 50, 30)
        rsi_overbought = st.slider("RSI Surchauffe", 50, 100, 70)
    
    # Exécution du backtest
    portfolio_values, trades = backtest_strategy(data, initial_capital, stop_loss_pct, rsi_oversold, rsi_overbought)
    returns = portfolio_values.pct_change().dropna()
    
    # Calcul des métriques
    final_value = portfolio_values[-1]
    total_return_pct = (final_value - initial_capital) / initial_capital * 100
    var_95, max_drawdown, sharpe = calculate_risk_metrics(returns)
    n_trades = len(trades) // 2
    
    # Affichage des résultats
    st.header("📊 Résultats du Backtesting")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Valeur Finale", f"{final_value:,.0f} XOF", f"{total_return_pct:.1f}%")
    col2.metric("Value at Risk (95%)", f"{var_95*100:.1f}%", "Perte quotidienne maximale")
    col3.metric("Max Drawdown", f"{max_drawdown*100:.1f}%", "Risque de baisse")
    col4.metric("Sharpe Ratio", f"{sharpe:.2f}", "Rendement ajusté au risque")
    
    # Graphiques d'analyse
    tab1, tab2, tab3 = st.tabs(["Performance", "Analyse des Risques", "Détails des Trades"])
    
    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=portfolio_values.index, y=portfolio_values, name='Portefeuille'))
        fig.add_trace(go.Scatter(x=data.index, y=data['Close']/data['Close'].iloc[0]*initial_capital, name='Buy & Hold'))
        fig.update_layout(title='Comparaison avec stratégie Buy & Hold',
                         yaxis_title='Valeur (XOF)',
                         hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Distribution des rendements
        st.subheader("Distribution des Rendements Quotidiens")
        hist_fig = go.Figure()
        hist_fig.add_trace(go.Histogram(x=returns*100, nbinsx=50, name='Rendements'))
        hist_fig.update_layout(xaxis_title='Rendement (%)', 
                             yaxis_title='Fréquence',
                             bargap=0.1)
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with tab2:
        # Courbe de drawdown
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown*100, fill='tozeroy'))
        fig.update_layout(title='Courbe de Drawdown',
                        yaxis_title='Drawdown (%)',
                        xaxis_title='Date')
        st.plotly_chart(fig, use_container_width=True)
        
        # Heatmap des performances
        st.subheader("Heatmap des Performances Mensuelles")
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=monthly_returns.values.reshape(-1, 12)*100,
            x=monthly_returns.index.strftime('%Y'),
            y=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            colorscale='RdYlGn'
        ))
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with tab3:
        # Analyse des trades
        if trades:
            trades_df = pd.DataFrame(trades, columns=['Type', 'Date', 'Prix', 'Quantité'])
            trades_df['Montant'] = trades_df['Prix'] * trades_df['Quantité']
            trades_df['Profit'] = np.where(trades_df['Type'] == 'Vente', 
                                         trades_df['Montant'].diff(), 0)
            st.dataframe(trades_df.style.format({
                'Prix': '{:.2f} XOF',
                'Montant': '{:,.0f} XOF',
                'Profit': '{:,.0f} XOF'
            }), use_container_width=True)
            
            # Graphique des profits
            profit_fig = go.Figure()
            profit_fig.add_trace(go.Bar(
                x=trades_df[trades_df['Type'] == 'Vente']['Date'],
                y=trades_df[trades_df['Type'] == 'Vente']['Profit'],
                name='Profit par trade'
            ))
            profit_fig.update_layout(title='Profit/Perte par Trade',
                                   yaxis_title='XOF')
            st.plotly_chart(profit_fig, use_container_width=True)
        else:
            st.warning("Aucun trade exécuté avec les paramètres actuels")

if __name__ == "__main__":
    main()
