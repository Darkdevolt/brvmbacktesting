import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Fonctions calculate_indicators et calculate_rsi (inchangées)
# ... [Les mêmes fonctions que précédemment] ...

def backtest_strategy(data, stop_loss_pct=2, take_profit_pct=4, montant_investi=100000.0):
    # ... [Le même code que précédemment] ...

# Nouvelle fonction pour le tableau annuel
def display_yearly_summary(data):
    data['Year'] = data.index.year
    yearly_data = []
    
    for year, group in data.groupby('Year'):
        if len(group) == 0:
            continue
            
        start_capital = group['Capital'].iloc[0]
        end_capital = group['Capital'].iloc[-1]
        annual_return = ((end_capital - start_capital) / start_capital * 100) if start_capital != 0 else 0
        
        trades = group[group['Signal'] != 0]
        n_trades = len(trades)
        winning_trades = len(trades[trades['Trade_Result'] > 0])
        win_rate = (winning_trades / n_trades * 100) if n_trades > 0 else 0
        
        # Calcul du drawdown maximum
        peak = group['Capital'].cummax()
        drawdown = (peak - group['Capital']) / peak * 100
        max_drawdown = drawdown.max()
        
        # Volatilité annuelle
        daily_returns = group['Capital'].pct_change().dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualisée
        
        yearly_data.append({
            'Année': year,
            'Rentabilité (%)': round(annual_return, 2),
            'Trades': n_trades,
            'Taux de Réussite (%)': round(win_rate, 2),
            'Drawdown Max (%)': round(max_drawdown, 2),
            'Volatilité (%)': round(volatility, 2),
            'Capital Final (CFA)': f"{end_capital:,.2f}"
        })
    
    yearly_df = pd.DataFrame(yearly_data)
    yearly_df.set_index('Année', inplace=True)
    
    st.write("**Synthèse Annuelle**")
    st.dataframe(yearly_df.style.format({
        'Rentabilité (%)': '{:.2f}%',
        'Taux de Réussite (%)': '{:.2f}%',
        'Drawdown Max (%)': '{:.2f}%',
        'Volatilité (%)': '{:.2f}%'
    }))

# Modifications dans l'affichage principal
def display_results(data, montant_investi):
    # ... [Le même code que précédemment] ...
    
    # Ajouter l'appel au tableau annuel
    display_yearly_summary(data)

# Les autres fonctions (plot_results, display_trades_table, plot_capital_evolution) restent inchangées
# ... [Le même code que précédemment] ...
