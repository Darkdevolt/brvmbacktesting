# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import matplotlib.pyplot as plt

st.title('Application de Backtesting Trading')

# Sidebar pour les paramètres
st.sidebar.header('Paramètres du Backtest')
ticker = st.sidebar.text_input('Symbole boursier', 'AAPL')
start_date = st.sidebar.date_input('Date de début', pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('Date de fin', pd.to_datetime('2023-12-31'))
