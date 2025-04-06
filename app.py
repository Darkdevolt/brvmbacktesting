def technical_analysis():
    st.title('📈 Analyse Technique Avancée')
    
    # Sélection de l'action
    selected_ticker = st.selectbox('Sélectionnez une action', list(CAC40_TICKERS.keys()), 
                                 format_func=lambda x: f"{x} - {CAC40_TICKERS[x]}")
    
    # Paramètres techniques
    st.sidebar.header('Paramètres Techniques')
    sma_short = st.sidebar.slider('Moyenne Mobile Courte', 5, 50, 20)
    sma_long = st.sidebar.slider('Moyenne Mobile Longue', 50, 200, 50)
    rsi_period = st.sidebar.slider('Période RSI', 5, 30, 14)
    rsi_overbought = st.sidebar.slider('Seuil RSI Surachat', 50, 90, 70)
    rsi_oversold = st.sidebar.slider('Seuil RSI Survendu', 10, 50, 30)
    commission = st.sidebar.number_input('Commission (%)', min_value=0.0, max_value=1.0, value=0.1, step=0.01) / 100
    initial_cash = st.sidebar.number_input('Capital initial (€)', min_value=1000, max_value=100000, value=10000)

    # Récupération des données
    @st.cache_data(ttl=3600)
    def get_historical_data(ticker):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)
        data = yf.download(ticker, start=start_date, end=end_date)
        return data

    try:
        data = get_historical_data(selected_ticker)
        if data.empty:
            st.error("Données non disponibles pour cette action.")
            return
    except Exception as e:
        st.error(f"Erreur: {str(e)}")
        return

    # Calcul des indicateurs
    data = calculate_indicators(data, sma_short, sma_long, rsi_period)

    # Stratégie de trading corrigée
    class SMACrossRSIStrategy(Strategy):
        def init(self):
            self.sma_short = self.I(lambda x: x.rolling(sma_short).mean(), self.data.Close)
            self.sma_long = self.I(lambda x: x.rolling(sma_long).mean(), self.data.Close)
            self.rsi = self.I(lambda x: 100 - (100 / (1 + (x.diff().where(x.diff() > 0, 0).rolling(rsi_period).mean() / -x.diff().where(x.diff() < 0, 0).rolling(rsi_period).mean())), self.data.Close)
                              
        
        def next(self):
            if (crossover(self.sma_short, self.sma_long)) and (self.rsi < rsi_overbought):
                self.buy()
            elif (crossover(self.sma_long, self.sma_short)) and (self.rsi > rsi_oversold):
                self.sell()

    # Backtesting corrigé
    st.subheader('Backtesting sur 10 ans')
    if st.button('Lancer le Backtest'):
        with st.spinner('Calcul en cours...'):
            # Préparation des données sans modifier les noms de colonnes
            data_bt = data.copy()
            
            # Exécution du backtest
            bt = Backtest(data_bt, SMACrossRSIStrategy, commission=commission, cash=initial_cash)
            results = bt.run()
            
            # Affichage des résultats
            st.success("Backtest terminé !")
            
            # Calcul des métriques avancées
            trades = results['_trades']
            winning_trades = trades[trades['PnL'] > 0]
            losing_trades = trades[trades['PnL'] < 0]
            
            # ... (reste du code inchangé)
