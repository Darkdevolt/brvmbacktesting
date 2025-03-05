def backtest_strategy(data, stop_loss_pct=2, take_profit_pct=4, montant_investi=100000.0):
    data = data.sort_index(ascending=True)  # Tri par date ascendante
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
        # Condition d'achat
        if (data['MA_Short'].iloc[i] > data['MA_Long'].iloc[i]) and (not premier_achat_effectue or actions_detenues > 0):
            if not premier_achat_effectue:
                # Premier achat
                ...
                premier_achat_effectue = True
            else:
                # Renforcement
                ...

        # Condition de vente
        if actions_detenues > 0:
            current_price = data['close'].iloc[i]
            stop_loss = prix_moyen_achat * (1 - stop_loss_pct / 100)
            take_profit = prix_moyen_achat * (1 + take_profit_pct / 100)
            if current_price <= stop_loss or current_price >= take_profit:
                ...
                premier_achat_effectue = False  # Reset après vente

        # Mise à jour du capital
        ...

    return data
