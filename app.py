import streamlit as st
import pandas as pd
import numpy as np
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf # Ré-ajouté pour Yahoo Finance
import io

# --- Configuration de la Page ---
st.set_page_config(
    page_title="Backtesting App (Yahoo Finance)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Fonctions Auxiliaires ---

@st.cache_data # Mise en cache du chargement des données
def load_data_yf(ticker, start_date, end_date):
    """Récupère les données depuis Yahoo Finance."""
    try:
        # Utilise la date actuelle si end_date est dans le futur pour éviter les erreurs yfinance
        end_date_corrected = min(pd.to_datetime(end_date), pd.to_datetime('today'))
        start_date_corrected = pd.to_datetime(start_date)

        if start_date_corrected > end_date_corrected:
             st.error("La date de début ne peut pas être postérieure à la date de fin.")
             return pd.DataFrame()

        df = yf.download(ticker, start=start_date_corrected, end=end_date_corrected, progress=False)
        if df.empty:
            st.error(f"Impossible de télécharger les données pour {ticker}. Vérifiez le symbole ou les dates.")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df['Date'] = pd.to_datetime(df['Date'])
        # Assurer la présence des colonnes standard (parfois yfinance les nomme différemment en fonction de la locale)
        df.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Adj Close': 'Adj Close', 'Volume': 'Volume'}, inplace=True)
        st.success(f"Données pour {ticker} chargées depuis Yahoo Finance ({len(df)} lignes).")
        return df
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données depuis Yahoo Finance : {e}")
        return pd.DataFrame()

@st.cache_data
def load_data_csv(uploaded_file):
    """Charge les données depuis un fichier CSV téléversé."""
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            # --- Validation des Données ---
            required_cols = ['Date', 'Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Le CSV doit contenir les colonnes : {', '.join(required_cols)}. Vérifiez votre fichier.")
                return pd.DataFrame()

            # Tentative d'analyse de la colonne Date
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except Exception:
                st.warning("Tentative d'analyse de la colonne 'Date' avec des formats courants...")
                try:
                    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
                    if df['Date'].isna().any():
                         st.error("Certaines dates n'ont pas pu être analysées. Assurez un format cohérent (ex: YYYY-MM-DD).")
                         return pd.DataFrame()
                except Exception as e:
                     st.error(f"Échec de l'analyse de la colonne 'Date'. Vérifiez le format. Erreur : {e}")
                     return pd.DataFrame()

            df = df.sort_values(by='Date').reset_index(drop=True)
            # Assurer les types numériques pour OHLCV
            for col in ['Open', 'High', 'Low', 'Close']:
                 if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'Volume' in df.columns:
                 df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

            initial_rows = len(df)
            df.dropna(subset=['Date', 'Open', 'High', 'Low', 'Close'], inplace=True)
            dropped_rows = initial_rows - len(df)
            if dropped_rows > 0:
                st.warning(f"{dropped_rows} lignes supprimées en raison de données essentielles manquantes/invalides.")

            if df.empty:
                 st.error("Aucune donnée valide restante après nettoyage. Vérifiez le fichier CSV.")
                 return pd.DataFrame()
            st.success(f"Données chargées depuis le fichier CSV '{uploaded_file.name}' ({len(df)} lignes).")
            return df
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier CSV : {e}. Assurez-vous que c'est un CSV valide.")
            return pd.DataFrame()
    return pd.DataFrame()


def calculate_indicators(df, strategy, params):
    """Calcule les indicateurs techniques."""
    df_indicators = df.copy()
    required_length = 0 # Longueur minimale de données pour certains indicateurs

    try:
        if strategy == "MA Crossover":
            required_length = params["ma_long"]
            if len(df_indicators) < required_length:
                st.warning(f"Données insuffisantes ({len(df_indicators)} lignes) pour calculer la SMA {required_length}. Minimum requis : {required_length} lignes.")
                return pd.DataFrame()
            df_indicators[f'SMA_{params["ma_short"]}'] = ta.sma(df_indicators["Close"], length=params["ma_short"])
            df_indicators[f'SMA_{params["ma_long"]}'] = ta.sma(df_indicators["Close"], length=params["ma_long"])
        elif strategy == "RSI":
            required_length = params["rsi_period"]
            if len(df_indicators) < required_length:
                 st.warning(f"Données insuffisantes ({len(df_indicators)} lignes) pour calculer le RSI {required_length}. Minimum requis : {required_length} lignes.")
                 return pd.DataFrame()
            df_indicators['RSI'] = ta.rsi(df_indicators["Close"], length=params["rsi_period"])
        elif strategy == "MACD":
            required_length = params["macd_slow"] + params["macd_signal"] # Estimation approximative
            if len(df_indicators) < required_length:
                 st.warning(f"Données insuffisantes ({len(df_indicators)} lignes) pour calculer le MACD. Minimum requis estimé : {required_length} lignes.")
                 return pd.DataFrame()
            macd = ta.macd(df_indicators["Close"], fast=params["macd_fast"], slow=params["macd_slow"], signal=params["macd_signal"])
            if macd is not None and not macd.empty:
                 # Renommer les colonnes MACD pour inclure les paramètres pour plus de clarté
                 macd.columns = [f"{col}_{params['macd_fast']}_{params['macd_slow']}_{params['macd_signal']}" for col in macd.columns]
                 df_indicators = pd.concat([df_indicators, macd], axis=1)
            else:
                 st.warning("Impossible de calculer le MACD. Vérifiez la longueur des données et les paramètres.")
        elif strategy == "Bollinger Bands":
            required_length = params["bb_period"]
            if len(df_indicators) < required_length:
                 st.warning(f"Données insuffisantes ({len(df_indicators)} lignes) pour calculer les Bandes de Bollinger {required_length}. Minimum requis : {required_length} lignes.")
                 return pd.DataFrame()
            bbands = ta.bbands(df_indicators["Close"], length=params["bb_period"], std=params["bb_std_dev"])
            if bbands is not None and not bbands.empty:
                 # Renommer les colonnes BBands pour inclure les paramètres
                 bbands.columns = [f"{col}_{params['bb_period']}_{float(params['bb_std_dev'])}" for col in bbands.columns]
                 df_indicators = pd.concat([df_indicators, bbands], axis=1)
            else:
                 st.warning("Impossible de calculer les Bandes de Bollinger.")

        df_indicators.dropna(inplace=True)
        if df_indicators.empty:
            st.warning(f"Aucune donnée restante après calcul des indicateurs {strategy}. La période de données est peut-être trop courte.")

    except Exception as e:
        st.error(f"Erreur lors du calcul des indicateurs pour {strategy}: {e}")
        return pd.DataFrame()

    return df_indicators

def generate_signals(df, strategy, params):
    """Génère les signaux d'achat/vente."""
    signals = pd.DataFrame(index=df.index)
    signals['Price'] = df['Close']
    signals['Buy'] = np.nan
    signals['Sell'] = np.nan

    if df.empty:
        st.warning("Impossible de générer les signaux : Données d'entrée vides.")
        return signals

    required_data_cols = ['Date', 'Close']
    if not all (c in df.columns for c in required_data_cols):
        st.error("Impossible de générer les signaux : Colonnes requises (Date, Close) manquantes.")
        return signals

    # --- Logique des signaux par stratégie ---
    try:
        if strategy == "MA Crossover":
            short_ma_col = f'SMA_{params["ma_short"]}'
            long_ma_col = f'SMA_{params["ma_long"]}'
            if short_ma_col not in df.columns or long_ma_col not in df.columns:
                 st.warning(f"Colonnes MA ({short_ma_col}, {long_ma_col}) non trouvées pour la génération de signaux.")
                 return signals

            signals.loc[(df[short_ma_col] > df[long_ma_col]) & (df[short_ma_col].shift(1) <= df[long_ma_col].shift(1)), 'Buy'] = df['Close']
            signals.loc[(df[short_ma_col] < df[long_ma_col]) & (df[short_ma_col].shift(1) >= df[long_ma_col].shift(1)), 'Sell'] = df['Close']

        elif strategy == "RSI":
            if 'RSI' not in df.columns:
                 st.warning("Colonne RSI non trouvée.")
                 return signals
            signals.loc[(df['RSI'] < params["rsi_lower"]) & (df['RSI'].shift(1) >= params["rsi_lower"]), 'Buy'] = df['Close']
            signals.loc[(df['RSI'] > params["rsi_upper"]) & (df['RSI'].shift(1) <= params["rsi_upper"]), 'Sell'] = df['Close']

        elif strategy == "MACD":
            macd_line = f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            signal_line = f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            if macd_line not in df.columns or signal_line not in df.columns:
                  st.warning(f"Colonnes MACD ({macd_line}, {signal_line}) non trouvées.")
                  return signals

            signals.loc[(df[macd_line] > df[signal_line]) & (df[macd_line].shift(1) <= df[signal_line].shift(1)), 'Buy'] = df['Close']
            signals.loc[(df[macd_line] < df[signal_line]) & (df[macd_line].shift(1) >= df[signal_line].shift(1)), 'Sell'] = df['Close']

        elif strategy == "Bollinger Bands":
            lower_band = f'BBL_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            upper_band = f'BBU_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            if lower_band not in df.columns or upper_band not in df.columns:
                 st.warning(f"Colonnes BBands ({lower_band}, {upper_band}) non trouvées.")
                 return signals

            signals.loc[(df['Close'] < df[lower_band]) & (df['Close'].shift(1) >= df[lower_band].shift(1)), 'Buy'] = df['Close']
            signals.loc[(df['Close'] > df[upper_band]) & (df['Close'].shift(1) <= df[upper_band].shift(1)), 'Sell'] = df['Close']

    except Exception as e:
         st.error(f"Erreur lors de la génération des signaux pour {strategy}: {e}")
         return pd.DataFrame()

    # --- Simulation de Backtest Simplifiée ---
    signals['Position'] = 0 # 0 = cash, 1 = en position
    signals['Portfolio_Value'] = np.nan
    signals['Return'] = 0.0

    initial_capital = 10000.0
    current_capital = initial_capital
    shares = 0
    signal_indices = signals.index

    for i in range(len(signals)):
        idx = signal_indices[i]
        buy_signal = not pd.isna(signals.loc[idx, 'Buy'])
        sell_signal = not pd.isna(signals.loc[idx, 'Sell'])
        current_position = signals.loc[signal_indices[i-1], 'Position'] if i > 0 else 0
        current_price = signals.loc[idx, 'Price']

        action_taken = False

        # Logique de Vente
        if (sell_signal and current_position == 1) or (i == len(signals) - 1 and current_position == 1):
            if shares > 0 and current_price > 0: # Vendre uniquement si on a des actions et que le prix est > 0
                current_capital += shares * current_price
                shares = 0
                signals.loc[idx, 'Position'] = 0
                action_taken = True
            elif current_position == 1:
                signals.loc[idx, 'Position'] = 0 # Marquer comme vendu même si shares était 0
                action_taken = True

        # Logique d'Achat
        if buy_signal and current_position == 0 and not action_taken:
            if current_capital > 0 and current_price > 0:
                shares_to_buy = current_capital / current_price
                current_capital = 0
                shares = shares_to_buy
                signals.loc[idx, 'Position'] = 1
                action_taken = True
            elif current_position == 0:
                 signals.loc[idx, 'Position'] = 1 # Marquer l'intention d'achat
                 action_taken = True

        # Mise à jour Position & Valeur Portefeuille
        if not action_taken:
             signals.loc[idx, 'Position'] = current_position

        current_holding_value = shares * current_price
        signals.loc[idx, 'Portfolio_Value'] = current_capital + current_holding_value

        # Calcul du retour journalier
        if i > 0 and signals.loc[signal_indices[i-1], 'Portfolio_Value'] != 0:
            signals.loc[idx, 'Return'] = (signals.loc[idx, 'Portfolio_Value'] / signals.loc[signal_indices[i-1], 'Portfolio_Value']) - 1
        elif i==0:
            signals.loc[idx, 'Portfolio_Value'] = initial_capital

    if not signals.empty:
         if pd.isna(signals['Portfolio_Value'].iloc[0]):
              signals['Portfolio_Value'].iloc[0] = initial_capital
         signals['Portfolio_Value'].ffill(inplace=True)

    return signals


def plot_results(df, signals, strategy, params, symbol_or_filename):
    """Crée un graphique Plotly interactif."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.7, 0.3])

    if df.empty:
        st.warning("Impossible de tracer les résultats : Données vides.")
        return fig

    # --- Graphique Chandelier ---
    fig.add_trace(go.Candlestick(x=df['Date'],
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='Prix'), row=1, col=1)

    # --- Ajout des Indicateurs Spécifiques ---
    try:
        if strategy == "MA Crossover":
            short_ma_col = f'SMA_{params["ma_short"]}'
            long_ma_col = f'SMA_{params["ma_long"]}'
            if short_ma_col in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[short_ma_col], mode='lines', name=f'SMA {params["ma_short"]}', line=dict(color='orange')), row=1, col=1)
            if long_ma_col in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[long_ma_col], mode='lines', name=f'SMA {params["ma_long"]}', line=dict(color='purple')), row=1, col=1)

        elif strategy == "Bollinger Bands":
            lower_band = f'BBL_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            upper_band = f'BBU_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            middle_band = f'BBM_{params["bb_period"]}_{float(params["bb_std_dev"])}'
            if upper_band in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', name='Bande Sup', line=dict(color='rgba(173,204,255,0.5)')), row=1, col=1)
            if lower_band in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', name='Bande Inf', line=dict(color='rgba(173,204,255,0.5)')), row=1, col=1)
            if middle_band in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[middle_band], mode='lines', name='Bande Milieu', line=dict(color='rgba(173,204,255,0.5)', dash='dash')), row=1, col=1)
            if upper_band in df.columns and lower_band in df.columns:
                  fig.add_trace(go.Scatter(x=df['Date'], y=df[upper_band], mode='lines', line=dict(width=0), showlegend=False), row=1, col=1)
                  fig.add_trace(go.Scatter(x=df['Date'], y=df[lower_band], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(173,204,255,0.2)', showlegend=False), row=1, col=1)

        # --- Ajout Signaux Achat/Vente ---
        buy_signals = signals[signals['Buy'].notna()]
        sell_signals = signals[signals['Sell'].notna()]
        if not buy_signals.empty:
            buy_dates = df.loc[buy_signals.index, 'Date']
            fig.add_trace(go.Scatter(x=buy_dates, y=buy_signals['Buy'], mode='markers', name='Signal Achat', marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        if not sell_signals.empty:
            sell_dates = df.loc[sell_signals.index, 'Date']
            fig.add_trace(go.Scatter(x=sell_dates, y=sell_signals['Sell'], mode='markers', name='Signal Vente', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

        # --- Ajout Indicateur 2ème Ligne ---
        if strategy == "RSI":
            if 'RSI' in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], mode='lines', name='RSI'), row=2, col=1)
                fig.add_hline(y=params["rsi_upper"], line_dash="dash", line_color="red", row=2, col=1, annotation_text=f'Surachat ({params["rsi_upper"]})', annotation_position="bottom right")
                fig.add_hline(y=params["rsi_lower"], line_dash="dash", line_color="green", row=2, col=1, annotation_text=f'Survente ({params["rsi_lower"]})', annotation_position="bottom right")
                fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)
        elif strategy == "MACD":
            macd_line = f'MACD_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            signal_line = f'MACDs_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            hist_line = f'MACDh_{params["macd_fast"]}_{params["macd_slow"]}_{params["macd_signal"]}'
            if macd_line in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[macd_line], mode='lines', name='MACD', line=dict(color='blue')), row=2, col=1)
            if signal_line in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df[signal_line], mode='lines', name='Ligne Signal', line=dict(color='orange')), row=2, col=1)
            if hist_line in df.columns:
                 colors = ['green' if val >= 0 else 'red' for val in df[hist_line]]
                 fig.add_trace(go.Bar(x=df['Date'], y=df[hist_line], name='Histogramme', marker_color=colors), row=2, col=1)
            fig.update_yaxes(title_text="MACD", row=2, col=1)
            fig.add_hline(y=0, line_dash="dash", line_color="grey", row=2, col=1)

    except Exception as e:
        st.error(f"Erreur lors de l'ajout des indicateurs/signaux au graphique: {e}")

    # --- Mise en Page ---
    fig.update_layout(
        title=f'Résultats Backtest pour "{symbol_or_filename}" ({strategy})',
        xaxis_title="Date",
        yaxis_title="Prix",
        xaxis_rangeslider_visible=False,
        height=600,
        legend_title="Légende",
        hovermode='x unified'
    )
    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=1, col=1)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, row=2, col=1)

    return fig

def calculate_performance(signals, initial_capital):
    """Calcule les métriques de performance de base."""
    if signals.empty or 'Portfolio_Value' not in signals.columns or signals['Portfolio_Value'].isna().all():
        st.warning("Calcul de performance impossible : les données de portefeuille sont vides ou invalides.")
        return {"Rendement Total (%)": 0, "Valeur Finale Portefeuille": f"${initial_capital:,.2f}"}

    first_valid_index = signals['Portfolio_Value'].first_valid_index()
    if first_valid_index is None:
         st.warning("Calcul de performance impossible : aucune valeur de portefeuille valide trouvée.")
         return {"Rendement Total (%)": 0, "Valeur Finale Portefeuille": f"${initial_capital:,.2f}"}

    effective_initial_capital = signals.loc[first_valid_index, 'Portfolio_Value']
    final_value = signals['Portfolio_Value'].iloc[-1]

    if pd.isna(effective_initial_capital) or effective_initial_capital == 0:
         total_return = 0
    else:
         total_return = ((final_value / effective_initial_capital) - 1) * 100

    metrics = {
        "Capital Initial": f"${initial_capital:,.2f}",
        "Valeur Finale Portefeuille": f"${final_value:,.2f}",
        "Rendement Total (%)": f"{total_return:.2f}%",
    }
    return metrics

# --- Barre Latérale Streamlit ---
st.sidebar.header("⚙️ Configuration")

# Sélection Source de Données
data_source = st.sidebar.radio(
    "Sélectionner la Source des Données",
    ("Yahoo Finance", "Téléverser CSV"),
    index=0 # Yahoo Finance par défaut
    )

uploaded_file = None
ticker = None
stock_display_name = None

if data_source == "Yahoo Finance":
    ticker = st.sidebar.text_input("Symbole Ticker", value="AAPL") # Défaut AAPL
else:
    uploaded_file = st.sidebar.file_uploader(
        "Téléverser Fichier CSV",
        type=['csv'],
        help="CSV avec colonnes: Date, Open, High, Low, Close (Volume optionnel). Format Date : YYYY-MM-DD ou MM/DD/YYYY."
    )

# Dates (Utilisé pour Yahoo ou pour filtrer CSV)
st.sidebar.subheader("Plage de Dates")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Date Début", pd.to_datetime("2020-01-01"))
with col2:
    # Utilisation de la date actuelle comme défaut pour la fin
    end_date = st.date_input("Date Fin", pd.to_datetime("today"))

# Sélection Stratégie
strategy = st.sidebar.selectbox(
    "Sélectionner Stratégie",
    ("MA Crossover", "RSI", "MACD", "Bollinger Bands")
)

# Paramètres Stratégie
params = {}
st.sidebar.subheader(f"Paramètres {strategy}")

if strategy == "MA Crossover":
    params["ma_short"] = st.sidebar.slider("Période MM Courte", 5, 100, 20)
    params["ma_long"] = st.sidebar.slider("Période MM Longue", 20, 200, 50)
    if params["ma_short"] >= params["ma_long"]: st.sidebar.warning("MM Courte doit être < MM Longue.")
elif strategy == "RSI":
    params["rsi_period"] = st.sidebar.slider("Période RSI", 7, 50, 14)
    params["rsi_upper"] = st.sidebar.slider("Seuil Surachat RSI", 50, 90, 70)
    params["rsi_lower"] = st.sidebar.slider("Seuil Survente RSI", 10, 50, 30)
    if params["rsi_lower"] >= params["rsi_upper"]: st.sidebar.warning("Seuil Inférieur RSI doit être < Seuil Supérieur.")
elif strategy == "MACD":
    params["macd_fast"] = st.sidebar.slider("Période Rapide MACD", 5, 50, 12)
    params["macd_slow"] = st.sidebar.slider("Période Lente MACD", 10, 100, 26)
    params["macd_signal"] = st.sidebar.slider("Période Signal MACD", 5, 50, 9)
    if params["macd_fast"] >= params["macd_slow"]: st.sidebar.warning("Période Rapide MACD doit être < Période Lente.")
elif strategy == "Bollinger Bands":
    params["bb_period"] = st.sidebar.slider("Période BB", 5, 100, 20)
    params["bb_std_dev"] = st.sidebar.slider("Écart-Type BB", 1.0, 4.0, 2.0, step=0.1)


# Bouton Backtest
run_button = st.sidebar.button("🚀 Lancer le Backtest")

# --- Zone Principale ---
st.title("📊 Outil de Backtesting Actions")

if run_button:
    # --- 1. Charger Données ---
    df = pd.DataFrame()
    base_df = pd.DataFrame() # Pour garder les données avant filtrage date

    if data_source == "Yahoo Finance":
        if ticker:
            base_df = load_data_yf(ticker, start_date, end_date)
            stock_display_name = ticker
        else:
            st.warning("Veuillez entrer un symbole Ticker.")
    else: # Téléverser CSV
        if uploaded_file:
            base_df = load_data_csv(uploaded_file)
            if not base_df.empty:
                stock_display_name = uploaded_file.name
                 # Filtrer le CSV par date ici si nécessaire (redondant avec yfinance mais ok)
                try:
                     mask = (base_df['Date'] >= pd.to_datetime(start_date)) & (base_df['Date'] <= pd.to_datetime(end_date))
                     df = base_df.loc[mask].copy()
                     if df.empty:
                          st.warning(f"Aucune donnée dans le fichier CSV pour la plage de dates sélectionnée : {start_date.strftime('%Y-%m-%d')} à {end_date.strftime('%Y-%m-%d')}.")
                     else:
                          st.info(f"Filtrage CSV par date appliqué ({len(df)} lignes).")
                except Exception as e:
                     st.error(f"Erreur lors du filtrage CSV par date : {e}")
                     df = pd.DataFrame() # Ne pas continuer si erreur de filtrage
            else:
                 st.error("Échec du chargement ou traitement du fichier CSV.")

        else:
            st.warning("Veuillez téléverser un fichier CSV.")

    # Si on utilise Yahoo, les dates sont déjà filtrées par l'API. Si CSV, on a filtré ci-dessus.
    if data_source == "Yahoo Finance":
        df = base_df # Utiliser les données de yfinance directement

    # --- Continuer si les données sont chargées et filtrées ---
    if not df.empty and stock_display_name:
        st.subheader(f"Résultats Backtest pour : {stock_display_name}")
        st.markdown(f"**Stratégie:** {strategy} | **Période:** {df['Date'].min().strftime('%Y-%m-%d')} à {df['Date'].max().strftime('%Y-%m-%d')}") # Afficher la période réelle des données utilisées

        # --- 2. Calculer Indicateurs ---
        df_processed = calculate_indicators(df.copy(), strategy, params)

        if not df_processed.empty:
            # --- 3. Générer Signaux & Lancer Backtest ---
            signals_df = generate_signals(df_processed.copy(), strategy, params)

            if not signals_df.empty and not signals_df['Portfolio_Value'].isna().all():
                # --- 4. Calculer Performance ---
                initial_capital = 10000.0
                performance_metrics = calculate_performance(signals_df, initial_capital)

                st.subheader("Métriques de Performance")
                perf_cols = st.columns(len(performance_metrics))
                for idx, (key, value) in enumerate(performance_metrics.items()):
                    perf_cols[idx].metric(label=key, value=str(value))

                # --- 5. Tracer Résultats ---
                st.subheader("Graphique Interactif")
                fig = plot_results(df_processed, signals_df, strategy, params, stock_display_name)
                st.plotly_chart(fig, use_container_width=True)

                # --- 6. Afficher Données (Optionnel) ---
                with st.expander("Voir Données et Signaux"):
                    st.subheader("Données Initiales (Après Filtrage Date)")
                    st.dataframe(df.head())
                    st.subheader("Données avec Indicateurs")
                    st.dataframe(df_processed.head())
                    st.subheader("Signaux et Portefeuille")
                    st.dataframe(signals_df)

            elif signals_df.empty:
                 st.warning("Aucun signal généré. Vérifiez les données et paramètres.")
                 st.write("Données avec Indicateurs (5 premières lignes):")
                 st.dataframe(df_processed.head())
            else:
                st.warning("La simulation de backtest n'a pas produit de valeurs de portefeuille valides.")
                st.dataframe(signals_df.head())

        # else: # df_processed était vide - message d'erreur déjà affiché dans calculate_indicators

    elif run_button and not stock_display_name:
         # Cas où l'on a cliqué Run mais sans source de données valide sélectionnée/fournie
         st.warning("Veuillez fournir une source de données valide (Ticker ou Fichier CSV) avant de lancer le backtest.")

else:
    st.info("Configurez les paramètres dans la barre latérale et cliquez sur 'Lancer le Backtest'.")
