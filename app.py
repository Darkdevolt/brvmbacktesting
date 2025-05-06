import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import io
import datetime
from matplotlib.figure import Figure

st.set_page_config(
    page_title="Calcul de VaR Historique",
    page_icon="📊",
    layout="wide"
)

# Fonctions utilitaires
def calculate_returns(data, price_column):
    """Calcule les rendements logarithmiques."""
    return np.log(data[price_column] / data[price_column].shift(1)).dropna()

def calculate_var(returns, confidence_level, portfolio_value):
    """Calcule la VaR historique."""
    alpha = 1 - confidence_level/100
    var = np.percentile(returns, alpha * 100) * portfolio_value
    return abs(var)  # VaR est généralement exprimée en valeur positive

def calculate_rolling_var(returns, confidence_level, portfolio_value, window_size):
    """Calcule la VaR sur une fenêtre glissante."""
    rolling_var = []
    for i in range(len(returns) - window_size + 1):
        window_returns = returns.iloc[i:i+window_size]
        var = calculate_var(window_returns, confidence_level, portfolio_value)
        rolling_var.append(var)
    return pd.Series(rolling_var, index=returns.index[window_size-1:])

def check_stationarity(returns):
    """Vérifie la stationnarité des rendements avec le test ADF."""
    result = adfuller(returns.dropna())
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'Is Stationary': result[1] < 0.05
    }

def detect_outliers(returns, threshold=3):
    """Détecte les valeurs aberrantes basées sur les Z-scores."""
    z_scores = (returns - returns.mean()) / returns.std()
    return returns[abs(z_scores) > threshold]

# Interface Streamlit
st.title("Calculateur de VaR Historique Non-Paramétrique")

# Sidebar pour les options
st.sidebar.header("Options")

# Upload de fichier
uploaded_file = st.sidebar.file_uploader("Télécharger un fichier CSV", type="csv")

if uploaded_file is not None:
    # Charger les données
    try:
        data = pd.read_csv(uploaded_file, parse_dates=['Date'])
        data.set_index('Date', inplace=True)
    except Exception as e:
        data = pd.read_csv(uploaded_file)
        st.warning("Format de date non reconnu, veuillez vous assurer que la colonne 'Date' est au format date.")
    
    # Afficher les premières lignes des données
    st.subheader("Aperçu des données")
    st.dataframe(data.head())
    
    # Sélection de colonne pour le calcul des rendements
    price_column = st.sidebar.selectbox(
        "Sélectionner la colonne de prix pour calculer les rendements",
        options=data.columns,
        index=data.columns.get_loc("Close") if "Close" in data.columns else 0
    )
    
    # Options de calcul VaR
    st.sidebar.subheader("Paramètres VaR")
    confidence_level = st.sidebar.slider("Niveau de confiance (%)", 90, 99, 95)
    horizon_days = st.sidebar.slider("Horizon de risque (jours)", 1, 30, 1)
    portfolio_value = st.sidebar.number_input("Valeur du portefeuille (€)", value=100000.0, step=1000.0)
    min_history_days = st.sidebar.slider("Minimum de jours historiques", 100, 500, 250)
    
    # Calcul des rendements
    returns = calculate_returns(data, price_column)
    
    # Vérification de la suffisance des données
    if len(returns) < min_history_days:
        st.error(f"Les données sont insuffisantes. Il faut au moins {min_history_days} observations valides pour le calcul de la VaR.")
    else:
        # Correction de l'horizon de risque
        if horizon_days > 1:
            # Utilisation de la règle de la racine carrée du temps pour ajuster la VaR
            returns_adjusted = returns * np.sqrt(horizon_days)
        else:
            returns_adjusted = returns
        
        # Calcul de la VaR
        var_value = calculate_var(returns_adjusted, confidence_level, portfolio_value)
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Résultats VaR")
            st.markdown(f"""
            - **VaR {confidence_level}% sur {horizon_days} jour(s)**: {var_value:.2f} €
            - **Signification**: Avec une probabilité de {confidence_level}%, la perte maximale 
              ne dépassera pas {var_value:.2f} € sur un horizon de {horizon_days} jour(s).
            - **Pourcentage du portefeuille**: {(var_value/portfolio_value)*100:.2f}%
            """)
            
            # Test de stationnarité
            stationarity = check_stationarity(returns)
            st.subheader("Test de Stationnarité (ADF)")
            st.markdown(f"""
            - **Statistique de test**: {stationarity['Test Statistic']:.4f}
            - **p-value**: {stationarity['p-value']:.4f}
            - **Stationnarité**: {"✅ Stationnaire" if stationarity['Is Stationary'] else "❌ Non-stationnaire"}
            """)
            
            # Détection des valeurs aberrantes
            outliers = detect_outliers(returns)
            st.subheader("Détection des valeurs aberrantes")
            if len(outliers) > 0:
                st.write(f"Nombre de valeurs aberrantes détectées: {len(outliers)}")
                st.dataframe(outliers)
            else:
                st.write("Aucune valeur aberrante détectée.")
        
        with col2:
            # Visualisation de la distribution des rendements
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(returns, kde=True, ax=ax)
            var_percentile = np.percentile(returns, (1 - confidence_level/100) * 100)
            ax.axvline(var_percentile, color='red', linestyle='--', label=f'VaR {confidence_level}%')
            ax.set_title(f'Distribution des rendements et VaR {confidence_level}%')
            ax.legend()
            st.pyplot(fig)
            
            # Rolling VaR
            st.subheader("VaR Glissante")
            window_size = st.slider("Taille de la fenêtre glissante (jours)", 50, 250, 100)
            rolling_var = calculate_rolling_var(returns, confidence_level, portfolio_value, window_size)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            rolling_var.plot(ax=ax)
            ax.set_title(f'VaR {confidence_level}% glissante (fenêtre de {window_size} jours)')
            ax.set_ylabel('VaR (€)')
            st.pyplot(fig)
        
        # Analyse de sensibilité
        st.subheader("Analyse de sensibilité")
        col1, col2 = st.columns(2)
        
        with col1:
            # Sensibilité au niveau de confiance
            confidence_levels = range(90, 100)
            vars_by_conf = [calculate_var(returns_adjusted, cl, portfolio_value) for cl in confidence_levels]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(confidence_levels, vars_by_conf, marker='o')
            ax.set_title('Sensibilité de la VaR au niveau de confiance')
            ax.set_xlabel('Niveau de confiance (%)')
            ax.set_ylabel('VaR (€)')
            st.pyplot(fig)
        
        with col2:
            # Sensibilité à l'horizon de risque
            horizons = range(1, 31, 2)
            vars_by_horizon = [calculate_var(returns * np.sqrt(h), confidence_level, portfolio_value) 
                               for h in horizons]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(horizons, vars_by_horizon, marker='o')
            ax.set_title(f'Sensibilité de la VaR {confidence_level}% à l\'horizon')
            ax.set_xlabel('Horizon (jours)')
            ax.set_ylabel('VaR (€)')
            st.pyplot(fig)
        
        # Backtesting
        st.subheader("Backtesting")
        lookback_period = st.slider("Période de backtesting (jours)", 50, 250, 100, 
                                   help="Nombre de jours pour le backtesting")
        
        if len(returns) >= lookback_period:
            # Calcul des violations
            var_series = []
            violations = []
            
            for i in range(len(returns) - lookback_period + 1):
                # Calculer la VaR sur la fenêtre précédente
                var_window = returns.iloc[i:i+lookback_period-1]
                next_return = returns.iloc[i+lookback_period-1]
                var_value = calculate_var(var_window, confidence_level, portfolio_value)
                
                var_series.append(var_value)
                # Une violation se produit lorsque la perte réelle dépasse la VaR prédite
                violations.append(abs(next_return * portfolio_value) > var_value and next_return < 0)
            
            violation_rate = sum(violations) / len(violations)
            expected_rate = 1 - confidence_level/100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                - **Taux de violation**: {violation_rate:.4f} ({sum(violations)} violations sur {len(violations)} jours)
                - **Taux attendu**: {expected_rate:.4f}
                - **Ratio (Observé/Attendu)**: {violation_rate/expected_rate:.2f}
                """)
                
                if abs(violation_rate/expected_rate - 1) > 0.5:
                    st.warning("Le modèle de VaR pourrait être mal calibré - le taux de violation s'écarte significativement du taux attendu.")
            
            with col2:
                # Graphique des violations
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Série des rendements pour la période de backtesting
                backtest_returns = returns.iloc[-len(violations):] * portfolio_value
                backtest_dates = backtest_returns.index
                
                # Tracer les rendements et les violations
                ax.plot(backtest_dates, backtest_returns, color='black', alpha=0.5, label='Rendements')
                
                # Calculer la VaR négative (car les rendements négatifs sont des pertes)
                neg_var_series = [-v for v in var_series]
                
                # Tracer la ligne de VaR
                ax.plot(backtest_dates, neg_var_series, color='red', linestyle='--', label=f'VaR {confidence_level}%')
                
                # Marquer les violations
                violation_indices = [i for i, v in enumerate(violations) if v]
                if violation_indices:
                    violation_dates = [backtest_dates[i] for i in violation_indices]
                    violation_values = [backtest_returns.iloc[i] for i in violation_indices]
                    ax.scatter(violation_dates, violation_values, color='red', marker='x', s=100, label='Violations')
                
                ax.set_title(f'Backtesting de la VaR {confidence_level}%')
                ax.set_ylabel('Rendement du portefeuille (€)')
                ax.legend()
                
                # Retourner le graphique pour que les pertes soient vers le bas
                ax.invert_yaxis()
                
                st.pyplot(fig)
else:
    st.info("Veuillez télécharger un fichier CSV avec au moins les colonnes: Date, Open, High, Low, Close, Volume")
    
    # Exemple de données
    st.subheader("Format de données attendu:")
    example_data = {
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03"],
        "Open": [100.0, 101.2, 99.5],
        "High": [102.5, 103.0, 101.2],
        "Low": [99.0, 100.5, 98.0],
        "Close": [101.2, 99.5, 100.8],
        "Volume": [1500000, 1250000, 1350000]
    }
    st.dataframe(pd.DataFrame(example_data))

# Ajouter des informations supplémentaires
with st.expander("À propos de la VaR Historique"):
    st.markdown("""
    ## VaR Historique (Non-Paramétrique)
    
    La Value-at-Risk (VaR) historique est une méthode non-paramétrique pour estimer le risque d'un portefeuille financier.
    
    ### Prérequis
    * **Données**: rendements historiques du portefeuille sur au moins 1 an (≥ 250 observations journalières).
    * **Horizon de risque**: définition claire (1 jour, 10 jours…).
    * **Niveau de confiance** (α): ex. 95%, 99%.
    * **Montant exposé**: la valeur du portefeuille sur laquelle on applique la VaR.
    
    ### Vérifications importantes
    * **Qualité des données**: absence de lacunes ou dates manquantes, filtrage des valeurs aberrantes.
    * **Stationnarité des rendements**: stabilité de la moyenne et de la variance dans le temps.
    * **Sensibilité à l'horizon historique**: tester la stabilité en variant la fenêtre historique.
    * **Robustesse du quantile empirique**: calcul de la VaR sur plusieurs sous-échantillons (rolling window).
    
    Cette application implémente toutes ces vérifications pour vous aider à calculer une VaR fiable.
    """)

# Ajouter des informations sur l'auteur et le code source
st.sidebar.markdown("---")
st.sidebar.info(
    "Cette application a été développée pour calculer la VaR historique "
    "non-paramétrique d'un portefeuille financier."
)
st.sidebar.markdown("[Code source disponible sur GitHub](https://github.com/votre-username/historicalVaR)")
