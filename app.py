import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
from datetime import datetime, timedelta
import io
import base64

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Calculateur de VaR Historique",
    page_icon="📊",
    layout="wide"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def display_header():
    """Affiche l'en-tête de l'application"""
    st.markdown('<div class="main-header">Calculateur de VaR Historique (Non-Paramétrique)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    Cette application permet de calculer la Valeur à Risque (VaR) historique d'un portefeuille basée sur les rendements passés.
    <ul>
        <li><strong>VaR Historique</strong> : Mesure non-paramétrique basée sur la distribution empirique des rendements passés.</li>
        <li><strong>Interprétation</strong> : Avec un niveau de confiance de X%, la perte ne dépassera pas la VaR dans (100-X)% des cas sur l'horizon spécifié.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

def get_data_source():
    """Interface pour choisir la source des données"""
    st.markdown('<div class="section-header">Source des données</div>', unsafe_allow_html=True)
    
    data_source = st.radio(
        "Choisir la source des données:",
        ["Yahoo Finance (Actions)", "Fichier CSV", "Données d'exemple"]
    )
    
    if data_source == "Yahoo Finance (Actions)":
        tickers = st.text_input("Symboles boursiers (séparés par des virgules)", "AAPL, MSFT, GOOGL")
        start_date = st.date_input("Date de début", datetime.now() - timedelta(days=365*2))
        end_date = st.date_input("Date de fin", datetime.now())
        
        if st.button("Charger les données"):
            with st.spinner('Chargement des données depuis Yahoo Finance...'):
                tickers_list = [ticker.strip() for ticker in tickers.split(',')]
                data = download_data(tickers_list, start_date, end_date)
                if data is not None:
                    st.session_state.data = data
                    st.session_state.returns = calculate_returns(data)
                    st.success(f"Données chargées avec succès pour {', '.join(tickers_list)}")
                else:
                    st.error("Erreur lors du chargement des données. Veuillez vérifier les symboles boursiers.")
    
    elif data_source == "Fichier CSV":
        uploaded_file = st.file_uploader("Télécharger un fichier CSV des prix ou rendements", type="csv")
        
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                
                data_type = st.radio(
                    "Type de données:",
                    ["Prix", "Rendements déjà calculés"]
                )
                
                if data_type == "Prix":
                    st.session_state.data = data
                    st.session_state.returns = calculate_returns(data)
                else:
                    st.session_state.returns = data
                    st.session_state.data = None
                
                st.success("Fichier CSV chargé avec succès")
            except Exception as e:
                st.error(f"Erreur lors du chargement du fichier: {e}")
    
    elif data_source == "Données d'exemple":
        if st.button("Charger les données d'exemple"):
            with st.spinner('Chargement des données d'exemple...'):
                data = load_sample_data()
                st.session_state.data = data
                st.session_state.returns = calculate_returns(data)
                st.success("Données d'exemple chargées avec succès")

def download_data(tickers, start_date, end_date):
    """Télécharge les données historiques depuis Yahoo Finance"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data, columns=[tickers[0]])
        return data
    except Exception as e:
        st.error(f"Erreur lors du téléchargement des données: {e}")
        return None

def load_sample_data():
    """Charge des données d'exemple"""
    # Créer des données simulées pour 3 actifs sur 2 ans
    np.random.seed(42)
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='B')
    
    # Simuler des prix avec une tendance haussière et des fluctuations
    asset1 = 100 * (1 + np.cumsum(np.random.normal(0.0005, 0.015, len(dates))))
    asset2 = 50 * (1 + np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
    asset3 = 75 * (1 + np.cumsum(np.random.normal(0.0007, 0.018, len(dates))))
    
    data = pd.DataFrame({
        'Asset1': asset1,
        'Asset2': asset2,
        'Asset3': asset3
    }, index=dates)
    
    return data

def calculate_returns(data):
    """Calcule les rendements journaliers à partir des prix"""
    returns = data.pct_change().dropna()
    return returns

def portfolio_allocation():
    """Interface pour allouer les poids du portefeuille"""
    st.markdown('<div class="section-header">Allocation du portefeuille</div>', unsafe_allow_html=True)
    
    if 'returns' not in st.session_state:
        st.warning("Veuillez d'abord charger des données")
        return None
    
    assets = st.session_state.returns.columns.tolist()
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.write("Définir les poids des actifs dans le portefeuille :")
        
        # Créer des sliders pour chaque actif
        weights = {}
        for asset in assets:
            weights[asset] = st.slider(f"Poids de {asset} (%)", 0, 100, int(100/len(assets)))
        
        # Normaliser les poids
        total_weight = sum(weights.values())
        if total_weight != 100:
            st.warning(f"La somme des poids ({total_weight}%) n'est pas égale à 100%. Les poids seront normalisés.")
        
        normalized_weights = {asset: weight/total_weight for asset, weight in weights.items()}
        
    with col2:
        st.write("**Poids normalisés :**")
        for asset, weight in normalized_weights.items():
            st.write(f"{asset}: {weight*100:.2f}%")
    
    # Créer un poids comme un array numpy
    weight_array = np.array([normalized_weights[asset] for asset in assets])
    
    return weight_array

def calculate_portfolio_returns(returns, weights):
    """Calcule les rendements du portefeuille"""
    portfolio_returns = (returns * weights).sum(axis=1)
    return portfolio_returns

def verify_data_quality(returns):
    """Vérifie la qualité des données de rendement"""
    st.markdown('<div class="section-header">Vérification de la qualité des données</div>', unsafe_allow_html=True)
    
    # Vérifier les valeurs manquantes
    missing_values = returns.isnull().sum().sum()
    if missing_values > 0:
        st.warning(f"Attention: {missing_values} valeurs manquantes détectées dans les rendements")
    else:
        st.success("Aucune valeur manquante détectée dans les rendements")
    
    # Vérifier les valeurs aberrantes (méthode simple basée sur les écarts-types)
    mean = returns.mean()
    std = returns.std()
    threshold = 3
    outliers = ((returns - mean).abs() > (threshold * std)).sum().sum()
    
    if outliers > 0:
        st.warning(f"Attention: {outliers} valeurs potentiellement aberrantes détectées (> {threshold} écarts-types)")
    else:
        st.success("Aucune valeur aberrante détectée")
    
    # Test de stationnarité (Augmented Dickey-Fuller)
    portfolio_returns = st.session_state.portfolio_returns
    adf_result = adfuller(portfolio_returns.dropna())
    
    stationarity_expander = st.expander("Résultats du test de stationnarité (Dickey-Fuller augmenté)")
    with stationarity_expander:
        st.write(f"Statistique ADF: {adf_result[0]:.4f}")
        st.write(f"P-value: {adf_result[1]:.4f}")
        st.write("Valeurs critiques:")
        for key, value in adf_result[4].items():
            st.write(f"   {key}: {value:.4f}")
        
        if adf_result[1] < 0.05:
            st.success("Les rendements semblent stationnaires (p-value < 0.05)")
        else:
            st.warning("Les rendements pourraient ne pas être stationnaires (p-value >= 0.05)")
    
    # Afficher les statistiques descriptives
    stats_expander = st.expander("Statistiques descriptives des rendements")
    with stats_expander:
        st.write(returns.describe())
    
    # Visualisation de la distribution des rendements
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique de la série temporelle
    portfolio_returns.plot(ax=axs[0])
    axs[0].set_title("Rendements du portefeuille")
    axs[0].set_xlabel("Date")
    axs[0].set_ylabel("Rendement")
    
    # Histogramme
    sns.histplot(portfolio_returns, kde=True, ax=axs[1])
    axs[1].set_title("Distribution des rendements")
    axs[1].set_xlabel("Rendement")
    axs[1].set_ylabel("Fréquence")
    
    plt.tight_layout()
    st.pyplot(fig)

def calculate_var(returns, confidence_level, holding_period, portfolio_value):
    """Calcule la VaR historique"""
    # Ajuster pour l'horizon de risque (règle de la racine carrée du temps)
    if holding_period > 1:
        # Pour la VaR historique sur plusieurs jours
        adj_returns = returns * np.sqrt(holding_period)
    else:
        adj_returns = returns
    
    # Calculer le quantile empirique
    var_percentile = 1 - confidence_level
    var_return = np.percentile(adj_returns, var_percentile * 100)
    
    # Convertir en valeur monétaire
    var_value = -var_return * portfolio_value
    
    return var_value, var_return

def test_var_sensitivity(returns, portfolio_value, confidence_levels, holding_periods):
    """Teste la sensibilité de la VaR à différents paramètres"""
    results = []
    
    for cl in confidence_levels:
        for hp in holding_periods:
            var_value, var_return = calculate_var(returns, cl, hp, portfolio_value)
            results.append({
                "Niveau de confiance": f"{cl*100:.1f}%",
                "Horizon (jours)": hp,
                "VaR (%)": f"{-var_return*100:.2f}%",
                "VaR (valeur)": f"{var_value:.2f}"
            })
    
    return pd.DataFrame(results)

def calculate_expected_shortfall(returns, confidence_level, holding_period, portfolio_value):
    """Calcule l'Expected Shortfall (ES) à partir des rendements"""
    # Seuil VaR
    var_percentile = 1 - confidence_level
    var_threshold = np.percentile(returns, var_percentile * 100)
    
    # Sélectionner les rendements inférieurs au seuil VaR
    tail_returns = returns[returns <= var_threshold]
    
    # Calculer l'ES comme la moyenne des rendements dans la queue
    es_return = tail_returns.mean()
    
    # Ajuster pour l'horizon de risque
    if holding_period > 1:
        es_return = es_return * np.sqrt(holding_period)
    
    # Convertir en valeur monétaire
    es_value = -es_return * portfolio_value
    
    return es_value, es_return

def backtest_var(returns, confidence_level, holding_period, window_size=252):
    """Effectue un backtest de la VaR sur une fenêtre glissante"""
    results = []
    
    for i in range(window_size, len(returns)):
        # Fenêtre glissante
        window = returns.iloc[i-window_size:i]
        
        # Date du point de test
        test_date = returns.index[i]
        
        # Rendement réel
        actual_return = returns.iloc[i]
        
        # Calculer la VaR sur la fenêtre
        _, var_return = calculate_var(window.values, confidence_level, holding_period, 1.0)
        
        # Violation de la VaR?
        var_breach = actual_return < var_return
        
        results.append({
            "Date": test_date,
            "Rendement Réel": actual_return,
            "VaR (%)": var_return,
            "Violation": var_breach
        })
    
    return pd.DataFrame(results)

def download_results(var_results, es_results, sensitivity_results, backtest_results=None):
    """Prépare un fichier Excel pour télécharger les résultats"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Résultats de la VaR
        var_df = pd.DataFrame([var_results])
        var_df.to_excel(writer, sheet_name='VaR_Results', index=False)
        
        # Résultats de l'ES
        es_df = pd.DataFrame([es_results])
        es_df.to_excel(writer, sheet_name='ES_Results', index=False)
        
        # Analyse de sensibilité
        sensitivity_results.to_excel(writer, sheet_name='Sensitivity_Analysis', index=False)
        
        # Backtest (si disponible)
        if backtest_results is not None:
            backtest_results.to_excel(writer, sheet_name='Backtest_Results', index=False)
    
    processed_data = output.getvalue()
    b64 = base64.b64encode(processed_data).decode()
    return f'data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}'

def run_var_calculation():
    """Effectue le calcul de la VaR avec les paramètres sélectionnés"""
    st.markdown('<div class="section-header">Paramètres de la VaR</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_level = st.slider("Niveau de confiance (%)", 90, 99, 95) / 100
        holding_period = st.number_input("Horizon de risque (jours)", 1, 30, 1, 1)
    
    with col2:
        portfolio_value = st.number_input("Valeur du portefeuille", 10000.0, 10000000.0, 100000.0, 10000.0)
    
    if st.button("Calculer la VaR"):
        portfolio_returns = st.session_state.portfolio_returns
        
        # Vérifier si nous avons suffisamment de données
        if len(portfolio_returns) < 250:
            st.warning(f"Attention: Seulement {len(portfolio_returns)} observations disponibles. Il est recommandé d'avoir au moins 250 observations.")
        
        # Calculer la VaR
        var_value, var_return = calculate_var(portfolio_returns.values, confidence_level, holding_period, portfolio_value)
        
        # Calculer l'Expected Shortfall
        es_value, es_return = calculate_expected_shortfall(portfolio_returns.values, confidence_level, holding_period, portfolio_value)
        
        # Afficher les résultats
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.markdown(f"### Résultats VaR et Expected Shortfall")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**VaR {confidence_level*100:.1f}% sur {holding_period} jour(s):**")
            st.markdown(f"- **Pourcentage:** {-var_return*100:.2f}%")
            st.markdown(f"- **Montant:** {var_value:.2f}")
        
        with col2:
            st.markdown(f"**Expected Shortfall {confidence_level*100:.1f}% sur {holding_period} jour(s):**")
            st.markdown(f"- **Pourcentage:** {-es_return*100:.2f}%")
            st.markdown(f"- **Montant:** {es_value:.2f}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Stocker les résultats dans la session
        st.session_state.var_results = {
            "Niveau de confiance": f"{confidence_level*100:.1f}%",
            "Horizon de risque": holding_period,
            "VaR (%)": f"{-var_return*100:.2f}%",
            "VaR (valeur)": var_value
        }
        
        st.session_state.es_results = {
            "Niveau de confiance": f"{confidence_level*100:.1f}%",
            "Horizon de risque": holding_period,
            "ES (%)": f"{-es_return*100:.2f}%",
            "ES (valeur)": es_value
        }
        
        # Analyse de sensibilité
        st.markdown('<div class="section-header">Analyse de sensibilité</div>', unsafe_allow_html=True)
        
        confidence_levels = [0.90, 0.95, 0.99]
        holding_periods = [1, 5, 10, 20]
        
        sensitivity_results = test_var_sensitivity(
            portfolio_returns.values, 
            portfolio_value, 
            confidence_levels, 
            holding_periods
        )
        
        st.write(sensitivity_results)
        st.session_state.sensitivity_results = sensitivity_results
        
        # Backtest
        if len(portfolio_returns) > 500:  # Seulement si nous avons suffisamment de données
            st.markdown('<div class="section-header">Backtest de la VaR</div>', unsafe_allow_html=True)
            
            backtest_results = backtest_var(
                portfolio_returns,
                confidence_level,
                holding_period
            )
            
            # Calculer le taux de violation
            violation_rate = backtest_results['Violation'].mean()
            expected_rate = 1 - confidence_level
            
            st.write(f"**Taux de violation:** {violation_rate:.2%} (Attendu: {expected_rate:.2%})")
            
            # Graphique du backtest
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(backtest_results['Date'], backtest_results['Rendement Réel'], label='Rendement réel', color='blue')
            ax.plot(backtest_results['Date'], backtest_results['VaR (%)'], label='VaR', color='red')
            ax.fill_between(
                backtest_results['Date'],
                backtest_results['VaR (%)'],
                backtest_results['Rendement Réel'].min() * 1.1,
                where=(backtest_results['Rendement Réel'] < backtest_results['VaR (%)']),
                color='red',
                alpha=0.3,
                label='Violations'
            )
            ax.set_title(f"Backtest de la VaR {confidence_level*100:.1f}% sur {holding_period} jour(s)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Rendement")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            st.session_state.backtest_results = backtest_results
        
        # Téléchargement des résultats
        backtest_data = st.session_state.get('backtest_results', None)
        excel_data = download_results(
            st.session_state.var_results,
            st.session_state.es_results,
            st.session_state.sensitivity_results,
            backtest_data
        )
        
        st.markdown(
            f'<a href="{excel_data}" download="var_results.xlsx">Télécharger les résultats (Excel)</a>',
            unsafe_allow_html=True
        )

def main():
    display_header()
    
    # Menu latéral avec étapes
    st.sidebar.markdown("## Étapes")
    step = st.sidebar.radio(
        "Sélectionner une étape:",
        ["1. Chargement des données", 
         "2. Allocation du portefeuille", 
         "3. Vérification de la qualité", 
         "4. Calcul de la VaR"]
    )
    
    if step == "1. Chargement des données":
        get_data_source()
        
        # Afficher un aperçu des données chargées
        if 'returns' in st.session_state:
            st.markdown('<div class="section-header">Aperçu des rendements</div>', unsafe_allow_html=True)
            st.write(st.session_state.returns.head())
            
            # Statistiques de base
            st.write(f"**Période:** {st.session_state.returns.index.min()} à {st.session_state.returns.index.max()}")
            st.write(f"**Nombre d'observations:** {len(st.session_state.returns)}")
            
            # Graphique des rendements
            st.markdown('<div class="section-header">Visualisation des rendements</div>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(10, 5))
            st.session_state.returns.plot(ax=ax)
            ax.set_title("Rendements des actifs")
            ax.set_xlabel("Date")
            ax.set_ylabel("Rendement")
            plt.tight_layout()
            st.pyplot(fig)
    
    elif step == "2. Allocation du portefeuille":
        if 'returns' not in st.session_state:
            st.warning("Veuillez d'abord charger des données (étape 1)")
        else:
            weights = portfolio_allocation()
            
            if weights is not None:
                # Calculer les rendements du portefeuille
                portfolio_returns = calculate_portfolio_returns(st.session_state.returns, weights)
                st.session_state.portfolio_returns = portfolio_returns
                
                # Afficher les statistiques du portefeuille
                st.markdown('<div class="section-header">Statistiques du portefeuille</div>', unsafe_allow_html=True)
                
                stats = {
                    "Rendement moyen (annualisé)": f"{portfolio_returns.mean() * 252 * 100:.2f}%",
                    "Volatilité (annualisée)": f"{portfolio_returns.std() * np.sqrt(252) * 100:.2f}%",
                    "Rendement minimum": f"{portfolio_returns.min() * 100:.2f}%",
                    "Rendement maximum": f"{portfolio_returns.max() * 100:.2f}%"
                }
                
                col1, col2 = st.columns(2)
                
                for i, (key, value) in enumerate(stats.items()):
                    if i < 2:
                        col1.metric(key, value)
                    else:
                        col2.metric(key, value)
                
                # Graphique des rendements du portefeuille
                fig, ax = plt.subplots(figsize=(10, 5))
                portfolio_returns.plot(ax=ax)
                ax.set_title("Rendements du portefeuille")
                ax.set_xlabel("Date")
                ax.set_ylabel("Rendement")
                plt.tight_layout()
                st.pyplot(fig)
    
    elif step == "3. Vérification de la qualité":
        if 'portfolio_returns' not in st.session_state:
            st.warning("Veuillez d'abord configurer l'allocation du portefeuille (étape 2)")
        else:
            verify_data_quality(st.session_state.returns)
    
    elif step == "4. Calcul de la VaR":
        if 'portfolio_returns' not in st.session_state:
            st.warning("Veuillez d'abord configurer l'allocation du portefeuille (étape 2)")
        else:
            run_var_calculation()
    
    # Informations dans la barre latérale
    st.sidebar.markdown("---")
    st.sidebar.markdown("### À propos")
    st.sidebar.info("""
    **Calculateur de VaR Historique**
    
    Cette application permet de calculer la Valeur à Risque (VaR) historique 
    d'un portefeuille à partir des rendements historiques.
    
    Développé en utilisant Streamlit.
    """)

if __name__ == "__main__":
    main()
