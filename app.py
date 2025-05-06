import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm, jarque_bera, shapiro, kstest, anderson
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Configuration de la page Streamlit
st.set_page_config(layout="wide", page_title="Calculateur de VaR Gaussienne")

st.title("📈 Calculateur de VaR Gaussienne")
st.markdown("""
Bienvenue dans cet outil de calcul de la Valeur à Risque (VaR) Gaussienne.
Chargez votre fichier CSV, sélectionnez les colonnes appropriées, puis suivez les étapes
pour analyser vos données de rendement et estimer la VaR.
""")

# --- Étape 1 : Collecte des données de rendements ---
st.header("Étape 1 : Collecte et sélection des données")
uploaded_file = st.file_uploader("Chargez votre fichier CSV de données historiques", type="csv")

# Initialiser les variables pour les noms de colonnes dans session_state si elles n'existent pas
if 'date_column_name' not in st.session_state:
    st.session_state.date_column_name = None
if 'close_column_name' not in st.session_state:
    st.session_state.close_column_name = None
if 'columns_validated' not in st.session_state:
    st.session_state.columns_validated = False
if 'data_processed' not in st.session_state:
    st.session_state.data_processed = None # Pour stocker les données traitées

if uploaded_file is not None:
    try:
        data_initial = pd.read_csv(uploaded_file)
        st.success("Fichier chargé avec succès !")
        st.write("Aperçu des premières lignes du fichier chargé :")
        st.write(data_initial.head())

        available_columns = data_initial.columns.tolist()

        st.subheader("Sélectionnez les colonnes à utiliser :")

        # Logique pour pré-sélectionner si les noms communs existent
        default_date_index = 0
        if 'Date' in available_columns:
            default_date_index = available_columns.index('Date')
        elif 'date' in available_columns: # common alternative
            default_date_index = available_columns.index('date')

        # Filtre les colonnes disponibles pour 'Close' pour ne pas inclure la colonne déjà sélectionnée pour la date (si elle a été choisie)
        # Cela sera mis à jour dynamiquement si Streamlit le permet facilement, sinon on pré-sélectionne.
        # Pour la simplicité de ce script, on ne fait pas de mise à jour dynamique complexe ici.

        default_close_index = 0
        potential_close_columns = [col for col in available_columns] # On ne filtre pas encore dynamiquement
        if 'Close' in potential_close_columns:
            default_close_index = potential_close_columns.index('Close')
        elif 'close' in potential_close_columns:
            default_close_index = potential_close_columns.index('close')
        elif 'Adj Close' in potential_close_columns: # common alternative
            default_close_index = potential_close_columns.index('Adj Close')


        col_select1, col_select2 = st.columns(2)
        with col_select1:
            selected_date_col = st.selectbox(
                "Sélectionnez la colonne contenant les DATES :",
                options=available_columns,
                index=default_date_index,
                key="sb_date_col"
            )
        with col_select2:
            selected_close_col = st.selectbox(
                "Sélectionnez la colonne contenant les prix de CLÔTURE :",
                options=[col for col in available_columns if col != selected_date_col], # Évite de sélectionner la même colonne
                index=([col for col in available_columns if col != selected_date_col].index('Close')
                       if 'Close' in [col for col in available_columns if col != selected_date_col]
                       else 0), # Tente de trouver 'Close' parmi les options restantes
                key="sb_close_col"
            )

        if st.button("Valider la sélection des colonnes et traiter les données", key="validate_cols_button"):
            if selected_date_col == selected_close_col:
                st.error("La colonne des dates et la colonne des prix de clôture doivent être différentes.")
                st.session_state.columns_validated = False
            else:
                st.session_state.date_column_name = selected_date_col
                st.session_state.close_column_name = selected_close_col
                st.session_state.columns_validated = True

                # Traitement des données après validation
                data = data_initial.copy() # Travailler sur une copie
                try:
                    # Convertir la colonne 'Date' en datetime
                    data['Date_Processed'] = pd.to_datetime(data[st.session_state.date_column_name])
                    data = data.sort_values(by='Date_Processed')
                    data.set_index('Date_Processed', inplace=True)
                except Exception as e:
                    st.error(f"Erreur lors de la conversion de la colonne '{st.session_state.date_column_name}' en date. Assurez-vous qu'elle est dans un format de date valide. Erreur: {e}")
                    st.session_state.columns_validated = False # Invalider si erreur
                    st.stop()

                # S'assurer que la colonne 'Close' est numérique
                if not pd.api.types.is_numeric_dtype(data[st.session_state.close_column_name]):
                    try:
                        # Tenter une conversion, en avertissant l'utilisateur
                        data[st.session_state.close_column_name] = pd.to_numeric(data[st.session_state.close_column_name], errors='coerce')
                        if data[st.session_state.close_column_name].isnull().any():
                            st.warning(f"Des valeurs non numériques dans la colonne '{st.session_state.close_column_name}' ont été converties en NaN après une tentative de conversion.")
                    except Exception as e:
                        st.error(f"La colonne '{st.session_state.close_column_name}' doit être de type numérique. Tentative de conversion échouée. Erreur: {e}")
                        st.session_state.columns_validated = False # Invalider si erreur
                        st.stop()

                data.dropna(subset=[st.session_state.close_column_name], inplace=True) # Supprimer les lignes où le close est NaN après conversion

                # Calcul des rendements
                data['Rendements'] = np.log(data[st.session_state.close_column_name] / data[st.session_state.close_column_name].shift(1))
                data.dropna(subset=['Rendements'], inplace=True) # Supprimer la première ligne NaN due au calcul du rendement

                st.session_state.data_processed = data # Stocker les données traitées

                st.success(f"Colonnes validées : Date='{st.session_state.date_column_name}', Clôture='{st.session_state.close_column_name}'. Données prêtes pour l'analyse.")
                st.subheader("Aperçu des données de rendements calculés :")
                st.write(st.session_state.data_processed[[st.session_state.close_column_name, 'Rendements']].head())

                if len(st.session_state.data_processed['Rendements']) < 250:
                    st.warning(f"Attention : Vous avez {len(st.session_state.data_processed['Rendements'])} observations de rendements. Il est recommandé d'en avoir au moins 250.")

    except pd.errors.EmptyDataError:
        st.error("Le fichier CSV est vide.")
        st.session_state.columns_validated = False
    except ValueError as ve:
        st.error(f"Erreur de valeur dans les données du CSV. Vérifiez le format. Détail: {ve}")
        st.session_state.columns_validated = False
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement ou de la pré-sélection des colonnes : {e}")
        st.session_state.columns_validated = False

# Le reste de l'application ne s'exécute que si les colonnes ont été validées et les données traitées
if st.session_state.columns_validated and st.session_state.data_processed is not None:
    data_for_analysis = st.session_state.data_processed
    rendements = data_for_analysis['Rendements']

    # --- Étape 2 : Analyse descriptive ---
    st.header("Étape 2 : Analyse descriptive")
    moyenne = rendements.mean()
    ecart_type = rendements.std()
    skewness_val = rendements.skew() # Renommé pour éviter conflit avec module
    kurtosis_val = rendements.kurtosis() # Kurtosis de Fisher (excès), normal = 0

    st.subheader("Statistiques descriptives des rendements :")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Moyenne (μ)", f"{moyenne:.6f}")
    col2.metric("Écart-type (σ)", f"{ecart_type:.6f}")
    col3.metric("Skewness", f"{skewness_val:.4f}")
    col4.metric("Kurtosis (excès)", f"{kurtosis_val:.4f}")

    if skewness_val != 0 or kurtosis_val > 0:
        st.warning("La skewness non nulle ou la kurtosis (excès) > 0 suggère une distribution potentiellement non normale.")
    else:
        st.info("La skewness et la kurtosis sont proches des valeurs attendues pour une distribution normale.")

    # --- Étape 3 : Test de normalité ---
    st.header("Étape 3 : Tests de normalité")
    st.markdown("Objectif : vérifier si les rendements suivent une loi normale.")

    alpha_test = st.slider("Seuil de significativité (alpha) pour les tests de normalité :", 0.01, 0.10, 0.05, 0.01, key="alpha_slider")

    jb_stat, jb_p_value = jarque_bera(rendements)
    shapiro_stat, shapiro_p_value = shapiro(rendements)
    ks_stat, ks_p_value = kstest(rendements, 'norm', args=(moyenne, ecart_type))
    ad_test = anderson(rendements, dist='norm')

    st.subheader("Résultats des tests statistiques :")
    results_df = pd.DataFrame({
        "Test": ["Jarque-Bera", "Shapiro-Wilk", "Kolmogorov-Smirnov", "Anderson-Darling"],
        "Statistique du test": [f"{jb_stat:.4f}", f"{shapiro_stat:.4f}", f"{ks_stat:.4f}", f"{ad_test.statistic:.4f}"],
        "P-value": [f"{jb_p_value:.4f}", f"{shapiro_p_value:.4f}", f"{ks_p_value:.4f}", "Voir ci-dessous"]
    })
    st.table(results_df)

    st.markdown("**Interprétation pour Anderson-Darling :**")
    for i in range(len(ad_test.critical_values)):
        sl, cv = ad_test.significance_level[i], ad_test.critical_values[i]
        if ad_test.statistic < cv:
            st.write(f"Statistique ({ad_test.statistic:.3f}) < valeur critique ({cv:.3f}) au niveau de significativité de {sl}%. On ne rejette pas la normalité.")
        else:
            st.write(f"Statistique ({ad_test.statistic:.3f}) > valeur critique ({cv:.3f}) au niveau de significativité de {sl}%. On rejette la normalité.")

    normality_rejected = False
    if jb_p_value < alpha_test:
        st.warning(f"Test de Jarque-Bera : P-value ({jb_p_value:.4f}) < {alpha_test} → On rejette l'hypothèse de normalité.")
        normality_rejected = True
    # ... (les autres tests de normalité suivent la même logique)
    else:
        st.success(f"Test de Jarque-Bera : P-value ({jb_p_value:.4f}) >= {alpha_test} → On ne rejette pas l'hypothèse de normalité.")

    if shapiro_p_value < alpha_test:
        st.warning(f"Test de Shapiro-Wilk : P-value ({shapiro_p_value:.4f}) < {alpha_test} → On rejette l'hypothèse de normalité.")
        normality_rejected = True if not normality_rejected else True # Conserver True si déjà rejeté
    else:
        st.success(f"Test de Shapiro-Wilk : P-value ({shapiro_p_value:.4f}) >= {alpha_test} → On ne rejette pas l'hypothèse de normalité.")

    if ks_p_value < alpha_test:
        st.warning(f"Test de Kolmogorov-Smirnov : P-value ({ks_p_value:.4f}) < {alpha_test} → On rejette l'hypothèse de normalité.")
        normality_rejected = True if not normality_rejected else True
    else:
        st.success(f"Test de Kolmogorov-Smirnov : P-value ({ks_p_value:.4f}) >= {alpha_test} → On ne rejette pas l'hypothèse de normalité.")


    # --- Étape 4 : Visualisation ---
    st.header("Étape 4 : Visualisation")

    st.subheader("Histogramme des rendements avec courbe de densité normale")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rendements, bins=50, density=True, alpha=0.6, color='g', label='Distribution des rendements')
    xmin, xmax_hist = ax.get_xlim() # Renommé pour éviter conflit
    x_norm = np.linspace(xmin, xmax_hist, 100) # Renommé pour éviter conflit
    p_norm = norm.pdf(x_norm, moyenne, ecart_type) # Renommé pour éviter conflit
    ax.plot(x_norm, p_norm, 'k', linewidth=2, label='Densité normale théorique')
    ax.set_title("Histogramme des rendements et densité normale")
    ax.set_xlabel("Rendements")
    ax.set_ylabel("Densité")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Q-Q Plot (Quantile-Quantile Plot)")
    fig_qq = plt.figure(figsize=(8, 6))
    sm.qqplot(rendements, line='s', ax=fig_qq.gca())
    plt.title("Q-Q Plot des rendements vs. Distribution Normale")
    st.pyplot(fig_qq)
    st.markdown("""
    Un Q-Q plot compare les quantiles de vos données aux quantiles d'une distribution normale théorique.
    Si les points suivent de près la ligne rouge, cela suggère que les données sont normalement distribuées.
    Des déviations systématiques de la ligne indiquent une non-normalité.
    """)

    # --- Étape 5 : Choix de la méthode VaR ---
    st.header("Étape 5 : Calcul de la VaR Gaussienne")

    if normality_rejected:
        st.error("""
        **Attention :** Les tests de normalité et/ou l'analyse descriptive suggèrent que
        la distribution des rendements **n'est probablement PAS normale**.
        La VaR gaussienne pourrait ne pas être appropriée.
        Envisagez des méthodes alternatives.
        """)
    else:
        st.success("""
        Les tests ne rejettent pas fortement l'hypothèse de normalité.
        Vous pouvez procéder avec la VaR gaussienne, mais restez prudent.
        """)

    st.subheader("Calcul de la VaR Gaussienne")
    conf_level_percent = st.slider("Niveau de confiance pour la VaR (en %)", 90.0, 99.9, 95.0, 0.1, key="conf_level_slider")
    conf_level = conf_level_percent / 100.0
    horizon_var = st.number_input("Horizon de la VaR (en jours)", min_value=1, value=1, step=1, key="horizon_input")

    var_rendement_1_jour = moyenne + norm.ppf(1 - conf_level) * ecart_type
    var_percent_1_jour = -var_rendement_1_jour * 100

    var_rendement_h_jours = (moyenne * horizon_var) + (norm.ppf(1 - conf_level) * ecart_type * np.sqrt(horizon_var))
    var_percent_h_jours = -var_rendement_h_jours * 100

    st.subheader(f"Résultats de la VaR Gaussienne ({conf_level_percent}%) :")
    st.metric(f"VaR à 1 jour (perte maximale en % du portefeuille)", f"{var_percent_1_jour:.4f}%")
    if horizon_var > 1:
        st.metric(f"VaR à {horizon_var} jours (perte maximale en % du portefeuille)", f"{var_percent_h_jours:.4f}%")

    st.markdown(f"""
    Interprétation : Il y a une probabilité de **{1-conf_level:.2%}** que la perte sur 1 jour dépasse **{var_percent_1_jour:.4f}%**.
    """)
    if horizon_var > 1:
        st.markdown(f"""
        Il y a une probabilité de **{1-conf_level:.2%}** que la perte sur {horizon_var} jours dépasse **{var_percent_h_jours:.4f}%**.
        """)

    st.markdown("""
    **Formule utilisée pour la VaR du rendement :**
    $$ VaR_{rendement, \\alpha} = \\mu + z_{\\alpha} \\cdot \\sigma $$
    Pour un horizon de $H$ jours :
    $$ VaR_{rendement, \\alpha, H} = \\mu \\cdot H + z_{\\alpha} \\cdot \\sigma \\cdot \\sqrt{H} $$
    """)

    # --- Étape 6 : Backtesting (Placeholder) ---
    st.header("Étape 6 : Backtesting (non implémenté)")
    st.info("Le backtesting est crucial mais non implémenté ici.")

elif uploaded_file is not None and not st.session_state.columns_validated:
    st.info("Veuillez valider la sélection des colonnes pour continuer l'analyse.")
elif uploaded_file is None:
    st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")


st.sidebar.header("À propos")
st.sidebar.info("""
Cette application Streamlit implémente le plan d'action pour calculer une VaR Gaussienne,
avec sélection manuelle des colonnes Date et Clôture.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Développé avec Python et Streamlit.")
