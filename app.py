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
Suivez les étapes ci-dessous pour analyser vos données de rendement et estimer la VaR.
""")

# --- Étape 1 : Collecte des données de rendements ---
st.header("Étape 1 : Collecte des données de rendements")
uploaded_file = st.file_uploader("Chargez votre fichier CSV de données historiques (avec les colonnes 'Date' et 'Close')", type="csv")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Fichier chargé avec succès !")

        # Vérifier les colonnes nécessaires
        required_columns = ['Date', 'Close'] # Adaptez 'Close' si le nom de votre colonne de prix est différent
        if not all(col in data.columns for col in required_columns):
            st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {', '.join(required_columns)}")
            st.stop()

        # Convertir la colonne 'Date' en datetime
        try:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values(by='Date')
            data.set_index('Date', inplace=True)
        except Exception as e:
            st.error(f"Erreur lors de la conversion de la colonne 'Date'. Assurez-vous qu'elle est dans un format de date valide. Erreur: {e}")
            st.stop()

        # Calcul des rendements (log-rendements sont souvent préférés)
        # Assurez-vous que la colonne 'Close' existe et est numérique
        if 'Close' not in data.columns:
            st.error("La colonne 'Close' est manquante dans vos données.")
            st.stop()
        if not pd.api.types.is_numeric_dtype(data['Close']):
            st.error("La colonne 'Close' doit être de type numérique.")
            st.stop()

        data['Rendements'] = np.log(data['Close'] / data['Close'].shift(1))
        data.dropna(inplace=True) # Supprimer la première ligne NaN due au calcul du rendement

        st.subheader("Aperçu des données de rendements :")
        st.write(data[['Close', 'Rendements']].head())

        if len(data['Rendements']) < 250:
            st.warning(f"Attention : Vous avez {len(data['Rendements'])} observations de rendements. Il est recommandé d'en avoir au moins 250.")

        rendements = data['Rendements']

        # --- Étape 2 : Analyse descriptive ---
        st.header("Étape 2 : Analyse descriptive")
        moyenne = rendements.mean()
        ecart_type = rendements.std()
        skewness = rendements.skew()
        kurtosis_val = rendements.kurtosis() # Kurtosis de Fisher (excès), normal = 0

        st.subheader("Statistiques descriptives des rendements :")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Moyenne (μ)", f"{moyenne:.6f}")
        col2.metric("Écart-type (σ)", f"{ecart_type:.6f}")
        col3.metric("Skewness", f"{skewness:.4f}")
        col4.metric("Kurtosis (excès)", f"{kurtosis_val:.4f}") # Kurtosis de Fisher, où 3 est normal, donc excès de 0.

        if skewness != 0 or kurtosis_val > 0: # kurtosis > 3 (Fisher > 0)
            st.warning("La skewness non nulle ou la kurtosis (excès) > 0 suggère une distribution potentiellement non normale.")
        else:
            st.info("La skewness et la kurtosis sont proches des valeurs attendues pour une distribution normale.")

        # --- Étape 3 : Test de normalité ---
        st.header("Étape 3 : Tests de normalité")
        st.markdown("Objectif : vérifier si les rendements suivent une loi normale.")

        alpha_test = st.slider("Seuil de significativité (alpha) pour les tests de normalité :", 0.01, 0.10, 0.05, 0.01)

        # Test de Jarque-Bera
        jb_stat, jb_p_value = jarque_bera(rendements)
        # Test de Shapiro-Wilk (plus adapté pour des échantillons plus petits)
        shapiro_stat, shapiro_p_value = shapiro(rendements)
        # Test de Kolmogorov-Smirnov (contre une normale spécifique, ici on utilise les paramètres estimés)
        ks_stat, ks_p_value = kstest(rendements, 'norm', args=(moyenne, ecart_type))
        # Test d'Anderson-Darling
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
        else:
            st.success(f"Test de Jarque-Bera : P-value ({jb_p_value:.4f}) >= {alpha_test} → On ne rejette pas l'hypothèse de normalité.")

        if shapiro_p_value < alpha_test:
            st.warning(f"Test de Shapiro-Wilk : P-value ({shapiro_p_value:.4f}) < {alpha_test} → On rejette l'hypothèse de normalité.")
            normality_rejected = True
        else:
            st.success(f"Test de Shapiro-Wilk : P-value ({shapiro_p_value:.4f}) >= {alpha_test} → On ne rejette pas l'hypothèse de normalité.")

        if ks_p_value < alpha_test:
            st.warning(f"Test de Kolmogorov-Smirnov : P-value ({ks_p_value:.4f}) < {alpha_test} → On rejette l'hypothèse de normalité.")
            normality_rejected = True
        else:
            st.success(f"Test de Kolmogorov-Smirnov : P-value ({ks_p_value:.4f}) >= {alpha_test} → On ne rejette pas l'hypothèse de normalité.")


        # --- Étape 4 : Visualisation ---
        st.header("Étape 4 : Visualisation")

        # Histogramme et densité
        st.subheader("Histogramme des rendements avec courbe de densité normale")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(rendements, bins=50, density=True, alpha=0.6, color='g', label='Distribution des rendements')
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, moyenne, ecart_type)
        ax.plot(x, p, 'k', linewidth=2, label='Densité normale théorique')
        ax.set_title("Histogramme des rendements et densité normale")
        ax.set_xlabel("Rendements")
        ax.set_ylabel("Densité")
        ax.legend()
        st.pyplot(fig)

        # Q-Q plot
        st.subheader("Q-Q Plot (Quantile-Quantile Plot)")
        fig_qq = plt.figure(figsize=(8, 6))
        sm.qqplot(rendements, line='s', ax=fig_qq.gca()) # 's' pour standardiser
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
            Envisagez des méthodes alternatives comme la VaR historique, la VaR Monte Carlo
            ou des distributions alternatives (ex: t de Student).
            """)
        else:
            st.success("""
            Les tests de normalité et l'analyse descriptive ne rejettent pas fortement l'hypothèse
            de normalité. Vous pouvez procéder avec la VaR gaussienne, mais restez prudent.
            """)

        st.subheader("Calcul de la VaR Gaussienne")
        conf_level_percent = st.slider("Niveau de confiance pour la VaR (en %)", 90.0, 99.9, 95.0, 0.1)
        conf_level = conf_level_percent / 100.0
        horizon_var = st.number_input("Horizon de la VaR (en jours)", min_value=1, value=1, step=1)

        # Calcul de z_alpha
        z_alpha = norm.ppf(1 - conf_level) # ppf donne le quantile pour la queue de gauche, donc 1-conf pour la perte

        # VaR pour 1 jour
        var_1_jour = moyenne - z_alpha * ecart_type  # Formule VaRalpha = mu - z_alpha * sigma. PERTE donc signe -
                                                # Si on veut la perte en positif : z_alpha * ecart_type - moyenne
                                                # Ou plus classiquement (pour une perte): VaR = PortfolioValue * (z_alpha * sigma - mu)
                                                # Ici on calcule la VaR du rendement.
                                                # Si mu est positif, il réduit la perte.
                                                # Si z_alpha est positif (ex pour 95%, z_alpha = 1.645),
                                                # z_alpha * sigma est la "perte potentielle due à la volatilité".
                                                # Le signe est important. Pour une perte, Z est généralement pris positif pour la queue gauche des pertes.
                                                # norm.ppf(alpha) donne Z pour la queue gauche. Si alpha=0.05, Z=-1.645.
                                                # VaR = mu + Z_alpha * sigma
                                                # VaR = moyenne_rendements + norm.ppf(1-niveau_de_confiance) * ecart_type_rendements
                                                # Si on veut la perte maximale (valeur positive):
                                                # VaR = -(moyenne_rendements + norm.ppf(1-niveau_de_confiance) * ecart_type_rendements)
                                                # Ou VaR = moyenne_rendements - norm.ppf(niveau_de_confiance) * ecart_type_rendements

        # Pour la VaR, on s'intéresse à la perte. Si z_alpha vient de norm.ppf(alpha_confiance),
        # il sera positif (ex: norm.ppf(0.95) = 1.645).
        # La formule VaR = mu - Z_alpha * sigma est correcte si Z_alpha est le quantile de la *perte*.
        # Conventionnellement, si on utilise alpha = 0.05 (niveau de significativité)
        # Z_alpha_perte = norm.ppf(0.05) = -1.645
        # VaR_rendement = moyenne + Z_alpha_perte * ecart_type
        # Si on utilise le niveau de confiance (ex: 95%), Z_confiance = norm.ppf(0.95) = 1.645
        # VaR_rendement = moyenne - Z_confiance * ecart_type
        # Cette dernière est celle utilisée dans votre formule.

        # VaR pour 1 jour
        z_score = norm.ppf(1 - conf_level) # Ceci est z_{1-alpha}, qui est négatif
                                          # Si on veut la VaR comme une perte (donc positive), on peut faire:
                                          # VaR = (moyenne - z_score_pour_conf_level * ecart_type) * (-1)
                                          # Où z_score_pour_conf_level = norm.ppf(conf_level)
                                          # Ou plus simplement, VaR = - (moyenne + norm.ppf(1-conf_level) * ecart_type)
                                          # Ou encore VaR = abs(moyenne + norm.ppf(1-conf_level)*ecart_type) si mu > 0
                                          # Ou bien VaR_perte = -(mu + Z_alpha*sigma) ou Z_alpha = norm.ppf(1-niveau_de_confiance)
                                          # La formule originale : VaR_alpha = mu - z_alpha_droite * sigma (où z_alpha_droite est le quantile de droite)

        # Prenons la convention où VaR est une perte (valeur positive)
        # Si le rendement est X ~ N(mu, sigma^2)
        # P(X < seuil_perte) = alpha  => seuil_perte = mu + z_alpha * sigma  (où z_alpha = norm.ppf(alpha))
        # La VaR est alors -seuil_perte si on veut la perte en valeur positive si le seuil est négatif
        # Ou, plus directement, la perte maximale est - (mu + z_alpha * sigma)
        # Si on utilise le niveau de confiance C (ex: 0.95), alpha = 1-C (ex: 0.05)
        # z_alpha = norm.ppf(1-C)
        # VaR_rendement = moyenne + z_alpha * ecart_type
        # Cette valeur sera négative, indiquant une perte. Pour l'exprimer en % de perte :
        var_rendement_1_jour = moyenne + norm.ppf(1 - conf_level) * ecart_type
        var_percent_1_jour = -var_rendement_1_jour * 100 # En pourcentage de perte

        # VaR pour H jours (ajustement par la racine carrée du temps)
        var_rendement_h_jours = (moyenne * horizon_var) + (norm.ppf(1 - conf_level) * ecart_type * np.sqrt(horizon_var))
        var_percent_h_jours = -var_rendement_h_jours * 100 # En pourcentage de perte

        st.subheader(f"Résultats de la VaR Gaussienne ({conf_level_percent}%) :")
        st.write(f"Quantile $z_{{{1-conf_level:.2f}}}$ (associé à la perte) : {norm.ppf(1-conf_level):.4f}")
        st.write(f"Quantile $z_{{{conf_level:.2f}}}$ (associé à la confiance) : {norm.ppf(conf_level):.4f}")

        st.metric(f"VaR à 1 jour (perte maximale en % du portefeuille)", f"{var_percent_1_jour:.4f}%")
        if horizon_var > 1:
            st.metric(f"VaR à {horizon_var} jours (perte maximale en % du portefeuille)", f"{var_percent_h_jours:.4f}%")

        st.markdown(f"""
        Interprétation :
        - Il y a une probabilité de **{1-conf_level:.2%}** que la perte sur 1 jour dépasse **{var_percent_1_jour:.4f}%**.
        - Autrement dit, nous sommes confiant à **{conf_level_percent}%** que la perte sur 1 jour ne dépassera pas **{var_percent_1_jour:.4f}%**.
        """)
        if horizon_var > 1:
            st.markdown(f"""
            - Il y a une probabilité de **{1-conf_level:.2%}** que la perte sur {horizon_var} jours dépasse **{var_percent_h_jours:.4f}%**.
            - Autrement dit, nous sommes confiant à **{conf_level_percent}%** que la perte sur {horizon_var} jours ne dépassera pas **{var_percent_h_jours:.4f}%**.
            """)

        st.markdown("""
        **Formule utilisée pour la VaR du rendement :**
        $$ VaR_{rendement, \alpha} = \mu + z_{\alpha} \cdot \sigma $$
        où $\mu$ est la moyenne des rendements, $\sigma$ est l'écart-type des rendements,
        et $z_{\alpha}$ est le quantile de la loi normale standard correspondant à la probabilité $\alpha = 1 - \text{niveau de confiance}$.
        Par exemple, pour une confiance de 95%, $\alpha = 0.05$, et $z_{0.05} \approx -1.645$.
        La VaR en pourcentage de perte est alors $-VaR_{rendement, \alpha} \times 100$.

        Pour un horizon de $H$ jours :
        $$ VaR_{rendement, \alpha, H} = \mu \cdot H + z_{\alpha} \cdot \sigma \cdot \sqrt{H} $$
        """)


        # --- Étape 6 : Backtesting (Placeholder) ---
        st.header("Étape 6 : Backtesting (non implémenté dans cette version)")
        st.info("""
        Le backtesting est une étape cruciale pour vérifier la validité ex post du modèle de VaR.
        Cela implique de comparer les pertes réelles aux VaR estimées sur une période passée et
        d'utiliser des tests statistiques comme le test de Kupiec ou le test de Christoffersen.
        Cette fonctionnalité n'est pas implémentée dans cette version de démonstration.
        """)

    except pd.errors.EmptyDataError:
        st.error("Le fichier CSV est vide.")
    except ValueError as ve:
        st.error(f"Erreur de valeur dans les données du CSV. Assurez-vous que les colonnes numériques le sont bien. Détail: {ve}")
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement du fichier : {e}")
else:
    st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")

st.sidebar.header("À propos")
st.sidebar.info("""
Cette application Streamlit implémente le plan d'action pour calculer une VaR Gaussienne.
Elle couvre la collecte de données, l'analyse descriptive, les tests de normalité,
la visualisation et le calcul de la VaR.
Le backtesting (Étape 6) n'est pas inclus.
""")
st.sidebar.markdown("---")
st.sidebar.markdown("Développé avec ❤️ en utilisant Python et Streamlit.")
