import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, normaltest, jarque_bera

st.set_page_config(page_title="VaR Calculator", layout="centered")

st.title("📉 Calcul de la Value at Risk (VaR) avec tests de normalité")

# Upload du fichier CSV
file = st.file_uploader("Téléversez le fichier 'HistoricalPrices.csv'", type=["csv"])

if file:
    df = pd.read_csv(file, dayfirst=False)
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y")
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)

    st.subheader("Aperçu des données")
    st.dataframe(df.tail())

    # Calcul des rendements
    df['Return'] = df['Close'].pct_change()
    returns = df['Return'].dropna()

    st.subheader("Statistiques descriptives")
    st.write(returns.describe())

    # Tests de normalité
    st.subheader("Tests de normalité")
    stat_shapiro, p_shapiro = shapiro(returns)
    stat_dagostino, p_dagostino = normaltest(returns)
    stat_jb, p_jb = jarque_bera(returns)

    st.write(f"**Shapiro-Wilk Test**: p-value = {p_shapiro:.5f}")
    st.write(f"**D'Agostino Test**: p-value = {p_dagostino:.5f}")
    st.write(f"**Jarque-Bera Test**: p-value = {p_jb:.5f}")

    if p_shapiro > 0.05 and p_dagostino > 0.05 and p_jb > 0.05:
        st.success("✅ Les rendements suivent approximativement une loi normale (selon les tests).")
    else:
        st.warning("⚠️ Les rendements ne suivent pas une loi normale stricte. Utiliser la VaR gaussienne avec prudence.")

    # Choix utilisateur
    st.subheader("Paramètres pour le calcul de la VaR")
    confidence_level = st.slider("Niveau de confiance (%)", 90, 99, 95)
    investment = st.number_input("Montant du portefeuille (€)", min_value=1000, value=100000)

    # Calcul VaR
    mean_return = returns.mean()
    std_return = returns.std()
    alpha = 1 - confidence_level / 100
    var_value = norm.ppf(alpha, mean_return, std_return) * investment

    st.subheader("Résultat")
    st.write(f"📌 **VaR {confidence_level}%** : {var_value:,.2f} €")
    st.write(f"Interprétation : Il y a {confidence_level}% de chances de ne pas perdre plus de {abs(var_value):,.2f} € en une journée.")

    # Histogramme des rendements
    st.subheader("Distribution des rendements")
    fig, ax = plt.subplots()
    returns.hist(bins=50, density=True, alpha=0.6, color='skyblue', ax=ax)
    x = np.linspace(returns.min(), returns.max(), 100)
    ax.plot(x, norm.pdf(x, mean_return, std_return), color='red')
    ax.set_title("Histogramme des rendements avec courbe normale")
    st.pyplot(fig)
else:
    st.info("Veuillez téléverser un fichier CSV pour commencer.")
