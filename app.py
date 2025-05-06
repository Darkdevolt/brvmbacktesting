import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, normaltest, jarque_bera

st.set_page_config(page_title="VaR Calculator", layout="centered")
st.title("📉 Calcul de la Value at Risk (VaR) avec tests de normalité")

# 1) Upload du fichier CSV
file = st.file_uploader("Téléversez votre fichier CSV de prix historiques", type=["csv"])
if not file:
    st.info("Veuillez téléverser un fichier CSV pour commencer.")
    st.stop()

# 2) Lecture en détectant le séparateur
try:
    df = pd.read_csv(file, sep=None, engine='python')  # s’essaye automatiquement à détecter le séparateur
except Exception:
    file.seek(0)
    df = pd.read_csv(file)

# 3) Nettoyage des noms de colonnes
df.columns = df.columns.str.strip()

# 4) Vérification des colonnes et conversion de la date
if "Date" not in df.columns:
    st.error("La colonne 'Date' est introuvable dans votre fichier.")
    st.stop()

# Conversion du format MM/JJ/AA ; on retombe sur un parsing générique si besoin
try:
    df['Date'] = pd.to_datetime(df['Date'], format="%m/%d/%y")
except Exception:
    df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)

df = df.sort_values('Date').set_index('Date')
st.subheader("Aperçu des données (5 dernières lignes)")
st.dataframe(df.tail())

# 5) Sélection de la colonne de prix
candidates = [c for c in df.columns if c.lower() in ('close','prix','price','last')]
if candidates:
    price_col = candidates[0]
else:
    price_col = st.selectbox("Choisissez la colonne de prix à utiliser", df.columns)

st.write(f"→ **Colonne de prix utilisée :** `{price_col}`")

# 6) Calcul des rendements
df['Return'] = df[price_col].pct_change().dropna()
returns = df['Return'].dropna()

# 7) Statistiques descriptives
st.subheader("Statistiques descriptives des rendements")
st.write(returns.describe().apply(lambda x: f"{x:.6f}"))

# 8) Tests de normalité
st.subheader("Tests de normalité")
stat_shapiro, p_shapiro = shapiro(returns)
stat_dago, p_dago = normaltest(returns)
stat_jb, p_jb = jarque_bera(returns)

st.write(f"- **Shapiro-Wilk** : p-value = {p_shapiro:.5f}")
st.write(f"- **D’Agostino** : p-value = {p_dago:.5f}")
st.write(f"- **Jarque-Bera** : p-value = {p_jb:.5f}")

if p_shapiro > 0.05 and p_dago > 0.05 and p_jb > 0.05:
    st.success("✅ Les rendements sont statistiquement compatibles avec une loi normale.")
else:
    st.warning("⚠️ Les rendements ne suivent pas strictement une loi normale.")

# 9) Paramètres VaR
st.subheader("Paramètres pour le calcul de la VaR")
confidence_level = st.slider("Niveau de confiance (%)", min_value=90, max_value=99, value=95, step=1)
investment = st.number_input("Montant du portefeuille (€)", min_value=1_000, value=100_000, step=1_000)

# 10) Calcul de la VaR gaussienne
mean_return = returns.mean()
std_return = returns.std()
alpha = 1 - confidence_level / 100
z_score = norm.ppf(alpha)
var_daily = -(mean_return + z_score * std_return) * investment

st.subheader("Résultat de la VaR")
st.write(f"**VaR {confidence_level}% (1 jour)** : {var_daily:,.2f} €")
st.write(f>Il y a {confidence_level}% de chances de ne pas dépasser une perte de {abs(var_daily):,.2f} € en une journée.")

# 11) Visualisation
# Histogramme + courbe normale
st.subheader("Distribution des rendements")
fig, ax = plt.subplots()
returns.hist(bins=50, density=True, alpha=0.6, ax=ax)
x = np.linspace(returns.min(), returns.max(), 100)
ax.plot(x, norm.pdf(x, mean_return, std_return), linestyle='--', label='Loi normale')
ax.set_xlabel("Rendement")
ax.set_ylabel("Densité")
ax.legend()
st.pyplot(fig)
