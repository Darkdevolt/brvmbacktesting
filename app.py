import pandas as pd

# Charger les données depuis le fichier CSV
df = pd.read_csv("NTLC - Données Historiques.csv")

# Correction des colonnes de prix
price_columns = ["Dernier", "Ouv.", " Plus Haut", "Plus Bas"]
for col in price_columns:
    df[col] = df[col].astype(str).str.replace(".", "").astype(float)

# Correction de la colonne "Vol."
df["Vol."] = df["Vol."].str.replace(",", ".").str.replace("K", "").astype(float) * 1000

# Correction de la colonne "Variation %"
df["Variation %"] = df["Variation %"].str.replace(",", ".").str.replace("%", "").astype(float)

# Afficher le DataFrame corrigé
print(df)
