import requests
from bs4 import BeautifulSoup
import pandas as pd

# URL de la page à scraper
url = 'https://fr.investing.com/indices/brvm-composite-historical-data'

# En-têtes pour la requête HTTP
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

# Effectuer la requête GET
response = requests.get(url, headers=headers)
response.raise_for_status()  # Vérifier que la requête a réussi

# Parser le contenu HTML
soup = BeautifulSoup(response.text, 'html.parser')

# Trouver le tableau contenant les données historiques
table = soup.find('table', {'id': 'curr_table'})

# Extraire les en-têtes du tableau
headers = [header.text.strip() for header in table.find_all('th')]

# Extraire les lignes du tableau
rows = []
for row in table.find_all('tr')[1:]:
    cols = [col.text.strip() for col in row.find_all('td')]
    if cols:
        rows.append(cols)

# Créer un DataFrame pandas
df = pd.DataFrame(rows, columns=headers)

# Afficher le DataFrame
print(df)