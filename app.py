import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd

@st.cache_data(show_spinner=True)
def get_historical_data(url):
    """
    Scrape la page et retourne un DataFrame des données historiques.
    """
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Rechercher la table contenant les données. Adaptez ce sélecteur selon la structure du site.
        table = soup.find("table")
        if table is None:
            st.error("Aucune table n'a été trouvée sur la page.")
            return None
        # Extraction des entêtes
        headers = [th.text.strip() for th in table.find_all("th")]
        # Extraction des lignes
        rows = []
        for tr in table.find_all("tr"):
            cols = tr.find_all("td")
            if cols:
                row = [td.text.strip() for td in cols]
                rows.append(row)
        if headers and rows:
            df = pd.DataFrame(rows, columns=headers)
            return df
    else:
        st.error(f"Erreur lors de la récupération de la page (HTTP {response.status_code}).")
    return None

# URL de la page contenant les données historiques (à adapter selon le site)
url = "https://www.sikafinance.com/marches/download/BRVMC"

st.title("Données historiques de la BRVM Composite")
st.write("Cette application récupère et affiche les données historiques de la BRVM via du web scraping.")

# Récupération des données
data = get_historical_data(url)

if data is not None:
    st.subheader("Aperçu des données")
    st.dataframe(data)
else:
    st.warning("Les données n'ont pas pu être récupérées.")

st.write("Ce code est hébergé sur GitHub et déployé via Streamlit Cloud.")