import streamlit as st
import pandas as pd
import investpy

# Fonction pour récupérer les données historiques d'une action
def get_historical_data(stock_name, country, from_date, to_date):
    try:
        data = investpy.get_stock_historical_data(stock=stock_name,
                                                  country=country,
                                                  from_date=from_date,
                                                  to_date=to_date)
        return data
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données : {e}")
        return None

# Interface utilisateur Streamlit
st.title("Données Historiques des Actions de la BRVM")

# Sélection de l'action
stock_name = st.text_input("Nom de l'action (ex: 'BOAS')", "BOAS")

# Dates de début et de fin
from_date = st.date_input("Date de début", pd.to_datetime("2022-01-01"))
to_date = st.date_input("Date de fin", pd.to_datetime("2023-01-01"))

# Bouton pour récupérer les données
if st.button("Récupérer les données"):
    data = get_historical_data(stock_name, 'cote d\'ivoire', from_date.strftime('%d/%m/%Y'), to_date.strftime('%d/%m/%Y'))
    if data is not None:
        st.write(f"Données pour l'action {stock_name} de la BRVM")
        st.dataframe(data)
        st.line_chart(data['Close'])