import streamlit as st
import pandas as pd
from io import TextIOWrapper

# Configuration de la page
st.set_page_config(
    page_title="Stratégie d'Investissement",
    page_icon="💹",
    layout="wide"
)

# Colonnes obligatoires
COLONNES_REQUISES = [
    "Exchange Date", "Close", "Net", "%Chg", "Open", 
    "Low", "High", "Volume", "Turnover - XOF", "Flow"
]

def verifier_csv(uploaded_file):
    """Vérifie le format du CSV avant traitement"""
    try:
        decoded_file = TextIOWrapper(uploaded_file, encoding='utf-8')
        df = pd.read_csv(decoded_file)
        return all(col in df.columns for col in COLONNES_REQUISES)
    except Exception as e:
        st.error(f"ERREUR CRITIQUE : Format de fichier invalide ({str(e)})")
        return False

def main():
    st.title("📈 Stratégie d'Investissement par Valeur Fondamentale")
    
    # Téléversement du CSV
    uploaded_file = st.file_uploader("Téléversez votre fichier historique", type="csv")
    
    # Paramètres d'investissement
    with st.sidebar:
        st.header("Paramètres")
        valeur_fondamentale = st.number_input(
            "Valeur fondamentale (XOF)", 
            min_value=0.0, 
            value=9000.0, 
            step=100.0
        )
        montant_min = st.number_input(
            "Montant minimum d'investissement", 
            min_value=10000.0, 
            value=10000.0, 
            step=1000.0
        )

    # Traitement principal
    if uploaded_file and valeur_fondamentale:
        if verifier_csv(uploaded_file):
            df = pd.read_csv(uploaded_file)
            
            # Conversion des nombres
            df["Close"] = df["Close"].str.replace('[^0-9.]', '', regex=True).astype(float)
            df["Turnover - XOF"] = df["Turnover - XOF"].str.replace('[^0-9.]', '', regex=True).astype(float)
            
            # Filtrage des opportunités
            opportunites = df[df["Close"] < valeur_fondamentale]
            
            if not opportunites.empty:
                st.success(f"{len(opportunites)} opportunité(s) détectée(s)")
                
                # Affichage interactif
                with st.expander("Voir les opportunités", expanded=True):
                    for idx, row in opportunites.iterrows():
                        col1, col2, col3 = st.columns([2, 2, 3])
                        with col1:
                            st.markdown(f"**Date** : {row['Exchange Date']}")
                        with col2:
                            st.markdown(f"**Prix** : {row['Close']} XOF")
                        with col3:
                            investissement = st.number_input(
                                f"Montant pour le {row['Exchange Date']}",
                                min_value=0.0,
                                value=0.0,
                                step=1000.0,
                                key=idx
                            )
                            
                # Validation finale
                if st.button("💸 Confirmer les investissements"):
                    investments = [st.session_state.get(key, 0) for key in range(len(opportunites))]
                    if any(0 < inv < montant_min for inv in investments):
                        st.error(f"Montant invalide ! Minimum requis : {montant_min} XOF")
                    else:
                        total = sum(investments)
                        st.balloons()
                        st.success(f"Investissement total confirmé : {total:,.0f} XOF")
            else:
                st.warning("Aucune opportunité selon vos critères")

if __name__ == "__main__":
    main()
