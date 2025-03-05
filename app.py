import streamlit as st
import pandas as pd
from io import BytesIO, TextIOWrapper

# Configuration de la page
st.set_page_config(
    page_title="Stratégie d'Investissement BRVM",
    page_icon="💹",
    layout="wide"
)

COLONNES_REQUISES = [
    "Exchange Date", "Close", "Net", "%Chg", "Open", 
    "Low", "High", "Volume", "Turnover - XOF", "Flow"
]

def verifier_csv(uploaded_file):
    """Vérifie le format du CSV de manière robuste"""
    try:
        # Création d'une copie indépendante du fichier
        bytes_data = uploaded_file.getvalue()
        buffer = BytesIO(bytes_data)
        
        # Lecture et vérification des colonnes
        df = pd.read_csv(TextIOWrapper(buffer, encoding='utf-8'))
        if not all(col in df.columns for col in COLONNES_REQUISES):
            st.error("Structure du fichier CSV invalide!")
            st.write(f"Colonnes requises: {COLONNES_REQUISES}")
            return False
        return True
    except Exception as e:
        st.error(f"Erreur de lecture : {str(e)}")
        return False

def main():
    st.title("📈 Stratégie d'Investissement BRVM")
    
    # Téléversement du fichier
    uploaded_file = st.file_uploader("Téléversez votre fichier historique (CSV)", type="csv")
    
    # Paramètres dans la sidebar
    with st.sidebar:
        st.header("Paramètres d'Investissement")
        valeur_fondamentale = st.number_input(
            "Valeur fondamentale (XOF)",
            min_value=0.0,
            value=9000.0,
            step=100.0
        )
        montant_min = st.number_input(
            "Montant minimum par investissement",
            min_value=10000.0,
            value=10000.0,
            step=1000.0
        )

    if uploaded_file and valeur_fondamentale:
        if verifier_csv(uploaded_file):
            # Réinitialisation du buffer pour lecture complète
            bytes_data = uploaded_file.getvalue()
            buffer = BytesIO(bytes_data)
            
            # Lecture définitive avec gestion d'erreur
            try:
                df = pd.read_csv(TextIOWrapper(buffer, encoding='utf-8'))
                
                # Nettoyage des données numériques
                df["Close"] = df["Close"].str.replace('[^0-9.]', '', regex=True).astype(float)
                df["Turnover - XOF"] = df["Turnover - XOF"].str.replace('[^0-9.]', '', regex=True).astype(float)
                
                # Détection des opportunités
                opportunites = df[df["Close"] < valeur_fondamentale]
                
                if not opportunites.empty:
                    st.success(f"🚀 {len(opportunites)} opportunité(s) détectée(s)!")
                    
                    # Affichage des opportunités
                    with st.expander("Détail des opportunités", expanded=True):
                        for idx, row in opportunites.iterrows():
                            col1, col2, col3 = st.columns([2, 2, 4])
                            with col1:
                                st.markdown(f"**Date** : {row['Exchange Date']}")
                            with col2:
                                st.markdown(f"**Prix de clôture** : {row['Close']:,.0f} XOF")
                            with col3:
                                investissement = st.number_input(
                                    f"Montant à investir ({montant_min:,.0f} XOF min)",
                                    min_value=0.0,
                                    value=0.0,
                                    step=1000.0,
                                    key=f"inv_{idx}"
                                )
                    
                    # Validation finale
                    if st.button("✅ Confirmer les investissements", type="primary"):
                        investments = [st.session_state.get(f"inv_{idx}", 0) for idx in range(len(opportunites))]
                        
                        if any(0 < inv < montant_min for inv in investments):
                            st.error(f"Erreur : Le montant minimum est de {montant_min:,.0f} XOF")
                        else:
                            total_investi = sum(investments)
                            st.balloons()
                            st.success(f"Investissement total validé : {total_investi:,.0f} XOF")
                            st.session_state.investissements = dict(zip(
                                opportunites["Exchange Date"], 
                                investments
                            ))
                else:
                    st.warning("Aucune opportunité selon vos critères actuels")
                    
            except Exception as e:
                st.error(f"Erreur lors du traitement des données : {str(e)}")

if __name__ == "__main__":
    main()
