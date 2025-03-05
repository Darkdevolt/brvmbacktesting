import streamlit as st
import pandas as pd
from io import BytesIO, TextIOWrapper

# Configuration de la page
st.set_page_config(
    page_title="Stratégie d'Investissement",
    page_icon="💹",
    layout="wide"
)

COLONNES_REQUISES = [
    "Exchange Date", "Close", "Net", "%Chg", "Open", 
    "Low", "High", "Volume", "Turnover - XOF", "Flow"
]

def verifier_csv(uploaded_file):
    """Vérifie le format du CSV sans consommer le fichier"""
    try:
        # Crée une copie du fichier uploadé
        bytes_data = uploaded_file.getvalue()
        buffer = BytesIO(bytes_data)
        
        decoded_file = TextIOWrapper(buffer, encoding='utf-8')
        df = pd.read_csv(decoded_file)
        
        # Vérification des colonnes
        if not all(col in df.columns for col in COLONNES_REQUISES):
            st.error("Colonnes manquantes dans le CSV !")
            st.write(f"Colonnes requises : {COLONNES_REQUISES}")
            return False
        
        return True
        
    except Exception as e:
        st.error(f"ERREUR : Format de fichier invalide ({str(e)})")
        return False

def main():
    st.title("📈 Stratégie d'Investissement par Valeur Fondamentale")
    
    uploaded_file = st.file_uploader("Téléversez votre fichier historique", type="csv")
    
    if uploaded_file:
        if verifier_csv(uploaded_file):
            # Réinitialise le pointeur du fichier
            bytes_data = uploaded_file.getvalue()
            buffer = BytesIO(bytes_data)
            
            # Lecture définitive du CSV
            df = pd.read_csv(buffer)
            
            # Conversion des nombres
            df["Close"] = df["Close"].str.replace('[^0-9.]', '', regex=True).astype(float)
            
            # Suite du traitement...
            # (Garder le reste de votre logique ici)

if __name__ == "__main__":
    main()
