import pandas as pd
import streamlit as st

def process_csv(file):
    # Lire le fichier CSV
    df = pd.read_csv(file)

    # Traiter les données
    # Exemple: Multiplier les valeurs de la colonne 'Vol.' par 1000 si elles contiennent 'K'
    if 'Vol.' in df.columns:
        df['Vol.'] = df['Vol.'].apply(lambda x: float(x.replace('K', '')) * 1000 if 'K' in str(x) else float(x))

    return df

def main():
    st.title("Traitement de fichiers CSV")

    # Upload du fichier CSV
    uploaded_file = st.file_uploader("Téléchargez votre fichier CSV", type=["csv"])

    if uploaded_file is not None:
        # Traiter le fichier
        df = process_csv(uploaded_file)

        # Afficher les données traitées
        st.write("Données traitées:")
        st.write(df)

        # Option pour télécharger les données traitées
        st.download_button(
            label="Télécharger les données traitées",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='donnees_traitees.csv',
            mime='text/csv',
        )

if __name__ == "__main__":
    main()