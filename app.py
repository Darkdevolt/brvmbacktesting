import streamlit as st
import pandas as pd
from io import BytesIO

# Titre de l'application
st.title("Correction et Traitement de Fichiers CSV/Excel")

# Téléverser un fichier
uploaded_file = st.file_uploader("Téléversez un fichier CSV ou Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Lire le fichier
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        # Afficher les premières lignes du fichier original
        st.subheader("Aperçu du fichier original")
        st.write(df.head())

        # Correction des données
        st.subheader("Correction des données")

        # Colonnes à traiter (supprimer le point et convertir en entiers)
        columns_to_convert = ['Dernier', 'Ouv.', 'Plus_Haut', 'Plus_Bas']
        for col in columns_to_convert:
            if col in df.columns:
                # Supprimer le point et convertir en entiers
                df[col] = df[col].astype(str).str.replace('.', '').astype(int)

        # Convertir la colonne 'Vol.' en nombre (supprimer 'K' et multiplier par 1000)
        if 'Vol.' in df.columns:
            # Nettoyer la colonne 'Vol.' : supprimer 'K' et remplacer ',' par '.'
            df['Vol.'] = df['Vol.'].astype(str).str.replace('K', '').str.replace(',', '.').astype(float) * 1000
            df['Vol.'] = df['Vol.'].astype(int)  # Convertir en entier

        # Convertir la colonne 'Date' en type datetime
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

        # Afficher les premières lignes du fichier corrigé
        st.subheader("Aperçu du fichier corrigé")
        st.write(df.head())

        # Télécharger le fichier corrigé
        st.subheader("Télécharger le fichier corrigé")
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Feuille1')
        output.seek(0)

        st.download_button(
            label="Télécharger le fichier corrigé (Excel)",
            data=output,
            file_name="fichier_corrige.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Option pour continuer à travailler avec le fichier corrigé
        if st.checkbox("Continuer à travailler avec le fichier corrigé"):
            st.subheader("Travaillez avec le fichier corrigé")
            st.write("Vous pouvez maintenant effectuer des analyses supplémentaires ici.")
            # Exemple : Afficher des statistiques descriptives
            st.write("Statistiques descriptives :")
            st.write(df.describe())

            # Exemple : Filtrer les données
            st.write("Filtrer les données :")
            if 'Vol.' in df.columns:
                min_volume = st.slider("Volume minimum", min_value=int(df['Vol.'].min()), max_value=int(df['Vol.'].max()))
                filtered_df = df[df['Vol.'] >= min_volume]
                st.write(filtered_df)

    except Exception as e:
        st.error(f"Une erreur s'est produite : {e}")
