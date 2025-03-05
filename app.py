import streamlit as st
import csv
import io

def process_data(file):
    # Lire le fichier CSV sans transformation automatique
    file_content = file.getvalue().decode("utf-8")
    reader = csv.reader(io.StringIO(file_content), delimiter=',', quotechar='"')

    # Convertir le CSV en liste de listes
    data = list(reader)

    # Nettoyage des en-têtes
    headers = [col.strip().replace('\ufeff', '') for col in data[0]]
    rows = data[1:]  # Toutes les lignes sauf l'en-tête

    # Remplacement du séparateur dans les colonnes concernées
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    vol_col = "Vol."

    # Trouver les indices des colonnes concernées
    col_indices = {col: headers.index(col) for col in numeric_cols if col in headers}
    vol_index = headers.index(vol_col) if vol_col in headers else None

    # Traiter chaque ligne
    for row in rows:
        # Remplacement des points par des virgules dans les colonnes numériques
        for col, idx in col_indices.items():
            row[idx] = row[idx].replace('.', ',')  # Remplacer . par ,

        # Convertir les volumes en supprimant 'K' et en multipliant par 1000
        if vol_index is not None:
            if 'K' in row[vol_index]:
                row[vol_index] = str(int(float(row[vol_index].replace('K', '').replace(',', '.')) * 1000))

    return [headers] + rows  # Retourner les données avec l'en-tête

def main():
    st.title("Traitement de CSV - Application BRVM")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            processed_data = process_data(uploaded_file)
            
            # Afficher le tableau formaté
            st.write("Données traitées :")
            st.dataframe(processed_data)
            
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")

if __name__ == '__main__':
    main()