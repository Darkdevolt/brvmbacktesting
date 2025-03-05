import streamlit as st
import csv
import io

def process_and_save_file(file):
    file_content = file.getvalue().decode("utf-8")
    reader = csv.reader(io.StringIO(file_content), delimiter=',', quotechar='"')

    # Convertir le CSV en liste de listes
    data = list(reader)

    # Nettoyage des en-têtes
    headers = [col.strip().replace('\ufeff', '') for col in data[0]]
    rows = data[1:]  # Toutes les lignes sauf l'en-tête

    # Trouver les indices des colonnes concernées
    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    vol_col = "Vol."

    col_indices = {col: headers.index(col) for col in numeric_cols if col in headers}
    vol_index = headers.index(vol_col) if vol_col in headers else None

    # Création du nouveau fichier CSV en mémoire
    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Réécriture des données avec les bonnes valeurs
    writer.writerow(headers)  # Écrire l'en-tête
    for row in rows:
        new_row = row.copy()  # Copier la ligne originale

        # Modifier les colonnes numériques : remplacer "." par ","
        for col, idx in col_indices.items():
            new_row[idx] = new_row[idx].replace('.', ',')  # Remplacer le point par une virgule

        # Modifier la colonne "Vol." : enlever 'K' et multiplier par 1000
        if vol_index is not None and 'K' in new_row[vol_index]:
            new_row[vol_index] = str(int(float(new_row[vol_index].replace('K', '').replace(',', '.')) * 1000))

        writer.writerow(new_row)  # Écrire la ligne modifiée

    return output.getvalue()  # Retourner le contenu du fichier corrigé

def main():
    st.title("Traitement et Correction de CSV")
    uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_csv_content = process_and_save_file(uploaded_file)
            
            # Bouton de téléchargement
            st.download_button(
                label="Télécharger le fichier corrigé",
                data=new_csv_content,
                file_name="fichier_corrige.csv",
                mime="text/csv"
            )

            st.success("Le fichier est prêt à être téléchargé !")

        except Exception as e:
            st.error(f"Erreur lors du traitement : {e}")

if __name__ == '__main__':
    main()