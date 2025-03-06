import streamlit as st
import csv
import io
from datetime import datetime  # Ajout pour la gestion des dates

def process_and_save_file(file):
    file_content = file.getvalue().decode("utf-8")
    reader = csv.reader(io.StringIO(file_content), delimiter=',', quotechar='"')

    data = list(reader)
    headers = [col.strip().replace('\ufeff', '') for col in data[0]]
    rows = data[1:]

    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    vol_col = "Vol."
    date_col = "Date"  # Colonne date à adapter

    # Trouver les indices des colonnes
    col_indices = {col: headers.index(col) for col in numeric_cols if col in headers}
    vol_index = headers.index(vol_col) if vol_col in headers else None
    date_index = headers.index(date_col) if date_col in headers else None

    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)  # Syntaxe corrigée

    writer.writerow(headers)
    for row in rows:
        new_row = row.copy()

        # 1. Conversion des nombres (BRVM utilise souvent des virgules)
        for col, idx in col_indices.items():
            new_row[idx] = new_row[idx].replace('.', ',')  # Conservez cette ligne si nécessaire

        # 2. Conversion du volume (ex: "1,5K" → 1500)
        if vol_index is not None and 'K' in new_row[vol_index]:
            vol_value = new_row[vol_index].replace('K', '').replace(',', '.')
            new_row[vol_index] = str(int(float(vol_value) * 1000))

        # 3. Correction des dates (ex: "01/01/2020" → "2020-01-01")
        if date_index is not None:
            try:
                original_date = datetime.strptime(new_row[date_index], "%d/%m/%Y")
                new_row[date_index] = original_date.strftime("%Y-%m-%d")  # Format universel
            except ValueError:
                pass  # Gestion d'erreur simple

        writer.writerow(new_row)

    return output.getvalue()

def main():
    st.set_page_config(page_title="BRVM Backtester", layout="wide")  # Configuration Streamlit
    
    st.title("🚀 Backtesting Automatique BRVM")
    st.markdown("""
    *Hébergé sur [GitHub](https://github.com/votrecompte/votrerepo) | Déployé avec [Streamlit Cloud](https://streamlit.io/cloud)*
    """)

    uploaded_file = st.file_uploader("Importer votre fichier CSV Historique", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_csv_content = process_and_save_file(uploaded_file)
            
            st.success("✅ Fichier corrigé avec succès !")
            st.download_button(
                label="📥 Télécharger le CSV corrigé",
                data=new_csv_content,
                file_name="BRVM_Backtest_Ready.csv",
                mime="text/csv"
            )

            # Section backtesting (à compléter avec votre logique)
            st.subheader("Paramètres de Backtesting")
            start_date = st.date_input("Date de début", value=datetime(2020, 1, 1))
            end_date = st.date_input("Date de fin", value=datetime.today())

            if st.button("Lancer le Backtest"):
                # Insérez ici votre logique de backtesting
                st.write("🔍 Backtesting en cours...")

        except Exception as e:
            st.error(f"❌ Erreur critique : {str(e)}")

if __name__ == '__main__':
    main()