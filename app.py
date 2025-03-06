import streamlit as st
import csv
import io
from datetime import datetime

def process_and_save_file(file):
    file_content = file.getvalue().decode("utf-8")
    reader = csv.reader(io.StringIO(file_content), delimiter=',', quotechar='"')

    data = list(reader)
    headers = [col.strip().replace('\ufeff', '') for col in data[0]]
    rows = data[1:]

    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    vol_col = "Vol."
    date_col = "Date"

    col_indices = {col: headers.index(col) for col in numeric_cols if col in headers}
    vol_index = headers.index(vol_col) if vol_col in headers else None
    date_index = headers.index(date_col) if date_col in headers else None

    output = io.StringIO()
    writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    writer.writerow(headers)
    for row in rows:
        new_row = row.copy()

        # Conversion des nombres
        for col, idx in col_indices.items():
            new_row[idx] = new_row[idx].replace('.', ',')

        # Conversion des volumes
        if vol_index is not None and 'K' in new_row[vol_index]:
            vol_value = new_row[vol_index].replace('K', '').replace(',', '.')
            new_row[vol_index] = str(int(float(vol_value) * 1000))

        # Conversion des dates (conservée mais sans sélection dans l'interface)
        if date_index is not None:
            try:
                original_date = datetime.strptime(new_row[date_index], "%d/%m/%Y")
                new_row[date_index] = original_date.strftime("%Y-%m-%d")
            except ValueError:
                pass

        writer.writerow(new_row)

    return output.getvalue()

def main():
    st.set_page_config(page_title="BRVM Backtester", layout="wide")
    
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

            # Section backtesting (sans sélection de date)
            st.subheader("Backtesting")
            if st.button("Lancer le Backtest"):
                # Votre logique de backtesting ici
                st.write("🔍 Backtesting en cours...")
                # Exemple de résultat (à adapter avec votre logique)
                st.write("📊 Résultats du backtesting :")
                st.write("- Rendement total : 15.3%")
                st.write("- Sharpe ratio : 1.2")
                st.write("- Drawdown maximal : -8.5%")

        except Exception as e:
            st.error(f"❌ Erreur critique : {str(e)}")

if __name__ == '__main__':
    main()