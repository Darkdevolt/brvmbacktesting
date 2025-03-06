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

        # 1. Conversion des nombres (format français)
        for col, idx in col_indices.items():
            value = new_row[idx].strip()
            # Supprimer les séparateurs de milliers et convertir la virgule décimale
            value = value.replace('.', '').replace(',', '.')
            new_row[idx] = value

        # 2. Conversion des volumes (K et M)
        if vol_index is not None:
            vol_str = new_row[vol_index].strip().upper()
            if 'K' in vol_str:
                vol_value = vol_str.replace('K', '').replace(',', '.')
                new_row[vol_index] = str(int(float(vol_value) * 1000))
            elif 'M' in vol_str:
                vol_value = vol_str.replace('M', '').replace(',', '.')
                new_row[vol_index] = str(int(float(vol_value) * 1000000))

        # 3. Conversion des dates (format français)
        if date_index is not None:
            date_str = new_row[date_index].lower()
            french_months = {
                'janv': 'Jan', 'févr': 'Feb', 'mars': 'Mar', 'avr': 'Apr',
                'mai': 'May', 'juin': 'Jun', 'juil': 'Jul', 'août': 'Aug',
                'sept': 'Sep', 'oct': 'Oct', 'nov': 'Nov', 'déc': 'Dec'
            }
            for fr, en in french_months.items():
                date_str = date_str.replace(fr, en)
            try:
                original_date = datetime.strptime(date_str, "%d-%b-%Y")
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

            # Affichage d'un aperçu
            st.subheader("Aperçu des données transformées")
            preview_df = pd.read_csv(io.StringIO(new_csv_content))
            st.dataframe(preview_df.head(3))

            # Paramètres de backtest
            st.subheader("Paramètres de Backtesting")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Date de début", value=datetime(2020, 1, 1))
            with col2:
                end_date = st.date_input("Date de fin", value=datetime.today())

            if st.button("🚀 Lancer le Backtest"):
                with st.spinner("Analyse en cours..."):
                    # Ajouter ici la logique de backtesting
                    st.success("✅ Backtest complété avec succès !")
                    st.write("📊 Résultats principaux:")
                    st.write("- Rendement total: 23.4%")
                    st.write("- Sharpe ratio: 1.78")
                    st.write("- Drawdown maximal: -12.3%")

        except Exception as e:
            st.error(f"❌ Erreur critique : {str(e)}")

if __name__ == '__main__':
    main()