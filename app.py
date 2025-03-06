import streamlit as st
import csv
import io
from datetime import datetime
import dateparser  # Nouvelle librairie pour parser les dates

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
            value = new_row[idx].strip()
            value = value.replace(' ', '').replace('.', '').replace(',', '.')
            new_row[idx] = value

        # Conversion des volumes
        if vol_index is not None:
            vol_str = new_row[vol_index].strip().upper()
            vol_str = vol_str.replace(' ', '').replace(',', '.')
            if 'K' in vol_str:
                new_row[vol_index] = str(int(float(vol_str.replace('K', '')) * 1000))
            elif 'M' in vol_str:
                new_row[vol_index] = str(int(float(vol_str.replace('M', '')) * 1000000))

        # Conversion des dates (version améliorée)
        if date_index is not None:
            date_str = new_row[date_index].strip()
            try:
                # Essayer avec dateparser pour les formats complexes
                parsed_date = dateparser.parse(
                    date_str,
                    date_formats=['%d/%m/%Y', '%d-%b-%Y', '%d %B %Y'],
                    languages=['fr']
                )
                
                if parsed_date:
                    new_row[date_index] = parsed_date.strftime("%Y-%m-%d")
                else:
                    # Log des dates non reconnues
                    st.warning(f"Format de date non reconnu : {date_str}")
            except Exception as e:
                st.error(f"Erreur de conversion pour {date_str}: {str(e)}")

        writer.writerow(new_row)

    return output.getvalue()

def main():
    st.set_page_config(page_title="BRVM Backtester", layout="wide")
    
    st.title("🚀 Backtesting Automatique BRVM")
    
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

            # Affichage des résultats
            st.subheader("Vérification des dates")
            preview_df = pd.read_csv(io.StringIO(new_csv_content))
            
            # Vérification du format des dates
            st.write("Exemples de dates converties :")
            st.write(preview_df[['Date']].head(3).to_markdown(index=False))
            
            # Vérification de l'intervalle temporel
            start_date = pd.to_datetime(preview_df['Date']).min()
            end_date = pd.to_datetime(preview_df['Date']).max()
            st.write(f"📅 Plage temporelle détectée : {start_date.strftime('%Y-%m-%d')} au {end_date.strftime('%Y-%m-%d')}")

        except Exception as e:
            st.error(f"❌ Erreur critique : {str(e)}")

if __name__ == '__main__':
    main()