import streamlit as st
import csv
import io

def process_and_save_file(file):
    file_content = file.getvalue().decode("utf-8")
    reader = csv.reader(io.StringIO(file_content), delimiter=',', quotechar='"')

    data = list(reader)
    headers = [col.strip().replace('\ufeff', '') for col in data[0]]
    rows = data[1:]

    numeric_cols = ["Dernier", "Ouv.", "Plus Haut", "Plus Bas"]
    vol_col = "Vol."

    col_indices = {col: headers.index(col) for col in numeric_cols if col in headers}
    vol_index = headers.index(vol_col) if vol_col in headers else None

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
                new_row[vol_index] = str(int(float(vol_str.replace('K', '')) * 1000)
            elif 'M' in vol_str:
                new_row[vol_index] = str(int(float(vol_str.replace('M', '')) * 1000000)

        writer.writerow(new_row)

    return output.getvalue()

def main():
    st.set_page_config(page_title="BRVM Data Cleaner", layout="wide")
    
    st.title("🧹 Nettoyeur de Données BRVM")
    st.markdown("""
    **Fonctionnalités exclusives :**
    - Conversion des formats numériques français → international
    - Transformation automatique des volumes (K/M → chiffres)
    """)

    uploaded_file = st.file_uploader("Importer votre fichier CSV BRVM", type=["csv"])
    
    if uploaded_file is not None:
        try:
            new_csv_content = process_and_save_file(uploaded_file)
            
            st.success("✅ Fichier transformé avec succès !")
            st.download_button(
                label="📥 Télécharger le CSV corrigé",
                data=new_csv_content,
                file_name="BRVM_Data_Cleaned.csv",
                mime="text/csv"
            )

            # Aperçu basique
            st.subheader("Aperçu du résultat")
            st.code("\n".join(new_csv_content.split('\n')[:5]))

        except Exception as e:
            st.error(f"❌ Erreur détectée : {str(e)}")

if __name__ == '__main__':
    main()