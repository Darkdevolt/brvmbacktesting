import csv

COLONNES_REQUISES = [
    "Exchange Date", "Close", "Net", "%Chg", "Open", "Low", "High", 
    "Volume", "Turnover - XOF", "Flow"
]

def verifier_format_csv(fichier_csv):
    """Vérifie si le fichier CSV a le bon format."""
    try:
        with open(fichier_csv, 'r', encoding='utf-8') as f:
            lecteur = csv.reader(f)
            en_tete = next(lecteur, None)
            if en_tete != COLONNES_REQUISES:
                return False
        return True
    except FileNotFoundError:
        raise FileNotFoundError("Fichier non trouvé !")
    except Exception as e:
        raise ValueError(f"Erreur de lecture : {str(e)}")

def analyser_achats(fichier_csv, valeur_fondamentale):
    """Analyse le CSV et suggère des achats si Close < Valeur Fondamentale."""
    if not verifier_format_csv(fichier_csv):
        print("Erreur : Format de fichier invalide. Colonnes requises :")
        print(", ".join(COLONNES_REQUISES))
        return

    with open(fichier_csv, 'r', encoding='utf-8') as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            try:
                date = ligne["Exchange Date"]
                close = float(ligne["Close"].replace(',', '').strip())
                turnover = float(ligne["Turnover - XOF"].replace(',', '').strip())

                if close < valeur_fondamentale:
                    print(f"\n📅 Date : {date}")
                    print(f"💵 Close : {close} XOF | Valeur Fondamentale : {valeur_fondamentale} XOF")
                    print("🔍 Opportunité d'achat détectée !")
                    
                    while True:
                        montant = input("Entrez le montant (≥ 10 000 XOF) ou 0 pour annuler : ")
                        if montant.replace('.', '', 1).isdigit():
                            montant = float(montant)
                            if montant == 0:
                                print("❌ Achat annulé.")
                                break
                            elif montant >= 10000:
                                print(f"✅ Achat de {montant} XOF confirmé !")
                                break
                            else:
                                print("⚠️ Le montant doit être ≥ 10 000 XOF ou 0.")
                        else:
                            print("⚠️ Entrée invalide. Utilisez un nombre.")

            except (KeyError, ValueError) as e:
                print(f"⚠️ Ligne ignorée (erreur : {str(e)}) : {ligne}")

if __name__ == "__main__":
    try:
        fichier = input("Chemin du fichier CSV (ex: NLTC.csv) : ")
        vf = float(input("Valeur fondamentale (XOF) : "))
        analyser_achats(fichier, vf)
    except ValueError:
        print("❌ La valeur fondamentale doit être un nombre positif.")
    except FileNotFoundError:
        print("❌ Fichier CSV introuvable.")
