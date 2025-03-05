from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd

# Lancez un navigateur, par exemple Chrome via le driver
driver = webdriver.Chrome()

# Accéder à la page
driver.get("URL_de_la_page")

# Patientez le temps que le JavaScript charge la table (éventuellement un WebDriverWait)
table = driver.find_element(By.TAG_NAME, "table")

# Récupérer le HTML de la table
html_table = table.get_attribute('outerHTML')

# Parsez le HTML récupéré
df = pd.read_html(html_table)[0]
print(df.head())

driver.quit()