import pandas as pd
import json

CSV_PATH = r"C:\cnn\clothing-dataset\images.csv"
OUTPUT_JSON = "classes.json"

# Carregar o CSV
df = pd.read_csv(CSV_PATH)

# Pegar classes únicas da coluna "label"
classes_unicas = sorted(df["label"].unique())

# Criar mapa ID -> Classe
id_para_classe = {str(i): classe for i, classe in enumerate(classes_unicas)}

# Salvar JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(id_para_classe, f, indent=4)

print("Arquivo classes.json criado!")
print(id_para_classe)
