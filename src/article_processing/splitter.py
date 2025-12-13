# splitter.py - BETenio
import pandas as pd
import os

input_csv = "news_corpus.csv"  # input CSV file
output_dir = "per_model_corpus"  # output directory

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_csv)

if "modelo" not in df.columns:
    raise ValueError("The CSV does not contain the column: 'modelo'")

for modelo, df_modelo in df.groupby("modelo"):
    modelo_str = str(modelo).strip().replace(" ", "_")
    output_path = os.path.join(output_dir, f"{modelo_str}.csv")

    df_modelo.to_csv(output_path, index=False, encoding="utf-8")
print("CSV splitted correctly (DONE).")
