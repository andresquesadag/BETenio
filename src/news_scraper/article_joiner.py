import csv
from pathlib import Path


URLS_FILE = "urls.txt"
ARTICLES_DIR = Path("news_data/articles")
OUTPUT_FILE = "human_es_news.csv"

MODEL_LABEL = "Reales"
AUTHORSHIP = "Humano"


with open(URLS_FILE, "r", encoding="utf-8") as f:
    urls = [line.strip() for line in f if line.strip()]


rows = []
missing_txt = 0

for index, _ in enumerate(urls):
    txt_filename = f"{index:06d}.txt"
    txt_path = ARTICLES_DIR / txt_filename

    if not txt_path.exists():
        missing_txt += 1
        continue

    text = txt_path.read_text(encoding="utf-8").strip()
    rows.append([text, MODEL_LABEL, AUTHORSHIP])


with open(OUTPUT_FILE, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

    writer.writerow(["cuerpo", "modelo", "authorship"])
    writer.writerows(rows)


print("Dataset successfully created.")
print("Total URLs:", len(urls))
print("Rows written:", len(rows))
print("Missing .txt files:", missing_txt)
