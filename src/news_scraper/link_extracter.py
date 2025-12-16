import csv

input_csv = "elpais_links_2015_2020.csv"
output_txt = "urls.txt"
column_name = "url"

with open(input_csv, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)

    if column_name not in reader.fieldnames:
        raise ValueError(
            f"There is no '{column_name}' column. Found columns: {reader.fieldnames}"
        )

    with open(output_txt, "w", encoding="utf-8") as txtfile:
        for row in reader:
            url = row[column_name].strip()
            if url:
                txtfile.write(url + "\n")

print("Done. Check urls.txt for the extracted URLs.")
