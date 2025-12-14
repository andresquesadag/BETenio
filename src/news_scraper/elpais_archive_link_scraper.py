#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
elpais_archive_scraper.py

Scrapes EL PAÍS hemeroteca for daily news links.

- Iterates over a date range (e.g. 2015-01-01 to 2020-12-31)
- For each day, opens: https://elpais.com/hemeroteca/YYYY-MM-DD/
- Extracts candidate article links and titles
- Saves to a CSV: date, title, url

Requirements:
    pip install requests beautifulsoup4 pandas tqdm
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import time

# ---------------- CONFIG ----------------

START_DATE = datetime(2015, 1, 1)
END_DATE = datetime(2020, 12, 31)

BASE_URL = "https://elpais.com/hemeroteca/{date}/"

OUTPUT_CSV = "elpais_links_2015_2020.csv"

REQUEST_TIMEOUT = 10
SLEEP_BETWEEN_REQUESTS = 1.0  # seconds (ser respetuosos)

# ----------------------------------------


def iter_dates(start: datetime, end: datetime):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def fetch_day(date_obj: datetime):
    """Fetch HTML for a given date from EL PAÍS hemeroteca."""
    date_str = date_obj.strftime("%Y-%m-%d")
    url = BASE_URL.format(date=date_str)

    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0 (news research scraper)"},
        )
    except Exception as e:
        print(f"[{date_str}] Request error:", e)
        return None, None

    if resp.status_code != 200:
        # 404 or similar: no archive page (or blocked)
        return None, None

    return date_str, resp.text


def extract_links(date_str: str, html: str):
    """
    Extract article links and titles from a hemeroteca HTML page.

    Nota: La estructura HTML puede cambiar, así que esto es heurístico.
    Lo importante es quedarnos con:
        - enlaces dentro del contenido central
        - textos suficientemente largos para ser titulares
        - URLs de elpais.com (no plus.elpais.com, ni hemeroteca, etc.)
    """
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # Heurística: muchos titulares están en h2/h3 con un <a> dentro.
    # Si eso falla, también miramos <a> largos en otros lugares.
    candidate_links = []

    # 1) h2/h3 > a
    for tag_name in ("h2", "h3"):
        for h in soup.find_all(tag_name):
            a = h.find("a", href=True)
            if a is not None:
                candidate_links.append(a)

    # 2) Si no encontramos nada, fallback: todos los <a>
    if not candidate_links:
        candidate_links = soup.find_all("a", href=True)

    for a in candidate_links:
        title = a.get_text(strip=True)
        href = a["href"]

        if not title or len(title) < 15:
            # descartamos títulos demasiado cortos tipo "España" o "Leer más"
            continue

        # Normalizar URL relativa
        if href.startswith("/"):
            href_full = "https://elpais.com" + href
        else:
            href_full = href

        # Filtrar URLs claramente no-noticia
        if "hemeroteca" in href_full:
            continue
        if "plus.elpais.com" in href_full:
            continue
        if "newsletter" in href_full:
            continue
        if "colecciones.elpais.com" in href_full:
            continue

        # Nos quedamos solo con dominio elpais.com (no cincodias, motor, etc. si quieres ser estricto)
        if not href_full.startswith("https://elpais.com"):
            continue

        results.append(
            {
                "date": date_str,
                "title": title,
                "url": href_full,
            }
        )

    return results


def main():
    print("=== EL PAÍS HEMEROTECA SCRAPER ===")
    print(f"Date range: {START_DATE.date()} -> {END_DATE.date()}")
    print("This will collect daily news links from El País.\n")

    all_rows = []

    for d in tqdm(list(iter_dates(START_DATE, END_DATE))):
        date_str, html = fetch_day(d)
        if html is None:
            # Could be no archive, 404, or request error
            continue

        day_links = extract_links(date_str, html)
        if day_links:
            all_rows.extend(day_links)

        # Respetar el servidor
        time.sleep(SLEEP_BETWEEN_REQUESTS)

    if not all_rows:
        print("No links collected. Check network or HTML structure.")
        return

    df = pd.DataFrame(all_rows)

    # Eliminar duplicados por URL
    before = len(df)
    df = df.drop_duplicates(subset=["url"])
    print(f"\nRemoved {before - len(df)} duplicate URLs.")

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved {len(df)} links to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
