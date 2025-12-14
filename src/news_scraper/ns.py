import os
import logging
import requests
from time import sleep
from datetime import datetime
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm

URLS_FILE = "urls.txt"

START_LINE = 0
END_LINE = 10

HTML_DIR = "news_data/htmls"
TEXT_DIR = "news_data/articles"
LOG_FILE = "news_data/ns.log"

REQUEST_TIMEOUT = 15
SLEEP_BETWEEN_REQUESTS = 1
RETRIES = 3
RETRY_SLEEP = 5

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; NewsScraperForBETenio/2.0)"}

os.makedirs(HTML_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.info("=" * 60)
logging.info("Block start | lines %d–%d", START_LINE, END_LINE)
start_time = datetime.now()

with open(URLS_FILE, encoding="utf-8") as f:
    urls = f.read().splitlines()[START_LINE:END_LINE]


def fetch_html(url):
    last_exception = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.text
        except Exception as e:
            last_exception = e
            logging.warning("Retry %d/%d | %s | %s", attempt, RETRIES, url, e)
            sleep(RETRY_SLEEP)
    raise last_exception


def extract_article(html):
    doc = Document(html)
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    lines = [line.strip() for line in soup.get_text("\n").splitlines() if line.strip()]
    return "\n".join(lines)


def file_id(i):
    return f"{START_LINE + i:06d}"


for i, url in enumerate(tqdm(urls, desc="Processing articles")):
    idx = file_id(i)

    try:
        html = fetch_html(url)

        with open(os.path.join(HTML_DIR, f"{idx}.html"), "w", encoding="utf-8") as f:
            f.write(html)

        article = extract_article(html)

        with open(os.path.join(TEXT_DIR, f"{idx}.txt"), "w", encoding="utf-8") as f:
            f.write(article)

        logging.info("OK | %s | %s", idx, url)
        sleep(SLEEP_BETWEEN_REQUESTS)

    except Exception as e:
        logging.error("FAILED | %s | %s | %s", idx, url, e)

end_time = datetime.now()
duration = end_time - start_time

logging.info("Block end | duration %s", duration)
logging.info("=" * 60)

print("Done. See scraping.log for details.")
