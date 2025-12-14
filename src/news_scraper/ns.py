import os
import argparse
import logging
import requests
from time import sleep
from datetime import datetime
from bs4 import BeautifulSoup
from readability import Document
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ethical article scraper with full audit logging"
    )
    parser.add_argument("--urls", required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--html-dir", default="html")
    parser.add_argument("--text-dir", default="articles")
    parser.add_argument("--log", default="scraping.log")
    parser.add_argument("--sleep", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=15)
    return parser.parse_args()


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_urls(path, start, end):
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()[start:end]


def fetch_html(url, headers, timeout, retries, retry_sleep):
    for attempt in range(1, retries + 1):
        logging.info("HTTP request attempt %d/%d | URL: %s", attempt, retries, url)
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            logging.info(
                "HTTP request succeeded | Status: %d | URL: %s", response.status_code, url
            )
            return response.text
        except Exception as e:
            logging.warning(
                "HTTP request failed | Attempt %d | URL: %s | Error: %s",
                attempt, url, e
            )
            if attempt < retries:
                logging.info("Sleeping %ds before retry | URL: %s", retry_sleep, url)
                sleep(retry_sleep)
            else:
                logging.error("All retries exhausted | URL: %s", url)
                raise


def extract_article_text(html):
    doc = Document(html)
    content_html = doc.summary(html_partial=True)
    soup = BeautifulSoup(content_html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(" ")
    return " ".join(text.split())


def save_html(html, directory, article_id):
    path = os.path.join(directory, f"{article_id}.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info("HTML saved | ID: %s | Path: %s", article_id, path)


def save_text(text, directory, article_id):
    path = os.path.join(directory, f"{article_id}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logging.info("Article text saved | ID: %s | Characters: %d", article_id, len(text))


def process_single_url(url, article_id, headers, args):
    logging.info("Processing started | ID: %s | URL: %s", article_id, url)

    html = fetch_html(url, headers, args.timeout, args.retries, args.retry_sleep)

    save_html(html, args.html_dir, article_id)

    article_text = extract_article_text(html)

    save_text(article_text, args.text_dir, article_id)

    logging.info("Processing completed | ID: %s | URL: %s", article_id, url)


def run_scraping_session(args):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; AcademicResearchBot/1.0)"}

    os.makedirs(args.html_dir, exist_ok=True)
    os.makedirs(args.text_dir, exist_ok=True)

    urls = load_urls(args.urls, args.start, args.end)

    logging.info(
        "Scraping session started | URLs: %s | Range: %d–%d | Count: %d",
        args.urls, args.start, args.end, len(urls)
    )
    logging.info(
        "Configuration | sleep=%ds | retries=%d | retry_sleep=%ds | timeout=%ds",
        args.sleep, args.retries, args.retry_sleep, args.timeout
    )

    start_time = datetime.now()

    for i, url in enumerate(tqdm(urls, desc="Scraping articles")):
        article_id = f"{args.start + i:06d}"

        try:
            process_single_url(url, article_id, headers, args)
            sleep(args.sleep)
        except Exception as e:
            logging.error(
                "Processing failed | ID: %s | URL: %s | Error: %s",
                article_id, url, e
            )

    duration = datetime.now() - start_time
    logging.info("Scraping session finished | Duration: %s", duration)


def main():
    args = parse_args()
    setup_logging(args.log)

    logging.info("=" * 80)
    run_scraping_session(args)
    logging.info("=" * 80)

    print("Done. See log for full details.")


if __name__ == "__main__":
    main()
