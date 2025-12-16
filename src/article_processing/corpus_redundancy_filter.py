# BETenio - Corpus Redundancy Filter

"""
corpus_redundancy_filter.py
----------------------

General-purpose cleaning script for news articles datasets.

INPUT:
    python corpus_redundancy_filter.py input.csv output.csv

FUNCTIONS:
    - Remove exact duplicates
    - Remove near-duplicates using sentence embeddings (cosine similarity > 0.90)
    - Detect high-frequency n-grams (3-5 grams)
    - Remove articles containing suspicious editorial templates
    - Output a clean CSV ready for ML training

REQUIRES:
    pip install pandas sentence-transformers scikit-learn numpy tqdm torch
"""

import sys
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer, util


NGRAM_RANGE = (3, 5)
MIN_NGRAM_FREQ = 15  # If an n-gram appears in at least this many articles, consider it suspicious (template)
SIMILARITY_THRESHOLD = 0.90


def load_dataset(path):
    df = pd.read_csv(path)
    if "cuerpo" not in df.columns:
        raise ValueError("El CSV debe tener la columna 'cuerpo'.")
    df["cuerpo"] = df["cuerpo"].astype(str)
    return df


def remove_exact_duplicates(df):
    before = len(df)
    df = df.drop_duplicates(subset=["cuerpo"])
    print(f"(DONE) Exact duplicates removed: {before - len(df)}")
    return df


def remove_near_duplicates(df):
    print("Embedding texts (this may take a few minutes)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeds = model.encode(
        df["cuerpo"].tolist(), convert_to_tensor=True, show_progress_bar=True
    )
    n = len(df)

    to_remove = set()

    print("Comparing embeddings for near-duplicates...")
    for i in tqdm(range(n)):
        if i in to_remove:
            continue
        sims = util.cos_sim(embeds[i], embeds)[0]
        near = torch.where(sims > SIMILARITY_THRESHOLD)[0].tolist()
        for j in near:
            if j != i:
                to_remove.add(j)

    print(f"(DONE) Near-duplicates removed: {len(to_remove)}")

    mask = [i not in to_remove for i in range(n)]
    return df.iloc[mask]


def detect_suspicious_ngrams(df):
    print("Extracting high-frequency n-grams...")
    vec = CountVectorizer(ngram_range=NGRAM_RANGE, min_df=MIN_NGRAM_FREQ)
    X = vec.fit_transform(df["cuerpo"])

    counts = np.asarray(X.sum(axis=0)).flatten()
    vocab = vec.get_feature_names_out()

    sorted_idx = counts.argsort()[::-1]
    suspicious = [(vocab[i], counts[i]) for i in sorted_idx[:50]]

    print("\n=== Suspicious N-grams Detected ===")
    for ngram, freq in suspicious[:20]:
        print(f"{ngram}  (freq={freq})")

    suspicious_set = set([ng for ng, f in suspicious])
    return suspicious_set


def remove_articles_with_suspicious_ngrams(df, suspicious_ngrams):
    print("Filtering articles containing suspicious n-grams...")
    mask = []

    for text in df["cuerpo"]:
        lowered = text.lower()
        bad = any(ng.lower() in lowered for ng in suspicious_ngrams)
        mask.append(not bad)

    before = len(df)
    df = df[mask]
    print(f"(DONE) Template-based articles removed: {before - len(df)}")
    return df


def clean_dataset(input_file, output_file):
    print("\n=== CLEANING DATASET ===")
    df = load_dataset(input_file)

    df = remove_exact_duplicates(df)

    # Near-duplicate removal (optional if large dataset)
    # You can comment this out to save time:
    # df = remove_near_duplicates(df)

    suspicious_ngrams = detect_suspicious_ngrams(df)
    df = remove_articles_with_suspicious_ngrams(df, suspicious_ngrams)

    print(f"\n(DONE) Final dataset size: {len(df)} rows")
    df.to_csv(output_file, index=False)
    print(f"(DONE) Saved cleaned dataset to {output_file}")


# ---------------------------------------
#   COMMAND LINE ENTRY
# ---------------------------------------
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("USAGE:")
        print("  python corpus_redundancy_filter.py input.csv output.csv")
        sys.exit(1)

    clean_dataset(sys.argv[1], sys.argv[2])
