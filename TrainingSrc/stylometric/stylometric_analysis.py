"""
Stylometric analysis (advanced) for Spanish news dataset.

- Works directly on the CSV dataset (train or test).
- Does NOT use BETenio; it only looks at the raw texts.

Features included:
  * Lexical metrics:
      - word_count, unique_word_count, char_count
      - avg_word_length, TTR, hapax_ratio
  * Sentence metrics:
      - sentence_count, avg_sentence_length
  * POS-based metrics (using spaCy es_core_news_md):
      - noun_ratio, verb_ratio, adj_ratio, adv_ratio
      - pronoun_ratio, first_person_pronoun_ratio
      - syntax_variety (number of distinct POS / tokens)
  * Stopword metrics:
      - stopword_ratio
  * Complexity metrics (approximate):
      - clause_token_ratio (tokens that look like clause markers)
      - complex_sentence_ratio (sentences with at least one clause marker)
  * Punctuation metrics:
      - punctuation_ratio
      - question_mark_ratio, exclamation_mark_ratio
      - comma_ratio, semicolon_ratio, colon_ratio, dash_ratio
  * N-gram analysis (bigrams + trigrams):
      - log-odds ranking of n-grams per class (IA vs Human)

Outputs:
  stylometry_outputs/advanced_features.csv
  stylometry_outputs/advanced_tests.csv
  stylometry_outputs/advanced_ngrams.csv

Usage (from TrainingSrc/):

  python -m stylometric.stylometric_advanced train_dataset.csv

"""

import os
import sys
import math
import string
import re
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
import spacy
from sklearn.feature_extraction.text import CountVectorizer

# spaCy model name for Spanish
SPACY_MODEL = "es_core_news_md"

# Output directory for all stylometric results
OUTPUT_DIR = "stylometry_outputs"

# Text + label column defaults for your dataset
DEFAULT_TEXT_COLUMNS = ("titular", "cuerpo")  # will be concatenated if both exist
DEFAULT_LABEL_COLUMN = "authorship"  # "IA" / "Humano"


# Utility functions


def load_spacy_model():
    """Load spaCy Spanish model."""
    print(f"Loading spaCy model: {SPACY_MODEL}")
    try:
        nlp = spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{SPACY_MODEL}' is not installed. "
            f"Run: python -m spacy download {SPACY_MODEL}"
        ) from exc
    return nlp


def build_text_column(df: pd.DataFrame, text_columns: Tuple[str, ...]) -> pd.Series:
    """
    Build the text column used for analysis.

    By default, concatenates 'titular' and 'cuerpo' if both exist,
    otherwise falls back to whichever is available.
    """
    available = [col for col in text_columns if col in df.columns]
    if not available:
        raise ValueError(
            f"None of the expected text columns {text_columns} are present in the dataset."
        )

    if len(available) == 1:
        print(f"Using column '{available[0]}' as text.")
        return df[available[0]].fillna("").astype(str)

    # Concatenate headline + body
    print(f"Using columns {available} concatenated as text.")
    text = df[available[0]].fillna("").astype(str)
    for col in available[1:]:
        text = text + " " + df[col].fillna("").astype(str)
    return text


def detect_labels(df: pd.DataFrame, label_col: str) -> Tuple[str, str]:
    """
    Detect IA and Human labels automatically.
    """
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in dataset.")

    labels = df[label_col].dropna().unique().tolist()
    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 labels in '{label_col}', found: {labels}")

    ia_label = None
    for lab in labels:
        if "IA" in str(lab).upper():
            ia_label = lab
            break
    if ia_label is None:
        ia_label = labels[0]  # fallback

    human_label = [lab for lab in labels if lab != ia_label][0]

    print(f"Detected labels → IA: '{ia_label}' | Human: '{human_label}'")
    return ia_label, human_label


# Feature extraction per document

FIRST_PERSON_PRONOUNS = {
    "yo",
    "nosotros",
    "nosotras",
    "me",
    "nos",
    "mí",
    "mío",
    "mía",
    "míos",
    "mías",
    "nuestro",
    "nuestra",
    "nuestros",
    "nuestras",
}

CLAUSE_DEPS = {
    "advcl",
    "csubj",
    "csubj:pass",
    "ccomp",
    "xcomp",
    "acl",
    "acl:relcl",
    "parataxis",
}


def compute_features_for_doc(doc: spacy.tokens.Doc) -> Dict[str, float]:
    """
    Compute all lexical, POS, punctuation and complexity features for a single spaCy Doc.
    """

    tokens = [t for t in doc if not t.is_space]
    words = [t for t in tokens if t.is_alpha]
    n_tokens = len(tokens)
    n_words = len(words)

    # Lexical / counts
    word_count = n_words
    unique_words = len(set(w.lemma_.lower() for w in words)) if n_words > 0 else 0
    char_count = sum(len(w.text) for w in words)

    avg_word_length = char_count / n_words if n_words > 0 else 0.0
    ttr = unique_words / n_words if n_words > 0 else 0.0

    # Hapax: words that appear only once in the doc
    if n_words > 0:
        freqs = {}
        for w in words:
            key = w.lemma_.lower()
            freqs[key] = freqs.get(key, 0) + 1
        hapax = sum(1 for f in freqs.values() if f == 1)
        hapax_ratio = hapax / n_words
    else:
        hapax_ratio = 0.0

    # Sentences
    sentences = list(doc.sents)
    sentence_count = len(sentences)
    if sentence_count > 0:
        words_per_sentence = [
            max(1, sum(1 for t in s if t.is_alpha)) for s in sentences
        ]
        avg_sentence_length = float(np.mean(words_per_sentence))
    else:
        avg_sentence_length = 0.0

    # POS counts
    noun_count = sum(1 for t in tokens if t.pos_ == "NOUN")
    verb_count = sum(1 for t in tokens if t.pos_ == "VERB")
    adj_count = sum(1 for t in tokens if t.pos_ == "ADJ")
    adv_count = sum(1 for t in tokens if t.pos_ == "ADV")
    pron_count = sum(1 for t in tokens if t.pos_ == "PRON")

    first_person_pron_count = sum(
        1 for t in tokens if t.text.lower() in FIRST_PERSON_PRONOUNS
    )

    noun_ratio = noun_count / n_tokens if n_tokens > 0 else 0.0
    verb_ratio = verb_count / n_tokens if n_tokens > 0 else 0.0
    adj_ratio = adj_count / n_tokens if n_tokens > 0 else 0.0
    adv_ratio = adv_count / n_tokens if n_tokens > 0 else 0.0
    pron_ratio = pron_count / n_tokens if n_tokens > 0 else 0.0
    first_person_pron_ratio = (
        first_person_pron_count / n_tokens if n_tokens > 0 else 0.0
    )

    # Stopwords
    stopword_count = sum(1 for t in tokens if t.is_stop)
    stopword_ratio = stopword_count / n_tokens if n_tokens > 0 else 0.0

    # Syntax variety (number of distinct POS / tokens)
    pos_types = set(t.pos_ for t in tokens)
    syntax_variety = len(pos_types) / n_tokens if n_tokens > 0 else 0.0

    # Complexity (approximate clause metrics)
    clause_tokens = [t for t in tokens if t.dep_ in CLAUSE_DEPS or t.tag_ == "SCONJ"]
    clause_token_ratio = len(clause_tokens) / n_tokens if n_tokens > 0 else 0.0

    complex_sentences = 0
    for s in sentences:
        if any(t.dep_ in CLAUSE_DEPS or t.tag_ == "SCONJ" for t in s):
            complex_sentences += 1
    complex_sentence_ratio = (
        complex_sentences / sentence_count if sentence_count > 0 else 0.0
    )

    # Punctuation metrics (character-based)
    text = doc.text
    total_chars = len(text)
    punct_chars = sum(1 for c in text if c in string.punctuation)
    punctuation_ratio = punct_chars / total_chars if total_chars > 0 else 0.0

    question_marks = text.count("?")
    exclamation_marks = text.count("!")
    commas = text.count(",")
    semicolons = text.count(";")
    colons = text.count(":")
    dashes = len(re.findall(r"[-–—]", text))

    denom = n_words if n_words > 0 else 1
    question_ratio = question_marks / denom
    exclamation_ratio = exclamation_marks / denom
    comma_ratio = commas / denom
    semicolon_ratio = semicolons / denom
    colon_ratio = colons / denom
    dash_ratio = dashes / denom

    return {
        "word_count": word_count,
        "unique_word_count": unique_words,
        "char_count": char_count,
        "avg_word_length": avg_word_length,
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_length,
        "noun_ratio": noun_ratio,
        "verb_ratio": verb_ratio,
        "adj_ratio": adj_ratio,
        "adv_ratio": adv_ratio,
        "pronoun_ratio": pron_ratio,
        "first_person_pronoun_ratio": first_person_pron_ratio,
        "stopword_ratio": stopword_ratio,
        "syntax_variety": syntax_variety,
        "clause_token_ratio": clause_token_ratio,
        "complex_sentence_ratio": complex_sentence_ratio,
        "punctuation_ratio": punctuation_ratio,
        "question_mark_ratio": question_ratio,
        "exclamation_mark_ratio": exclamation_ratio,
        "comma_ratio": comma_ratio,
        "semicolon_ratio": semicolon_ratio,
        "colon_ratio": colon_ratio,
        "dash_ratio": dash_ratio,
    }


def run_feature_extraction(
    df: pd.DataFrame,
    text_series: pd.Series,
    label_col: str,
    batch_size: int = 64,
) -> pd.DataFrame:
    """
    Run spaCy over the dataset and compute all features.
    """
    nlp = load_spacy_model()

    print(f"Computing features for {len(df)} texts...")
    rows = []
    for doc, (_, row) in zip(
        nlp.pipe(text_series.tolist(), batch_size=batch_size),
        df.iterrows(),
    ):
        feats = compute_features_for_doc(doc)
        feats[label_col] = row[label_col]
        if "modelo" in df.columns:
            feats["source"] = row["modelo"]
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "advanced_features.csv")
    feat_df.to_csv(out_path, index=False)
    print(f"Saved advanced features → {out_path}")
    return feat_df


# Statistical authorship tests


def compute_effect_sizes(
    feat_df: pd.DataFrame, label_col: str, ia_label: str, human_label: str
) -> pd.DataFrame:
    """
    Compute IA vs Human statistics for each numeric feature.
    """
    numeric_cols = [
        c
        for c in feat_df.columns
        if c != label_col
        and feat_df[c].dtype in (np.float64, np.float32, np.int64, np.int32)
        and c not in ["source"]
    ]

    rows = []
    for feat in numeric_cols:
        ia_vals = feat_df[feat_df[label_col] == ia_label][feat].dropna()
        hum_vals = feat_df[feat_df[label_col] == human_label][feat].dropna()

        if len(ia_vals) < 5 or len(hum_vals) < 5:
            continue

        mean_ia = ia_vals.mean()
        mean_h = hum_vals.mean()
        std_ia = ia_vals.std()
        std_h = hum_vals.std()

        pooled_sd = (
            math.sqrt((std_ia**2 + std_h**2) / 2.0)
            if (std_ia > 0 and std_h > 0)
            else np.nan
        )
        cohen_d = (
            (mean_ia - mean_h) / pooled_sd
            if pooled_sd and not np.isnan(pooled_sd)
            else np.nan
        )

        t_stat, p_val = stats.ttest_ind(ia_vals, hum_vals, equal_var=False)
        ks_stat, ks_p = stats.ks_2samp(ia_vals, hum_vals)

        rows.append(
            {
                "feature": feat,
                "ia_mean": mean_ia,
                "human_mean": mean_h,
                "ia_std": std_ia,
                "human_std": std_h,
                "cohen_d": cohen_d,
                "t_stat": t_stat,
                "p_value": p_val,
                "ks_stat": ks_stat,
                "ks_p_value": ks_p,
            }
        )

    stats_df = pd.DataFrame(rows).sort_values("cohen_d", ascending=False)
    out_path = os.path.join(OUTPUT_DIR, "advanced_tests.csv")
    stats_df.to_csv(out_path, index=False)
    print(f"Saved IA vs Human tests → {out_path}")
    return stats_df


# N-gram analysis (bigrams + trigrams)


def compute_ngram_log_odds(
    texts: List[str],
    labels: List[str],
    ia_label: str,
    human_label: str,
    min_df: int = 20,
    max_features: int = 10000,
    top_k: int = 50,
) -> pd.DataFrame:
    """
    Compute bigram+trigram log-odds per class (IA vs Human).
    Inspired by Monroe et al. 2008.
    """
    print("Computing bigram/trigram log-odds...")

    vectorizer = CountVectorizer(
        ngram_range=(2, 3),
        min_df=min_df,
        max_features=max_features,
        analyzer="word",
    )

    X = vectorizer.fit_transform(texts)
    vocab = np.array(vectorizer.get_feature_names_out())

    labels_arr = np.array(labels)
    ia_mask = labels_arr == ia_label
    human_mask = labels_arr == human_label

    X_ia = X[ia_mask]
    X_h = X[human_mask]

    ia_counts = np.asarray(X_ia.sum(axis=0)).flatten()
    h_counts = np.asarray(X_h.sum(axis=0)).flatten()

    # Laplace smoothing
    alpha = 1.0
    ia_total = ia_counts.sum() + alpha * len(vocab)
    h_total = h_counts.sum() + alpha * len(vocab)

    ia_probs = (ia_counts + alpha) / ia_total
    h_probs = (h_counts + alpha) / h_total

    log_odds_ia = np.log(ia_probs / h_probs)
    log_odds_h = -log_odds_ia  # symmetric

    # Build ranked tables
    df_ia = pd.DataFrame(
        {
            "ngram": vocab,
            "class": ia_label,
            "log_odds": log_odds_ia,
            "freq_in_class": ia_counts,
            "freq_in_other": h_counts,
        }
    ).sort_values("log_odds", ascending=False)

    df_h = pd.DataFrame(
        {
            "ngram": vocab,
            "class": human_label,
            "log_odds": log_odds_h,
            "freq_in_class": h_counts,
            "freq_in_other": ia_counts,
        }
    ).sort_values("log_odds", ascending=False)

    df_top = pd.concat([df_ia.head(top_k), df_h.head(top_k)], ignore_index=True)
    out_path = os.path.join(OUTPUT_DIR, "advanced_ngrams.csv")
    df_top.to_csv(out_path, index=False)
    print(f"Saved top n-grams → {out_path}")
    return df_top


# Main entry point


def run_advanced_stylometry(
    csv_path: str,
    text_columns: Tuple[str, ...] = DEFAULT_TEXT_COLUMNS,
    label_column: str = DEFAULT_LABEL_COLUMN,
):
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    text_series = build_text_column(df, text_columns)
    ia_label, human_label = detect_labels(df, label_column)

    feat_df = run_feature_extraction(df, text_series, label_column)
    stats_df = compute_effect_sizes(feat_df, label_column, ia_label, human_label)

    # N-gram analysis uses the same text as spaCy
    _ = compute_ngram_log_odds(
        texts=text_series.tolist(),
        labels=df[label_column].tolist(),
        ia_label=ia_label,
        human_label=human_label,
        min_df=20,
        max_features=10000,
        top_k=50,
    )

    print("\n=== Advanced stylometric analysis completed ===")
    print(f"- Features: {os.path.join(OUTPUT_DIR, 'advanced_features.csv')}")
    print(f"- IA vs Human tests: {os.path.join(OUTPUT_DIR, 'advanced_tests.csv')}")
    print(f"- Top n-grams: {os.path.join(OUTPUT_DIR, 'advanced_ngrams.csv')}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stylometric_advanced.py <dataset_csv_path>")
        sys.exit(1)

    csv_path_arg = sys.argv[1]
    run_advanced_stylometry(csv_path_arg)
