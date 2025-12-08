import os
import pandas as pd
import nltk
import textstat
from scipy.stats import ttest_ind, ks_2samp

# Download punkt tokenizer silently if not already present
nltk.download("punkt", quiet=True)


def _compute_features(text: str) -> dict:
    """
    Compute basic stylometric features for a single text.
    """
    text = str(text)
    words = nltk.word_tokenize(text)
    sentences = nltk.sent_tokenize(text)

    total_words = len(words)
    unique_words = len(set(words))

    ttr = unique_words / total_words if total_words > 0 else 0.0

    hapax = 0
    if total_words > 0:
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        hapax = sum(1 for w, c in freq.items() if c == 1)
    hapax_ratio = hapax / total_words if total_words > 0 else 0.0

    avg_sentence_len = total_words / max(1, len(sentences))
    avg_word_len = sum(len(w) for w in words) / total_words if total_words > 0 else 0.0

    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = 0.0

    try:
        fog = textstat.gunning_fog(text)
    except Exception:
        fog = 0.0

    return {
        "ttr": ttr,
        "hapax_ratio": hapax_ratio,
        "avg_sentence_length": avg_sentence_len,
        "avg_word_length": avg_word_len,
        "flesch": flesch,
        "fog": fog,
    }


def run_stylometric_analysis(csv_path: str) -> None:
    """
    Run stylometric analysis on a dataset CSV.

    Parameters
    ----------
    csv_path : str
        Path to a CSV containing at least 'cuerpo', 'authorship' and optionally 'modelo'.
    """
    os.makedirs("stylometry_outputs", exist_ok=True)

    df = pd.read_csv(csv_path)

    if "cuerpo" not in df.columns or "authorship" not in df.columns:
        raise ValueError("CSV must contain at least 'cuerpo' and 'authorship' columns.")

    rows = []
    for _, row in df.iterrows():
        feats = _compute_features(row["cuerpo"])
        feats["authorship"] = row["authorship"]
        feats["source"] = row.get("modelo", "unknown")
        rows.append(feats)

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv("stylometry_outputs/stylometric_features.csv", index=False)

    # Compare IA vs Humano
    ia = feat_df[feat_df["authorship"] == "IA"]
    human = feat_df[feat_df["authorship"] == "Humano"]

    stats_rows = []
    metrics = [
        "ttr",
        "hapax_ratio",
        "avg_sentence_length",
        "avg_word_length",
        "flesch",
        "fog",
    ]

    for col in metrics:
        if len(ia[col]) > 1 and len(human[col]) > 1:
            t_stat, p_val = ttest_ind(ia[col], human[col], equal_var=False)
            ks_stat, ks_p = ks_2samp(ia[col], human[col])
        else:
            t_stat, p_val, ks_stat, ks_p = (0.0, 1.0, 0.0, 1.0)

        stats_rows.append(
            {
                "metric": col,
                "ia_mean": ia[col].mean(),
                "humano_mean": human[col].mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "ks_stat": ks_stat,
                "ks_p_value": ks_p,
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv("stylometry_outputs/stylometric_tests.csv", index=False)

    print("Stylometric analysis complete")
    print("Outputs written to 'stylometry_outputs/'")
