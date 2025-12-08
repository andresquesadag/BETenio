import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def performance_by_source(df, prefix: str = "test") -> None:
    """
    Compute metrics per source (e.g., Claude, GPT, Deepseek, Reales).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: true_label, predicted_label, confidence, source, correct.
    prefix : str
        Prefix for output filenames.
    """
    os.makedirs("analysis_outputs", exist_ok=True)

    if "source" not in df.columns:
        print("No 'source' column found, skipping performance-by-source analysis.")
        return

    rows = []
    for source, grp in df.groupby("source"):
        y_true = grp["true_label"]
        y_pred = grp["predicted_label"]

        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        rows.append(
            {
                "source": source,
                "n": len(grp),
                "accuracy": acc,
                "precision": p,
                "recall": r,
                "f1": f1,
                "errors": (~grp["correct"]).sum(),
                "mean_confidence": grp["confidence"].mean(),
            }
        )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(f"analysis_outputs/{prefix}_performance_by_source.csv", index=False)

    print(f"✓ Performance-by-source ({prefix}) complete")
