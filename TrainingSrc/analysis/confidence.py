import os
import matplotlib.pyplot as plt


def confidence_analysis(df, prefix: str = "test") -> None:
    """
    Generate basic confidence analysis plots and CSV summaries.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: confidence, correct, true_label, source (optional).
    prefix : str
        Prefix to distinguish between train/test sets in output filenames.
    """
    os.makedirs("analysis_outputs", exist_ok=True)

    # 1) Global confidence distribution
    plt.figure(figsize=(8, 5))
    plt.hist(df["confidence"], bins=30)
    plt.title(f"Confidence distribution ({prefix})")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"analysis_outputs/{prefix}_confidence_distribution.png")
    plt.close()

    # 2) Confidence for correct vs incorrect predictions
    plt.figure(figsize=(8, 5))
    correct_conf = df[df["correct"]]["confidence"]
    incorrect_conf = df[~df["correct"]]["confidence"]

    plt.hist(correct_conf, bins=30, alpha=0.7, label="Correct")
    plt.hist(incorrect_conf, bins=30, alpha=0.7, label="Incorrect")
    plt.title(f"Confidence: correct vs incorrect ({prefix})")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis_outputs/{prefix}_confidence_correct_vs_incorrect.png")
    plt.close()

    # 3) Mean confidence by true label
    df.groupby("true_label")["confidence"].mean().to_csv(
        f"analysis_outputs/{prefix}_confidence_by_true_label.csv"
    )

    # 4) Mean confidence by source (if available)
    if "source" in df.columns:
        df.groupby("source")["confidence"].mean().to_csv(
            f"analysis_outputs/{prefix}_confidence_by_source.csv"
        )

    print(f"Confidence analysis ({prefix}) complete")
