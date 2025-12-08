import os


def error_analysis(df, prefix: str = "test") -> None:
    """
    Save detailed information about misclassified samples.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: text, true_label, predicted_label, confidence, source, correct.
    prefix : str
        Prefix for output filenames.
    """
    os.makedirs("analysis_outputs", exist_ok=True)

    errors = df[~df["correct"]]
    errors_path = f"analysis_outputs/{prefix}_errors.csv"
    errors.to_csv(errors_path, index=False)

    detailed_path = f"analysis_outputs/{prefix}_errors_detailed.txt"
    with open(detailed_path, "w", encoding="utf-8") as f:
        for _, row in errors.iterrows():
            f.write("------ ERROR ------\n")
            f.write(f"Source: {row.get('source', 'unknown')}\n")
            f.write(f"True: {row['true_label']} | Pred: {row['predicted_label']}\n")
            f.write(f"Confidence: {row['confidence']}\n")
            f.write("Text preview:\n")
            f.write(str(row["text"])[:500] + "\n\n")

    print(f"Error analysis ({prefix}) complete")
