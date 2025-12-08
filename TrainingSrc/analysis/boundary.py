import os
import matplotlib.pyplot as plt


def decision_boundary_analysis(df, prefix: str = "test") -> None:
    """
    Visualize the decision boundary using P(Humano) vs P(IA).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns: prob_Humano, prob_IA, correct.
    prefix : str
        Prefix for output filenames.
    """
    os.makedirs("analysis_outputs", exist_ok=True)

    if "prob_IA" not in df.columns or "prob_Humano" not in df.columns:
        print("No probability columns found, skipping decision boundary analysis.")
        return

    plt.figure(figsize=(7, 7))
    correct_mask = df["correct"]

    # Correct points
    plt.scatter(
        df["prob_Humano"][correct_mask],
        df["prob_IA"][correct_mask],
        alpha=0.6,
        label="Correct",
    )
    # Incorrect points
    plt.scatter(
        df["prob_Humano"][~correct_mask],
        df["prob_IA"][~correct_mask],
        alpha=0.9,
        label="Incorrect",
        marker="x",
    )

    plt.axvline(0.5, linestyle="--", color="gray")
    plt.axhline(0.5, linestyle="--", color="gray")
    plt.xlabel("P(Humano)")
    plt.ylabel("P(IA)")
    plt.title(f"Decision boundary scatter ({prefix})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"analysis_outputs/{prefix}_decision_boundary.png")
    plt.close()

    print(f"✓ Decision boundary analysis ({prefix}) complete")
