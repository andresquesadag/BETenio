import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog


# ======================================================
# Utility: open file dialog
# ======================================================
def ask_file(title):
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(title=title)
    root.destroy()
    return path


# ======================================================
# LIME ANALYSIS (AUTO-DETECT COLUMNS)
# ======================================================
def analyze_lime_features(lime_pos_path, lime_neg_path):
    print("\n=== LIME Analysis ===")

    # Load WITHOUT header (your files do not have one)
    df_pos = pd.read_csv(lime_pos_path, header=None)
    df_neg = pd.read_csv(lime_neg_path, header=None)

    def clean_lime(df):
        # Detect numeric column = importance
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Detect text-like column = feature
        text_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if len(num_cols) == 0 or len(text_cols) == 0:
            raise ValueError(
                f"Could not infer feature/importance columns. Columns: {df.columns}"
            )

        imp_col = num_cols[0]
        feat_col = text_cols[0]

        cleaned = pd.DataFrame(
            {
                "feature": df[feat_col].astype(str),
                "importance": df[imp_col].astype(float),
            }
        )
        return cleaned

    df_pos_clean = clean_lime(df_pos)
    df_neg_clean = clean_lime(df_neg)

    # Group and aggregate
    pos_group = df_pos_clean.groupby("feature")["importance"].agg(
        ["mean", "std", "count"]
    )
    pos_group["class"] = "IA"

    neg_group = df_neg_clean.groupby("feature")["importance"].agg(
        ["mean", "std", "count"]
    )
    neg_group["class"] = "Human"

    df_ranked = pd.concat([pos_group, neg_group])
    df_ranked = df_ranked.sort_values("mean", ascending=False)

    os.makedirs("extended_analysis", exist_ok=True)
    df_ranked.to_csv("extended_analysis/lime_feature_rankings.csv")

    print("✔ Saved LIME rankings → extended_analysis/lime_feature_rankings.csv")
    return df_ranked


# ======================================================
# Stylometric effect-size analysis
# ======================================================
def analyze_stylometry(styl_feat_path, styl_tests_path):
    print("\n=== Stylometric Analysis ===")

    df = pd.read_csv(styl_feat_path)

    required_cols = ["authorship"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' is missing in stylometric_features.csv")

    numeric_cols = [
        c
        for c in df.columns
        if df[c].dtype in [np.float64, np.int64] and c not in ["source"]
    ]

    rows = []
    for feat in numeric_cols:
        ia_vals = df[df["authorship"] == "IA"][feat].dropna()
        hum_vals = df[df["authorship"] == "Human"][feat].dropna()

        mean_ia = ia_vals.mean()
        mean_h = hum_vals.mean()

        std_ia = ia_vals.std()
        std_h = hum_vals.std()

        pooled_sd = np.sqrt(((std_ia**2 + std_h**2) / 2))
        d = (mean_ia - mean_h) / pooled_sd if pooled_sd > 0 else np.nan

        t_stat, p_val = stats.ttest_ind(ia_vals, hum_vals, equal_var=False)

        rows.append(
            {
                "feature": feat,
                "mean_IA": mean_ia,
                "mean_Human": mean_h,
                "std_IA": std_ia,
                "std_Human": std_h,
                "cohen_d": d,
                "t_stat": t_stat,
                "p_value": p_val,
            }
        )

    df_stats = pd.DataFrame(rows)
    df_stats = df_stats.sort_values("cohen_d", ascending=False)

    os.makedirs("extended_analysis", exist_ok=True)
    df_stats.to_csv("extended_analysis/stylometry_effect_sizes.csv", index=False)

    print(
        "✔ Saved stylometry effect sizes → extended_analysis/stylometry_effect_sizes.csv"
    )
    return df_stats


# ======================================================
# Generate summary text report
# ======================================================
def generate_text_report(lime_df, styl_df):
    print("\n=== Generating summary report ===")

    lines = []
    lines.append("=== EXTENDED ANALYSIS REPORT ===\n")
    lines.append("Summary of LIME feature importance and stylometric separability.\n")

    top_ia = lime_df[lime_df["class"] == "IA"].head(20)
    lines.append("\n--- TOP IA FEATURES (LIME) ---\n")
    for _, r in top_ia.iterrows():
        lines.append(f"{r.name}: mean={r['mean']:.4f}, count={r['count']}\n")

    top_h = (
        lime_df[lime_df["class"] == "Human"]
        .sort_values("mean", ascending=False)
        .head(20)
    )
    lines.append("\n--- TOP HUMAN FEATURES (LIME) ---\n")
    for _, r in top_h.iterrows():
        lines.append(f"{r.name}: mean={r['mean']:.4f}, count={r['count']}\n")

    lines.append("\n--- STRONGEST STYLOMETRIC SEPARATIONS ---\n")
    top_eff = styl_df.sort_values("cohen_d", ascending=False).head(10)
    for _, r in top_eff.iterrows():
        lines.append(
            f"{r['feature']} → Cohen's d={r['cohen_d']:.3f}, p={r['p_value']:.2e}, "
            f"IA_mean={r['mean_IA']:.3f}, Human_mean={r['mean_Human']:.3f}\n"
        )

    with open("extended_analysis/analysis_report.txt", "w", encoding="utf8") as f:
        f.writelines(lines)

    print("✔ Saved summary → extended_analysis/analysis_report.txt")


# ======================================================
# MAIN SCRIPT
# ======================================================
def main():
    print("\n=== EXTENDED RESULTS ANALYZER ===\n")

    print("Select LIME POSITIVE CSV (IA indicators):")
    lime_pos = ask_file("Select test_lime_positive_features.csv")

    print("Select LIME NEGATIVE CSV (Human indicators):")
    lime_neg = ask_file("Select test_lime_negative_features.csv")

    print("Select Stylometric Features CSV:")
    styl_feat = ask_file("Select stylometric_features.csv")

    print("Select Stylometric Tests CSV:")
    styl_tests = ask_file("Select stylometric_tests.csv")

    lime_df = analyze_lime_features(lime_pos, lime_neg)
    styl_df = analyze_stylometry(styl_feat, styl_tests)

    generate_text_report(lime_df, styl_df)

    print("\n=== DONE! ===")
    print("Outputs saved in 'extended_analysis/' folder.\n")


if __name__ == "__main__":
    main()
