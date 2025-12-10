import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from analysis.lime_local import lime_global_analysis

# ========= CONFIG ==========
MODEL_PATH = "BETenio"   # Ajusta si tu modelo está en otra carpeta
TEST_DATASET_PATH = "test_dataset.csv"

NUM_LIME_SAMPLES = 80         # 40 IA + 40 Humano
PERTURBATIONS = 300
# ===========================


def load_model_and_tokenizer(model_path):
    print(f"Loading model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer


def balanced_sample(df, label_column, total_samples):
    labels = df[label_column].unique().tolist()
    per_class = total_samples // len(labels)

    samples = []
    for lab in labels:
        subset = df[df[label_column] == lab]
        n = min(per_class, len(subset))
        samples.append(subset.sample(n, random_state=42))

    return pd.concat(samples)


def main():
    print("\n=== RUNNING LIME ONLY ===")

    # Load dataset
    df = pd.read_csv(TEST_DATASET_PATH)

    # Detect label column
    if "authorship" in df.columns:
        label_col = "authorship"
    else:
        raise ValueError("Dataset must contain a column named 'authorship'.")

    print(f"Detected label column: {label_col}")
    print(f"Unique labels: {df[label_col].unique().tolist()}")

    # Balanced sampling
    print(f"Sampling {NUM_LIME_SAMPLES} texts (balanced IA / Humano)...")
    df_sample = balanced_sample(df, label_col, NUM_LIME_SAMPLES)

    # Load model
    load_model_and_tokenizer(MODEL_PATH)

    # Run LIME
    lime_global_analysis(
        MODEL_PATH,          # model_path
        df_sample,           # df
        prefix="LIME_ONLY",
        num_samples=NUM_LIME_SAMPLES,
        perturbations=PERTURBATIONS
    )

    print("\n=== LIME ANALYSIS COMPLETE ===")
    print("Outputs saved in: analysis_outputs/")


if __name__ == "__main__":
    main()
