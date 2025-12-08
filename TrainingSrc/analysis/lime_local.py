import os
import torch
import joblib
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer


def lime_global_analysis(
    model_path: str,
    df,
    prefix: str = "test",
    num_samples: int = 10,  # número de textos explicados
    perturbations: int = 300,  # textos sintéticos que LIME genera (default=5000)
    max_length: int = 256,  # más pequeño = menos VRAM / más rápido
):
    """
    GPU-safe and low-memory LIME analysis.
    """

    os.makedirs("analysis_outputs", exist_ok=True)

    # Select GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"LIME running on device: {device}")

    # Load model on GPU
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    label_encoder = joblib.load(os.path.join(model_path, "label_encoder.pkl"))
    class_names = list(label_encoder.classes_)

    explainer = LimeTextExplainer(class_names=class_names)

    # Limit number of examples explained
    subset = df.sample(min(num_samples, len(df)), random_state=42)

    ##################################################
    # GPU-FRIENDLY predict_proba (1 text per pass)
    ##################################################
    def predict_proba(texts):
        results = []

        for t in texts:
            enc = tokenizer(
                t,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            results.append(probs)

        return np.vstack(results)

    ##################################################
    # AGGREGATE FEATURE IMPORTANCE
    ##################################################
    pos_weights = {}
    neg_weights = {}

    txt_out = f"analysis_outputs/{prefix}_lime_explanations.txt"
    with open(txt_out, "w", encoding="utf-8") as f:

        for _, row in subset.iterrows():
            text = str(row["text"])

            exp = explainer.explain_instance(
                text, predict_proba, num_features=10, num_samples=perturbations
            )

            # Save explanation
            f.write("TEXT (truncated):\n")
            f.write(text[:500] + "\n\n")
            f.write("EXPLANATION:\n")
            f.write(str(exp.as_list()) + "\n\n")

            # Accumulate weights
            for word, weight in exp.as_list():
                if weight >= 0:
                    pos_weights[word] = pos_weights.get(word, 0) + weight
                else:
                    neg_weights[word] = neg_weights.get(word, 0) + abs(weight)

    # Save aggregated importance
    pd.Series(pos_weights).sort_values(ascending=False).to_csv(
        f"analysis_outputs/{prefix}_lime_positive_features.csv"
    )
    pd.Series(neg_weights).sort_values(ascending=False).to_csv(
        f"analysis_outputs/{prefix}_lime_negative_features.csv"
    )

    print(f"✓ LIME analysis ({prefix}) completed.")
