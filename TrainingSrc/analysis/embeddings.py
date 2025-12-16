import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.decomposition import PCA


def embedding_visualization(
    model_path: str, df, prefix: str = "test", sample_size: int = 800
) -> None:
    """
    Extract CLS embeddings for a sample of texts and visualize them with PCA.

    Parameters
    ----------
    model_path : str
        Path where the fine-tuned model is saved.
    df : pandas.DataFrame
        Must contain columns: text, true_label.
    prefix : str
        Prefix for output filenames.
    sample_size : int
        Maximum number of samples for visualization.
    """
    os.makedirs("analysis_outputs", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sample = df.sample(min(sample_size, len(df)), random_state=42)

    embeddings = []

    for text in sample["text"]:
        enc = tokenizer(
            str(text),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            # Use the base model to obtain CLS embeddings
            outputs = model.base_model(**enc)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        embeddings.append(cls_embedding)

    X = np.vstack(embeddings)

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(X)

    # Map labels to integers for coloring
    label_codes = sample["true_label"].astype("category").cat.codes.values

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=label_codes,
        cmap="coolwarm",
        alpha=0.7,
    )
    plt.title(f"PCA embeddings ({prefix})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(f"analysis_outputs/{prefix}_embeddings_pca.png")
    plt.close()

    print(f"✓ Embedding visualization ({prefix}) complete")
