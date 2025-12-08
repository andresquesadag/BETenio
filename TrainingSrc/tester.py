import random
import sys
import pandas as pd
import joblib
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askdirectory
from sklearn.metrics import confusion_matrix
import os


def check_gpu_availability():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"CUDA is available! Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(
            f"Current GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")
    return device


def load_and_prepare_data(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["cuerpo"].tolist()
    labels = df["authorship"].tolist()

    # Encode labels to integers
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    # Split into train and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    return train_texts, val_texts, train_labels, val_labels, label_encoder


def setup_model_and_tokenizer(model_name, num_labels, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, ignore_mismatched_sizes=True
    )

    # Move model to GPU if available
    model.to(device)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_dataset(texts, labels, tokenizer, max_length=512):
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )

    dataset = Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )

    return dataset


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train_model(
    csv_path,
    model_name="dccuchile/bert-base-spanish-wwm-uncased",
    output_dir="./results",
):
    device = check_gpu_availability()

    print("Loading and preparing data...")
    train_texts, val_texts, train_labels, val_labels, label_encoder = (
        load_and_prepare_data(csv_path)
    )

    num_labels = len(label_encoder.classes_)
    print(f"Number of unique labels: {num_labels}")
    print(f"Label classes: {label_encoder.classes_}")

    print(f"Loading model: {model_name}")
    model, tokenizer = setup_model_and_tokenizer(model_name, num_labels, device)

    print("Creating datasets...")
    train_dataset = create_dataset(train_texts, train_labels, tokenizer)
    val_dataset = create_dataset(val_texts, val_labels, tokenizer)

    print("Datasets created")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if torch.cuda.is_available():
        train_batch_size = 16
        eval_batch_size = 32
        fp16 = True
        dataloader_pin_memory = True
    else:
        train_batch_size = 8
        eval_batch_size = 16
        fp16 = False
        dataloader_pin_memory = False

    random.seed()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=random.randint(0, 2**32 - 1),
        fp16=fp16,
        dataloader_pin_memory=dataloader_pin_memory,
        gradient_checkpointing=True,
        report_to=None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    joblib.dump(label_encoder, f"{output_dir}/label_encoder.pkl")

    print(f"Training completed! Model saved to {output_dir}")

    return trainer, label_encoder


def predict_with_trained_model(texts, model_path, tokenizer_path=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer_path is None:
        tokenizer_path = model_path

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")

    predictions = []

    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu()
            predicted_class_id = probs.argmax().item()
            confidence = probs.max().item()

        # Decode prediction
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
        current_prediction = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_probabilities": {
                label: prob.item()
                for label, prob in zip(
                    label_encoder.classes_, probs[0]
                )  # Move to CPU for numpy conversion
            },
        }
        predictions.append(current_prediction)

    return predictions



# --- Extended Analysis ---

def testModel(test_path, model_path):

    # Try importing analysis modules — if missing, skip gracefully
    try:
        from analysis.confidence import confidence_analysis
        from analysis.errors import error_analysis
        from analysis.perf_source import performance_by_source
        from analysis.boundary import decision_boundary_analysis
        from analysis.lime_local import lime_global_analysis
        from analysis.embeddings import embedding_visualization
        analysis_available = True
    except:
        print("WARNING: Analysis modules not found. Extended analysis skipped.")
        analysis_available = False

    # Load CSV
    test_data = pd.read_csv(test_path)
    body_list = test_data["cuerpo"].to_list()
    model_list = test_data["modelo"].to_list()
    label_list = test_data["authorship"].to_list()

    # Base predictions (your original code)
    predictions_list = predict_with_trained_model(body_list, model_path)

    predictions = [p["predicted_label"] for p in predictions_list]
    confidences = [p["confidence"] for p in predictions_list]
    prob_IA = [p["all_probabilities"].get("IA", float("nan")) for p in predictions_list]
    prob_H = [p["all_probabilities"].get("Humano", float("nan")) for p in predictions_list]

    # Standard metrics
    accuracy = accuracy_score(label_list, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(label_list, predictions, average='weighted')
    unique_models = test_data["modelo"].unique()
    fails = {modelo: 0 for modelo in unique_models}

    for i in range(len(predictions)):
        if predictions[i] != label_list[i]:
            fails[model_list[i]] += 1

    print("-----------------")
    print("FINAL RESULTS: ")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(confusion_matrix(label_list, predictions))
    print("Fails by model:")
    print(fails)
    print("-----------------")

    # If analysis modules not present, we stop here
    if not analysis_available:
        return

    # Build DataFrame for analysis
    df_results = pd.DataFrame({
        "text": body_list,
        "true_label": label_list,
        "predicted_label": predictions,
        "confidence": confidences,
        "prob_IA": prob_IA,
        "prob_Humano": prob_H,
        "correct": [p == t for p, t in zip(predictions, label_list)],
        "source": model_list,
        "length_chars": [len(str(x)) for x in body_list],
        "length_words": [len(str(x).split()) for x in body_list],
    })

    os.makedirs("analysis_outputs", exist_ok=True)

    prefix = "test" if "test" in os.path.basename(test_path).lower() else "train"

    df_results.to_csv(f"analysis_outputs/{prefix}_predictions_with_confidence.csv", index=False)

    print(f"Running extended analysis for {prefix}...")

    confidence_analysis(df_results, prefix)
    error_analysis(df_results, prefix)
    performance_by_source(df_results, prefix)
    decision_boundary_analysis(df_results, prefix)
    lime_global_analysis(model_path, df_results, prefix)
    embedding_visualization(model_path, df_results, prefix)

    print(f"Extended analysis completed for {prefix} dataset.")



# Example usage
if __name__ == "__main__":

    # Tk().withdraw()
    folderPath = askdirectory()
    train_path = folderPath + "/train_dataset.csv"
    test_path = folderPath + "/test_dataset.csv"
    output_path = folderPath + "/BETenio"

    print("Test dataset results:")
    testModel(test_path=test_path, model_path=output_path)
    print("Train dataset results:")
    testModel(test_path=train_path, model_path=output_path)
