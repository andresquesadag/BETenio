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

from tester import testModel


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
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class_id = predictions.argmax().item()
            confidence = predictions.max().item()

        # Decode prediction
        predicted_label = label_encoder.inverse_transform([predicted_class_id])[0]
        current_prediction = {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_probabilities": {
                label: prob.item()
                for label, prob in zip(
                    label_encoder.classes_, predictions[0].cpu()
                )  # Move to CPU for numpy conversion
            },
        }
        predictions.append(current_prediction)

    return predictions



if __name__ == "__main__":

    Tk().withdraw()
    folderPath = askdirectory()
    train_path = folderPath + "/train_dataset.csv"
    test_path = folderPath + "/test_dataset.csv"
    output_path = folderPath + "/BETenio"

    os.makedirs("analysis_outputs", exist_ok=True)  # ←←← NEW

    do_training = input(
        "\nDo you want to train the model or use an existing one? (0 = train, 1 = existing): "
    )
    if do_training == "1":
        print("Skipping training. Using existing model.")
    else:
        print("Training model...")
        train_model(csv_path=train_path, output_dir=output_path)

    print("\n========= TEST DATASET RESULTS =========")
    testModel(
        test_path=test_path, model_path=output_path
    )  # ←←← uses new extended tester

    print("\n========= TRAIN DATASET RESULTS =========")
    testModel(test_path=train_path, model_path=output_path)
