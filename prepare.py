"""
Data preparation and evaluation module for Yeast protein localization classification.
DO NOT MODIFY this file.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import json
import os

RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "yeast.csv")

FEATURE_COLS = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc"]
TARGET_COL = "name"


def load_data():
    """Load and split the yeast dataset into train/val/test sets."""
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y_raw = df[TARGET_COL].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    num_classes = len(class_names)

    # Split: 60% train, 20% val, 20% test (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Fit scaler on training data only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "class_names": class_names,
        "num_classes": num_classes,
        "num_features": X_train.shape[1],
        "scaler": scaler,
    }

    print(f"Dataset loaded: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Features: {FEATURE_COLS}")

    return data


def evaluate(y_true, y_pred_proba, y_pred, split="val"):
    """
    Evaluate model predictions and print/return metrics.

    Args:
        y_true: Ground truth labels (integer encoded)
        y_pred_proba: Predicted probabilities, shape (n_samples, n_classes)
        y_pred: Predicted class labels (integer encoded)
        split: Name of the split being evaluated ('val' or 'test')

    Returns:
        dict with metric names and values
    """
    # AUC-ROC (one-vs-rest, macro average)
    try:
        auroc = roc_auc_score(
            y_true, y_pred_proba, multi_class="ovr", average="macro"
        )
    except ValueError:
        auroc = 0.0

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    metrics = {
        f"{split}_auroc": round(auroc, 5),
        f"{split}_f1": round(f1, 5),
        f"{split}_accuracy": round(acc, 5),
    }

    print(f"\n{'='*50}")
    print(f"  {split.upper()} EVALUATION RESULTS")
    print(f"{'='*50}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}\n")

    # Save metrics to file
    metrics_path = os.path.join(os.path.dirname(__file__), f"metrics_{split}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
