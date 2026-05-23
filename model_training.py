from __future__ import annotations

import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from preprocessing import preprocess_for_tfidf


MODEL_COMPARISON_CACHE: pd.DataFrame | None = None


def load_dataset_simplified(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Expect columns: text + label or similar.
    # Try common names.
    text_col = None
    label_col = None
    for c in df.columns:
        lc = c.lower()
        if text_col is None and ("text" in lc or "news" in lc or "statement" in lc or "content" in lc):
            text_col = c
        if label_col is None and ("label" in lc or lc == "target" or "fake" in lc):
            label_col = c

    # Fallback heuristics
    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    return df


def train_models_and_compare(train_csv: str, test_csv: str) -> pd.DataFrame:
    train_df = load_dataset_simplified(train_csv)
    test_df = load_dataset_simplified(test_csv)

    train_df["clean"] = train_df["text"].astype(str).apply(preprocess_for_tfidf)
    test_df["clean"] = test_df["text"].astype(str).apply(preprocess_for_tfidf)

    vectorizer = TfidfVectorizer(max_features=20000)
    X_train = vectorizer.fit_transform(train_df["clean"])
    X_test = vectorizer.transform(test_df["clean"])

    y_train = train_df["label"].values
    y_test = test_df["label"].values

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "SVM (Linear)": LinearSVC(),
        "Naive Bayes": MultinomialNB(),
    }

    rows = []
    for name, clf in models.items():
        if name.startswith("SVM"):
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
        else:
            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)

        rows.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, pred, average="weighted", zero_division=0),
            }
        )

    return pd.DataFrame(rows)


def compute_model_comparison() -> pd.DataFrame:
    global MODEL_COMPARISON_CACHE
    if MODEL_COMPARISON_CACHE is not None:
        return MODEL_COMPARISON_CACHE

    # Use repo csvs if available
    # train.csv and test.csv exist in the root.
    train_csv = "train.csv"
    test_csv = "valid.csv" if os.path.exists("valid.csv") else "test.csv"

    if not os.path.exists(train_csv):
        MODEL_COMPARISON_CACHE = pd.DataFrame(
            [{"Model": "(missing data)", "Accuracy": 0, "Precision": 0, "Recall": 0, "F1 Score": 0}]
        )
        return MODEL_COMPARISON_CACHE

    MODEL_COMPARISON_CACHE = train_models_and_compare(train_csv, test_csv)
    return MODEL_COMPARISON_CACHE

