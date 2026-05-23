"""Train and save model artifacts for the Streamlit app.

Creates:
- model.pkl
- vectorizer.pkl

Keeps the ML logic compatible with the Fake/Real mapping used in the original project:
- class 0 => Fake
- class 1 => Real
"""

import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from preprocessing import preprocess_for_tfidf


MODEL_PATH = "model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Detect likely columns
    text_col = None
    label_col = None
    for c in df.columns:
        lc = c.lower()
        if text_col is None and ("text" in lc or "news" in lc or "statement" in lc or "content" in lc):
            text_col = c
        if label_col is None and ("label" in lc or lc == "target" or "fake" in lc):
            label_col = c

    if text_col is None:
        text_col = df.columns[0]
    if label_col is None:
        label_col = df.columns[1]

    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    return df


def main():
    train_csv = "train.csv"
    if not os.path.exists(train_csv):
        raise FileNotFoundError("train.csv not found in project root")

    df = load_dataset(train_csv)

    df["clean"] = df["text"].astype(str).apply(preprocess_for_tfidf)

    vectorizer = TfidfVectorizer(max_features=20000)
    X = vectorizer.fit_transform(df["clean"])

    y = df["label"].values

    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved {MODEL_PATH} and {VECTORIZER_PATH}")


if __name__ == "__main__":
    main()

