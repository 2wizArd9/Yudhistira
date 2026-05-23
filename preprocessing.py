from __future__ import annotations

import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Safe downloads (only runs if missing resources)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


STOPS = set(stopwords.words("english"))
STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def tokenize(text: str) -> List[str]:
    return [t for t in re.split(r"\s+", text) if t]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPS]


def stem(tokens: List[str]) -> List[str]:
    return [STEMMER.stem(t) for t in tokens]


def preprocess_for_tfidf(text: str) -> str:
    """Full preprocessing pipeline returning a string for TF-IDF."""
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = stem(tokens)
    return " ".join(tokens)

