"""Global singleton cache for model.pkl and vectorizer.pkl — loaded once per process."""

from __future__ import annotations

import io
import os
import pickle
import threading
import time
from typing import Any

_lock = threading.Lock()
_model: Any = None
_vectorizer: Any = None
_loaded = False
_loading = False


def _time_log(msg: str, elapsed: float) -> None:
    print(f"[TIME] {msg} in {elapsed:.3f}s", flush=True)


def _safe_pickle_load(path: str, timeout_s: float = 10.0) -> Any:
    """Safely load a pickle with a hard timeout via subprocess.

    Thread-based timeouts cannot kill a stuck unpickling operation.
    This function uses a subprocess so we can enforce timeout.
    """

    import subprocess
    import sys

    # Inline python loader; prints "OK" then dumps pickled bytes of result.
    # Using pickle to shuttle object across process can be large, but we
    # expect ML artifacts to be picklable and sizes reasonable.
    code = r"""
import pickle, sys
path = sys.argv[1]
with open(path, 'rb') as f:
    obj = pickle.load(f)
# re-pickle for transport
sys.stdout.buffer.write(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
"""

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Loading timed out for {path} after {timeout_s}s") from e

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"pickle.load failed for {path}: {err}")

    try:
        obj = pickle.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"Loaded bytes could not be unpickled for {path}: {e}") from e

    _time_log(f"{os.path.basename(path)} loaded", time.perf_counter() - t0)
    return obj



def _safe_joblib_load(path: str, timeout_s: float = 10.0) -> Any:
    """Safely load with joblib in a subprocess (hard timeout)."""

    import subprocess
    import sys

    code = r"""
import sys, joblib, pickle
path = sys.argv[1]
obj = joblib.load(path)
sys.stdout.buffer.write(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
"""

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(f"Loading timed out for {path} after {timeout_s}s") from e

    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"joblib.load failed for {path}: {err}")

    import pickle

    try:
        obj = pickle.loads(proc.stdout)
    except Exception as e:
        raise RuntimeError(f"Loaded bytes could not be unpickled for {path}: {e}") from e

    _time_log(f"{os.path.basename(path)} loaded", time.perf_counter() - t0)
    return obj



def _is_probably_corrupt(path: str) -> bool:
    """Heuristic corruption/incompatibility detection based on file size/header."""
    if not os.path.exists(path):
        return True
    size = os.path.getsize(path)
    if size < 1024:  # far too small for a TF-IDF + model bundle
        return True
    try:
        with open(path, "rb") as f:
            head = f.read(2)
        # pickle protocol typically starts with 0x80 (protocol 2+). We'll be permissive.
        return head != b"\x80\x04" and head != b"\x80\x03" and head != b"\x80\x05" and head != b"\x80\x02"
    except Exception:
        return True


def _retrain_lightweight_logreg() -> tuple[Any, Any]:
    """Retrain a lightweight LogisticRegression + TF-IDF from available CSV data."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    from preprocessing import preprocess_for_tfidf

    # Prefer train.csv + valid.csv; fall back to test.csv
    train_csv = "train.csv" if os.path.exists("train.csv") else None
    valid_csv = "valid.csv" if os.path.exists("valid.csv") else None
    test_csv = "test.csv" if os.path.exists("test.csv") else None

    if train_csv is None:
        raise FileNotFoundError("Cannot retrain: train.csv not found in project root")

    # Load data
    import pandas as pd

    def _load_any(csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
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

    train_df = _load_any(train_csv)
    if valid_csv is not None:
        valid_df = _load_any(valid_csv)
    elif test_csv is not None:
        valid_df = _load_any(test_csv)
    else:
        valid_df = None

    # Build training corpus
    train_df["clean"] = train_df["text"].astype(str).apply(preprocess_for_tfidf)
    if valid_df is not None:
        valid_df["clean"] = valid_df["text"].astype(str).apply(preprocess_for_tfidf)

    vectorizer = TfidfVectorizer(max_features=20000)
    X_train = vectorizer.fit_transform(train_df["clean"])
    y_train = train_df["label"].values

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    # Save fresh artifacts
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("[TIME] Retraining complete; saved model.pkl and vectorizer.pkl", flush=True)
    return model, vectorizer


def _load_model_and_vectorizer() -> tuple[Any, Any]:
    """Load model/vectorizer with corruption detection, dual loader strategy, and fallback retraining."""

    model_path = "model.pkl"
    vect_path = "vectorizer.pkl"

    # Model
    model: Any | None = None
    model_loaded = False
    model_err: Exception | None = None

    if not _is_probably_corrupt(model_path):
        # Try pickle then joblib
        for loader in ("pickle", "joblib"):
            try:
                if loader == "pickle":
                    model = _safe_pickle_load(model_path, timeout_s=10.0)
                else:
                    model = _safe_joblib_load(model_path, timeout_s=10.0)
                model_loaded = True
                break
            except Exception as e:  # noqa: BLE001
                model_err = e

    if not model_loaded:
        print(f"[TIME] model.pkl failed validation/loading: {model_err}. Retraining...", flush=True)
        model, _vectorizer_tmp = _retrain_lightweight_logreg()
    else:
        print("[TIME] model loaded successfully", flush=True)
        _vectorizer_tmp = None

    # Vectorizer (separately)
    vectorizer: Any | None = None
    if not _is_probably_corrupt(vect_path):
        for loader in ("pickle", "joblib"):
            try:
                if loader == "pickle":
                    vectorizer = _safe_pickle_load(vect_path, timeout_s=10.0)
                else:
                    vectorizer = _safe_joblib_load(vect_path, timeout_s=10.0)
                print("[TIME] vectorizer loaded successfully", flush=True)
                break
            except Exception:
                vectorizer = None

    if vectorizer is None:
        # Retrain gives us both; if model already retrained, just load vectorizer.
        if _vectorizer_tmp is not None:
            vectorizer = _vectorizer_tmp
        else:
            print("[TIME] vectorizer.pkl failed validation/loading. Retraining...", flush=True)
            model, vectorizer = _retrain_lightweight_logreg()

    return model, vectorizer


def get_models() -> tuple[object, object]:
    """Return cached (model, vectorizer). Loads from disk only once."""
    global _model, _vectorizer, _loaded, _loading

    if _loaded and _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    with _lock:
        if _loaded and _model is not None and _vectorizer is not None:
            return _model, _vectorizer

        _loading = True
        t0 = time.perf_counter()
        _model, _vectorizer = _load_model_and_vectorizer()
        _loaded = True
        _loading = False
        _time_log("ML artifacts ready (total)", time.perf_counter() - t0)

    return _model, _vectorizer



def is_loaded() -> bool:
    return _loaded


def preload_models_async() -> None:
    """Background warm-up — does not block Flask bind."""

    def _job() -> None:
        try:
            if not _loaded:
                print("[TIME] Background ML preload started", flush=True)
                get_models()
                print("[TIME] Background ML preload finished", flush=True)
        except Exception as e:
            print(f"[TIME] Background ML preload failed: {e}", flush=True)

    threading.Thread(target=_job, daemon=True, name="ml-preload").start()


def preload_models_blocking() -> None:
    """Synchronous warm-up before accepting traffic (optional)."""
    if not _loaded:
        print("[TIME] Blocking ML preload started", flush=True)
        get_models()
