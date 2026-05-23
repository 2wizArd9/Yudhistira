"""ML analysis — fast single-pass inference with timing and timeouts."""

from __future__ import annotations

import re
import threading
import time
from typing import Any

ML_INFERENCE_TIMEOUT = 20.0
PROBA_TIMEOUT = 2.0


def _time_log(step: str, elapsed: float) -> None:
    print(f"[TIME] {step} in {elapsed:.3f}s", flush=True)


def preprocess_text(text: str) -> str:
    t0 = time.perf_counter()
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s.,'\"-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    out = text.strip()
    _time_log("text preprocessing", time.perf_counter() - t0)
    return out


def ml_emergency_fallback(cleaned: str = "", reason: str = "ML inference timeout") -> dict[str, Any]:
    return {
        "label": -1,
        "is_real": False,
        "display": "UNKNOWN",
        "confidence": 50.0,
        "real_score": 0.5,
        "cleaned_text": cleaned,
        "explanation": reason,
        "ml_only": True,
    }


def _get_classifier(model: object) -> object:
    if hasattr(model, "named_steps"):
        steps = list(model.named_steps.values())
        if steps:
            return steps[-1]
    return model


def _transform(vectorizer: object, text: str):
    t0 = time.perf_counter()
    X = vectorizer.transform([text])
    if getattr(X, "ndim", 2) == 1:
        X = X.reshape(1, -1)
    _time_log("vectorizer.transform()", time.perf_counter() - t0)
    return X


def _predict_label_fast(model: object, vectorizer: object, text: str) -> int:
    """Single transform + predict — never reloads pickles."""
    clf = _get_classifier(model)

    # Fast path: full pipeline accepts raw strings
    if hasattr(model, "predict") and hasattr(model, "named_steps"):
        try:
            t0 = time.perf_counter()
            pred = model.predict([text])
            _time_log("model.predict() [pipeline]", time.perf_counter() - t0)
            return int(pred[0])
        except Exception:
            pass

    X = _transform(vectorizer, text)

    t0 = time.perf_counter()
    pred = clf.predict(X)
    _time_log("model.predict()", time.perf_counter() - t0)
    return int(pred[0])


def _predict_proba_safe(model: object, vectorizer: object, text: str, label: int, X=None) -> float | None:
    """predict_proba with thread timeout — falls back to None (use default confidence)."""
    clf = _get_classifier(model)

    def _job() -> float | None:
        try:
            if hasattr(model, "predict_proba") and hasattr(model, "named_steps"):
                proba = model.predict_proba([text])[0]
                return float(proba[label])
        except Exception:
            pass
        try:
            mat = X if X is not None else _transform(vectorizer, text)
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(mat)[0]
                return float(proba[label])
        except Exception:
            pass
        return None

    holder: list[float | None] = [None]
    err: list[Exception | None] = [None]

    def _target() -> None:
        try:
            holder[0] = _job()
        except Exception as e:
            err[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=PROBA_TIMEOUT)
    if t.is_alive():
        print("[TIME] model.predict_proba() timed out — using predict-only fallback", flush=True)
        return None
    if err[0]:
        print(f"[TIME] model.predict_proba() failed: {err[0]}", flush=True)
        return None
    return holder[0]


def _run_ml_core(cleaned: str) -> dict[str, Any]:
    """Core inference — assumes models already in cache."""
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    from model_cache import get_models
    model, vectorizer = get_models()
    print("[ML] using cached model", flush=True)
    _time_log("get_models() [cache]", time.perf_counter() - t0)


    label = _predict_label_fast(model, vectorizer, cleaned)

    # Reuse TF-IDF matrix for proba (avoid second transform)
    X = None
    if not (hasattr(model, "predict") and hasattr(model, "named_steps")):
        t0 = time.perf_counter()
        X = vectorizer.transform([cleaned])
        if getattr(X, "ndim", 2) == 1:
            X = X.reshape(1, -1)
        _time_log("vectorizer.transform() [cached for proba]", time.perf_counter() - t0)

    t0 = time.perf_counter()
    proba = _predict_proba_safe(model, vectorizer, cleaned, label, X=X)
    _time_log("predict_proba phase", time.perf_counter() - t0)

    if proba is not None:
        confidence = round(proba * 100, 1)
        real_score = proba if label == 1 else (1 - proba)
    else:
        confidence = 72.0 if label == 1 else 68.0
        real_score = confidence / 100.0

    is_real = label == 1
    display = "Real ✅" if is_real else "Fake ⚠️"

    elapsed_total = time.perf_counter() - t_total
    print(f"[ML] prediction completed in {elapsed_total:.3f}s", flush=True)
    _time_log("ML prediction completed (total)", elapsed_total)

    return {

        "label": label,
        "is_real": is_real,
        "display": display,
        "confidence": confidence,
        "real_score": real_score,
        "cleaned_text": cleaned,
    }


def get_ml_analysis(news: str) -> dict[str, Any]:
    """
    Run ML pipeline (sync). Never reloads pickles — uses model_cache only.
    Timeout is enforced by run_full_verification() caller.
    """
    t0 = time.perf_counter()
    cleaned = preprocess_text(news)
    try:
        return _run_ml_core(cleaned)
    except Exception as e:
        print(f"[TIME] ML inference error: {e} — emergency fallback", flush=True)
        return ml_emergency_fallback(cleaned, f"ML inference error: {e}")
    finally:
        _time_log("get_ml_analysis() wall time", time.perf_counter() - t0)
