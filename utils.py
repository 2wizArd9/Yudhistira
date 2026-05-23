from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import streamlit as st

from preprocessing import preprocess_for_tfidf
from model_training import MODEL_COMPARISON_CACHE, compute_model_comparison


def run_model_comparison():
    """Backwards-compatible alias used by the Streamlit UI."""
    return compute_model_comparison()





ARTIFACT_MODEL = "model.pkl"
ARTIFACT_VECTORIZER = "vectorizer.pkl"


def load_artifacts() -> Tuple[Any, Any, Dict[str, Any]]:
    """Load trained model + vectorizer artifacts.

    Must exist for the UI to work.
    """
    if not os.path.exists(ARTIFACT_MODEL) or not os.path.exists(ARTIFACT_VECTORIZER):
        st.warning(
            "Model artifacts not found. Please run `python train_front_model.py` to generate model.pkl and vectorizer.pkl."
        )

    with open(ARTIFACT_MODEL, "rb") as f:
        model = pickle.load(f)

    with open(ARTIFACT_VECTORIZER, "rb") as f:
        vectorizer = pickle.load(f)

    return model, vectorizer, {}


def predict_with_explanations(model: Any, vectorizer: Any, text: str) -> Dict[str, Any]:
    cleaned = preprocess_for_tfidf(text)

    # For explainability, we want the TF-IDF vector.
    X = vectorizer.transform([cleaned])
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    else:
        # fallback to decision function
        proba = None

    pred = model.predict(X)[0]

    # Heuristic mapping based on original project convention:
    # In the original code: prediction[0] == 0 => Fake else Real
    # We'll treat class 0 as Fake.
    # scikit-learn may store classes_ ordering; handle robustly.
    classes = getattr(model, "classes_", np.array([0, 1]))

    def get_prob_for_label(target_label):
        if proba is None:
            return None
        # target_label is class value, not index.
        idx = int(np.where(classes == target_label)[0][0])
        return float(proba[idx])

    prob_fake = get_prob_for_label(0)
    prob_real = get_prob_for_label(1)

    # Confidence: if probabilities available, take max(probs)
    if prob_fake is not None and prob_real is not None:
        conf = max(prob_fake, prob_real) * 100.0
    else:
        conf = 50.0

    label = "FAKE NEWS" if int(pred) == 0 else "REAL NEWS"

    # Top TF-IDF contributing tokens based on linear model weights when possible.
    top_features = []
    if hasattr(model, "coef_"):
        # coef_ shape: (1, n_features) or (n_classes, n_features)
        coef = model.coef_
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)

        # Choose coefficients for class 0 vs 1 depending on pred.
        class_index = int(np.where(classes == int(pred))[0][0]) if len(classes) else 0
        weights = coef[class_index]

        # get non-zero tfidf indices
        x_row = X.toarray().ravel()
        nonzero_idx = np.where(x_row != 0)[0]

        # compute contribution score = tfidf * weight
        contrib = x_row[nonzero_idx] * weights[nonzero_idx]

        # pick top 12 by absolute contribution
        order = np.argsort(np.abs(contrib))[::-1][:12]
        feature_names = vectorizer.get_feature_names_out()
        top_features = [
            {
                "token": str(feature_names[nonzero_idx[i]]),
                "score": float(contrib[i]),
            }
            for i in order
        ]

    return {
        "original_text": text,
        "cleaned_text": cleaned,
        "label": label,
        "pred": int(pred),
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "confidence": float(conf),
        "top_features": top_features,
        "vectorizer": vectorizer,
        "tfidf_dim": int(getattr(X, "shape", [0, 0])[1])
        if hasattr(X, "shape")
        else None,
    }


def render_prediction_explainability(result: Dict[str, Any]):
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go

    top_features = result.get("top_features") or []

    c_fake = result.get("prob_fake")
    c_real = result.get("prob_real")

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Probability**")
        if c_fake is not None and c_real is not None:
            vals = [c_fake, c_real]
            labels = ["FAKE", "REAL"]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=[v * 100 for v in vals],
                        marker_color=["#ef4444", "#22c55e"],
                    )
                ]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e7eefc"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=260,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model does not provide predict_proba; showing label only.")

    with col2:
        st.write("**Key TF-IDF tokens**")
        if top_features:
            df = pd.DataFrame(top_features)
            # show top positive vs negative by score
            df_pos = df[df["score"] >= 0].sort_values("score", ascending=False).head(6)
            df_neg = df[df["score"] < 0].sort_values("score").head(6)

            st.markdown("**Influential for the chosen class**")
            if not df.empty:
                st.write(", ".join(df.head(8)["token"].tolist()))

            st.markdown("**Negative/inhibiting signals**")
            if not df_neg.empty:
                st.write(", ".join(df_neg["token"].tolist()))
        else:
            st.info("Explainability weights not available (model has no coef_).")

    st.markdown("---")
    st.write("**Sample explanation**")
    st.info(
        f"The model is most influenced by tokens like: "
        + (", ".join([t["token"] for t in (top_features[:5] if top_features else [])]) if top_features else "(insufficient data)")
        + f". Based on TF‑IDF similarity, it classifies this input as **{result['label']}**."
    )

    # Expanders for preprocessing
    with st.expander("🧼 Preprocessing details", expanded=False):
        st.write("**Original text**")
        st.text_area("Original", result["original_text"], height=120)
        st.write("**Cleaned text (for TF‑IDF)**")
        st.text_area("Cleaned", result["cleaned_text"], height=120)


def render_visualizations():
    # Keep lightweight: we can render generic placeholders.
    # Model-specific dataset visualizations require training/test artifacts.
    st.caption("Visualizations will be computed from available training artifacts when running training.")


