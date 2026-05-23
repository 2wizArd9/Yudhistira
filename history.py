"""Prediction history (JSONL) for Veritas AI.

Lightweight, stable storage for placement/demo purposes.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict


HISTORY_PATH = "verification_history.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _query_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()


def log_verification(event: Dict[str, Any]) -> None:
    """Append one JSON event to history file."""
    try:
        os.makedirs(os.path.dirname(HISTORY_PATH) or ".", exist_ok=True)
        with open(HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # Never break the main app due to logging failures.
        return


def build_history_event(
    *,
    query: str,
    ml_prediction: Any,
    hybrid_prediction: str,
    final_confidence: float,
    ml_score: float | None,
    live_verification_score: float | None,
    source_trust_score: float | None,
    provider_used: str | None,
    trusted_sources_count: int,
    verified_sources_count: int | None,
    api_used: str | None,
    verification_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Build one persisted history event (JSONL).

    Notes:
    - Keep this schema stable because analytics depend on it.
    - Some apps may not produce every score; we normalize to floats/defaults.
    """

    ml_confidence = None
    # Backward-compat: if ml_score is actually the ML confidence, prefer it.
    if ml_score is not None:
        ml_confidence = float(ml_score)

    return {
        "timestamp_utc": _utc_now_iso(),
        "query_hash": _query_hash(query),
        "user_query": query,
        "hybrid_prediction": hybrid_prediction,
        "final_confidence": float(final_confidence or 0.0),

        # requested fields
        "input_text": query,
        "prediction": hybrid_prediction,
        "confidence": float(final_confidence or 0.0),

        "ml_score": float(ml_score) if ml_score is not None else None,
        "live_verification_score": float(live_verification_score) if live_verification_score is not None else None,
        "source_trust_score": float(source_trust_score) if source_trust_score is not None else None,
        "provider_used": provider_used,

        "trusted_sources_count": int(trusted_sources_count or 0),
        "verified_sources_count": int(verified_sources_count) if verified_sources_count is not None else int(trusted_sources_count or 0),

        # backward compat fields
        "ml_prediction": ml_prediction,
        "ml_confidence": ml_confidence,

        "api_status": "available" if api_used else "unavailable",
        "api_used": api_used,

        "verification_result": verification_result,
    }



def load_history(limit: int = 200) -> list[Dict[str, Any]]:
    if not os.path.exists(HISTORY_PATH):
        return []

    rows: list[Dict[str, Any]] = []
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if limit and len(rows) > limit:
            rows = rows[-limit:]
    except Exception:
        return []

    return rows


def history_stats(history: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple stats for analytics rendering."""
    stats = {
        "count": len(history),
        "real": 0,
        "fake": 0,
        "unverified": 0,
        "api_available": 0,
        "trusted_count_avg": 0.0,
    }

    if not history:
        return stats

    real = 0
    fake = 0
    unverified = 0
    api_available = 0
    trusted_sum = 0

    for h in history:
        trusted_sum += int(h.get("trusted_sources_count") or 0)
        api_available += 1 if h.get("api_used") else 0

        pred = (h.get("hybrid_prediction") or "").lower()
        if "real" in pred:
            real += 1
        elif "fake" in pred:
            fake += 1

        vr = h.get("verification_result") or {}
        if vr.get("match_count", 0) == 0:
            unverified += 1

    stats["real"] = real
    stats["fake"] = fake
    stats["unverified"] = unverified
    stats["api_available"] = api_available
    stats["trusted_count_avg"] = trusted_sum / max(1, len(history))

    return stats

