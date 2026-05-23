"""Analytics figures for Veritas AI.

Server-rendered Plotly (no client fetch needed).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("veritas.analytics")

import plotly.graph_objects as go

from history import history_stats, load_history



def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _placeholder_plot(title: str, subtitle: str = "") -> str:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=f"{title}{'<br>' + subtitle if subtitle else ''}",
        showarrow=False,
        font=dict(color="#e8f0ff"),
        align="center",
        xref="paper",
        yref="paper",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
        font=dict(color="#e8f0ff"),
        showlegend=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig.to_html(full_html=False, include_plotlyjs=False)


def build_analytics_figures(limit: int = 200) -> dict[str, str]:
    history = load_history(limit=limit)
    logger.debug("[ANALYTICS] loaded %s history records", len(history))
    stats = history_stats(history)

    # Always return all required keys (even when history is empty)
    if not history:
        return {
            "fake_real": _placeholder_plot(
                "No history yet", "Run a prediction to populate analytics."
            ),
            "availability": _placeholder_plot(
                "No verification history", "Live verification starts after /predict calls."
            ),
            "trend": _placeholder_plot(
                "No trend data", "Confidence trend will appear after multiple verifications."
            ),
            "trusted_avg": _placeholder_plot(
                "No trusted matches", "Trusted stats appear after sources are verified."
            ),
            "stats_json": json.dumps(stats),
        }

    # Fake vs Real
    logger.debug("[ANALYTICS] generating fake_real chart")
    fig1 = go.Figure(
        data=[
            go.Bar(

                x=["REAL", "FAKE"],
                y=[stats["real"], stats["fake"]],
                marker_color=["rgba(16,185,129,0.85)", "rgba(239,68,68,0.85)"],
            )
        ]
    )
    fig1.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f0ff"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
    )
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(gridcolor="rgba(99,179,237,0.12)")

    # Trusted sources average + API availability
    logger.debug("[ANALYTICS] generating availability chart")
    api_total = max(1, stats["count"])

    api_rate = stats["api_available"] / api_total

    fig2 = go.Figure()
    fig2.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=round(api_rate * 100, 1),
            title={"text": "Live verification availability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "rgba(34,211,238,0.9)"},
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                    {"range": [50, 100], "color": "rgba(16,185,129,0.15)"},
                ],
            },
        )
    )
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f0ff"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
    )

    # Recent trend (confidence over time)
    logger.debug("[ANALYTICS] generating trend chart")
    # Use last N entries

    xs = []
    ys = []
    for h in history[-30:]:
        xs.append(str(h.get("timestamp_utc", ""))[:10] or "")
        ys.append(_safe_float(h.get("final_confidence")))

    fig3 = go.Figure(
        data=[go.Scatter(x=xs, y=ys, mode="lines+markers", line=dict(color="#22d3ee"))]
    )
    fig3.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f0ff"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
    )
    fig3.update_xaxes(showgrid=False)
    fig3.update_yaxes(gridcolor="rgba(99,179,237,0.12)")

    # Trusted distribution (average trusted count)
    logger.debug("[ANALYTICS] generating trusted_avg chart")
    fig4 = go.Figure()

    fig4.add_trace(
        go.Indicator(
            mode="number+delta",
            value=round(stats["trusted_count_avg"], 1),
            delta={"reference": 2, "valueformat": ".1f"},
            title={"text": "Avg trusted matches / query"},
        )
    )
    fig4.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e8f0ff"),
        margin=dict(l=10, r=10, t=30, b=10),
        height=280,
    )

    # Convert to HTML snippets (div+script) for embedding
    return {
        "fake_real": fig1.to_html(full_html=False, include_plotlyjs="cdn"),
        "availability": fig2.to_html(full_html=False, include_plotlyjs="cdn"),
        "trend": fig3.to_html(full_html=False, include_plotlyjs="cdn"),
        "trusted_avg": fig4.to_html(full_html=False, include_plotlyjs="cdn"),
        "stats_json": json.dumps(stats),
    }