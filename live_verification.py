"""Backward-compatible wrapper — delegates to news_client."""

from __future__ import annotations

from typing import Any

from news_client import RawArticle, build_search_queries, combined_similarity, search_live_news


def search_live_news_hybrid(query: str) -> tuple[list[dict[str, Any]], str | None, str | None]:
    """Legacy interface used by older verification code."""
    result = search_live_news(query)
    raw = [
        {
            "title": a.title,
            "description": a.description,
            "source": a.source,
            "publishedAt": a.published,
            "url": a.url,
            "api_used": a.api_provider,
        }
        for a in result.articles
    ]
    api_used = ", ".join(result.api_used) if result.api_used else None
    return raw, api_used, result.warning


def compute_similarity_to_query(query: str, article: dict[str, Any]) -> float:
    raw = RawArticle(
        title=article.get("title") or "",
        description=article.get("description") or "",
        source=article.get("source") or "Unknown",
        published=article.get("publishedAt") or "",
        url=article.get("url") or "",
        api_provider=article.get("api_used") or "",
    )
    return combined_similarity(query, raw)
