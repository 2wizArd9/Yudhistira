"""
Live news search via GNews and NewsAPI.
Runs only on /predict — never at Flask startup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from concurrent.futures import wait as futures_wait
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import requests

from api_config import ensure_env_loaded, get_api_config

logger = logging.getLogger("veritas.news")

STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "that", "this", "these", "those", "it", "its", "as", "by", "with", "from",
    "about", "into", "over", "after", "before", "not", "no", "yes", "said",
    "says", "say", "new", "just", "more", "than", "when", "who", "what", "how",
})

# Strict limits — never block /predict indefinitely
REQUEST_TIMEOUT = 5  # seconds (connect + read total for requests library)
HTTP_TIMEOUT = (2, 5)  # (connect, read) cap
MAX_ARTICLES_PER_API = 5
MAX_ARTICLES_TOTAL = 5
CACHE_TTL = 600
MAX_LIVE_SEARCH_SECONDS = 5.0
MAX_SEARCH_QUERIES = 1


@dataclass
class RawArticle:
    title: str
    description: str
    source: str
    published: str
    url: str
    api_provider: str


@dataclass
class SearchResult:
    articles: list[RawArticle] = field(default_factory=list)
    api_used: list[str] = field(default_factory=list)
    warning: str | None = None
    total_fetched: int = 0
    queries_tried: list[str] = field(default_factory=list)


_cache: dict[str, tuple[float, SearchResult]] = {}


def _debug(msg: str) -> None:
    print(msg, flush=True)
    logger.info(msg)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_keywords(text: str, max_keywords: int = 8) -> list[str]:
    words = normalize_text(text).split()
    keywords = [w for w in words if len(w) > 2 and w not in STOPWORDS]
    seen: set[str] = set()
    unique: list[str] = []
    for w in keywords:
        if w not in seen:
            seen.add(w)
            unique.append(w)
        if len(unique) >= max_keywords:
            break
    return unique


def build_search_queries(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    words = normalized.split()
    queries: list[str] = []
    headline = " ".join(words[:12])
    if headline:
        queries.append(headline)
    kw = extract_keywords(text, max_keywords=8)
    if kw:
        keyword_query = " ".join(kw[:6])
        if keyword_query and keyword_query not in queries:
            queries.append(keyword_query)
    return queries[:MAX_SEARCH_QUERIES]


def combined_similarity(query: str, article: RawArticle) -> float:
    """Hybrid similarity for live verification.

    Uses normalized title+description, adds stopword-ignored overlap,
    and fuzzy SequenceMatcher ratios.

    Note: RawArticle currently carries title/description; some providers may
    also inject more text into description (or content truncated into it).
    """
    q = normalize_text(query)
    if not q:
        return 0.0

    title = normalize_text(article.title)
    desc = normalize_text(article.description)

    # Stopword-ignored text for overlap/fuzzy
    def remove_stops(s: str) -> str:
        if not s:
            return ""
        toks = [t for t in s.split() if t not in STOPWORDS]
        return " ".join(toks)

    q_nostop = remove_stops(q)
    title_nostop = remove_stops(title)
    desc_nostop = remove_stops(desc)

    combined = f"{title} {desc}".strip()
    combined_nostop = f"{title_nostop} {desc_nostop}".strip()

    def ratio(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        return SequenceMatcher(None, a, b).ratio()

    q_words = set(q_nostop.split())
    combined_words = set(combined_nostop.split())
    overlap = len(q_words & combined_words) / max(len(q_words), 1)
    overlap_score = min(1.0, overlap * 1.25)

    # Fuzzy matching across multiple views
    title_score = ratio(q_nostop, title_nostop)
    desc_score = ratio(q_nostop, desc_nostop)
    full_score = ratio(q_nostop, combined_nostop)

    # Title exact-ish bonus (helps headlines like "NASA launched Artemis")
    # without overfitting.
    title_token_overlap = len(set(title_nostop.split()) & set(q_nostop.split())) / max(
        len(set(q_nostop.split())), 1
    )
    title_token_bonus = min(1.0, title_token_overlap * 1.15)

    score = (
        title_score * 0.40
        + desc_score * 0.25
        + full_score * 0.20
        + overlap_score * 0.10
        + title_token_bonus * 0.05
    )

    return max(0.0, min(1.0, score))



def _cache_get(key: str) -> SearchResult | None:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, result = entry
    if time.time() - ts > CACHE_TTL:
        _cache.pop(key, None)
        return None
    return result


def _cache_set(key: str, result: SearchResult) -> None:
    if len(_cache) > 100:
        oldest = min(_cache.keys(), key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[key] = (time.time(), result)


def _safe_http_get(url: str, params: dict[str, Any], provider: str) -> dict[str, Any]:
    """HTTP GET with strict timeout and safe JSON parse."""
    _debug(f"API request sent — {provider}")
    try:
        resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        try:
            return resp.json()
        except json.JSONDecodeError as e:
            _debug(f"API invalid JSON — {provider}: {e}")
            raise ValueError(f"{provider}: invalid JSON response") from e
    except requests.Timeout:
        _debug(f"API timeout — {provider}")
        raise
    except requests.ConnectionError as e:
        _debug(f"API connection error — {provider}: {e}")
        raise
    except requests.RequestException as e:
        _debug(f"API request failed — {provider}: {e}")
        raise


def _parse_gnews(data: dict[str, Any]) -> list[RawArticle]:
    articles: list[RawArticle] = []
    for item in (data.get("articles") or [])[:MAX_ARTICLES_PER_API]:
        title = (item.get("title") or "").strip()
        if not title or title.lower() == "[removed]":
            continue
        articles.append(
            RawArticle(
                title=title,
                description=(item.get("description") or "").strip(),
                source=(item.get("source") or {}).get("name") or "Unknown",
                published=(item.get("publishedAt") or "")[:10],
                url=(item.get("url") or "").strip(),
                api_provider="GNews",
            )
        )
    return articles


def _parse_newsapi(data: dict[str, Any]) -> list[RawArticle]:
    if data.get("status") == "error":
        raise RuntimeError(data.get("message", "NewsAPI error"))
    articles: list[RawArticle] = []
    for item in (data.get("articles") or [])[:MAX_ARTICLES_PER_API]:
        title = (item.get("title") or "").strip()
        if not title or title.lower() == "[removed]":
            continue
        articles.append(
            RawArticle(
                title=title,
                description=(item.get("description") or item.get("content") or "")[:500].strip(),
                source=(item.get("source") or {}).get("name") or "Unknown",
                published=(item.get("publishedAt") or "")[:10],
                url=(item.get("url") or "").strip(),
                api_provider="NewsAPI",
            )
        )
    return articles


def _fetch_gnews(query: str, api_key: str) -> list[RawArticle]:
    data = _safe_http_get(
        "https://gnews.io/api/v4/search",
        {"q": query, "lang": "en", "max": MAX_ARTICLES_PER_API, "sortby": "publishedAt", "apikey": api_key},
        "GNews",
    )
    articles = _parse_gnews(data)
    logger.info("[GNews] SUCCESS — articles=%d", len(articles))
    return articles


def _fetch_newsapi(query: str, api_key: str) -> list[RawArticle]:
    data = _safe_http_get(
        "https://newsapi.org/v2/everything",
        {
            "q": query,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": MAX_ARTICLES_PER_API,
            "apiKey": api_key,
        },
        "NewsAPI",
    )
    articles = _parse_newsapi(data)
    logger.info("[NewsAPI] SUCCESS — articles=%d", len(articles))
    return articles


def _dedupe_articles(articles: list[RawArticle]) -> list[RawArticle]:
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    out: list[RawArticle] = []
    for a in articles:
        url_key = a.url.lower()
        title_key = hashlib.sha256(normalize_text(a.title).encode()).hexdigest()[:16]
        if a.url and url_key in seen_urls:
            continue
        if title_key in seen_titles:
            continue
        if a.url:
            seen_urls.add(url_key)
        seen_titles.add(title_key)
        out.append(a)
        if len(out) >= MAX_ARTICLES_TOTAL:
            break
    return out


def _search_apis_parallel(query: str, cfg) -> tuple[list[RawArticle], list[str], list[str]]:
    """Fetch from all configured APIs in parallel with hard deadline."""
    all_articles: list[RawArticle] = []
    apis_used: list[str] = []
    errors: list[str] = []

    def _gnews_job():
        return _fetch_gnews(query, cfg.gnews_api_key)

    def _newsapi_job():
        return _fetch_newsapi(query, cfg.news_api_key)

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        if cfg.has_gnews:
            futures[pool.submit(_gnews_job)] = "GNews"
        if cfg.has_newsapi:
            futures[pool.submit(_newsapi_job)] = "NewsAPI"

        if not futures:
            return [], [], []

        done, not_done = futures_wait(
            futures.keys(),
            timeout=MAX_LIVE_SEARCH_SECONDS,
            return_when="ALL_COMPLETED",
        )

        for fut in not_done:
            provider = futures[fut]
            fut.cancel()
            _debug(f"API timeout — {provider} (overall deadline)")
            errors.append(f"{provider}: timed out.")

        for fut in done:
            provider = futures[fut]
            try:
                batch = fut.result(timeout=0)
                all_articles.extend(batch)
                apis_used.append(provider)
            except requests.Timeout:
                _debug(f"API timeout — {provider}")
                errors.append(f"{provider}: request timed out.")
            except (requests.RequestException, ValueError, RuntimeError, json.JSONDecodeError) as e:
                errors.append(f"{provider}: {e}")
            except Exception as e:
                errors.append(f"{provider}: {e}")

    return all_articles, apis_used, errors


def search_live_news(text: str) -> SearchResult:
    """Search live news — max ~5 seconds, max 5 articles."""
    _debug("Starting live verification")
    ensure_env_loaded()
    cfg = get_api_config()

    if not cfg.is_live_enabled:
        _debug("Returning ML fallback — no API keys")
        return SearchResult(
            warning=(
                "No live API keys configured. Add NEWS_API_KEY and/or GNEWS_API_KEY to "
                f"{cfg.env_path} (see .env.example)."
            ),
        )

    queries = build_search_queries(text)[:MAX_SEARCH_QUERIES]
    if not queries:
        return SearchResult(warning="Could not build a search query from the input text.")

    cache_key = hashlib.sha256("|".join(queries).encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached:
        _debug(f"Live verification cache hit — {len(cached.articles)} articles")
        return cached

    query = queries[0]
    t0 = time.monotonic()

    try:
        all_articles, apis_used, errors = _search_apis_parallel(query, cfg)
    except FuturesTimeout:
        _debug("API timeout — overall live search")
        return SearchResult(
            warning="Live verification timed out. Using ML analysis only.",
            queries_tried=queries,
        )
    except Exception as e:
        _debug(f"Returning ML fallback — live search error: {e}")
        return SearchResult(
            warning=f"Live verification failed ({e}). Using ML analysis only.",
            queries_tried=queries,
        )

    elapsed = time.monotonic() - t0
    merged = _dedupe_articles(all_articles)[:MAX_ARTICLES_TOTAL]

    _debug(
        f"Live verification done in {elapsed:.2f}s — "
        f"fetched={len(all_articles)} unique={len(merged)} APIs={apis_used or 'none'}"
    )

    warning = None
    if not merged:
        warning = "No trusted live sources found for this query."
        if errors:
            warning += " " + "; ".join(errors[:2])
    elif errors:
        logger.warning("[Live Search] Partial errors: %s", errors)

    result = SearchResult(
        articles=merged,
        api_used=apis_used,
        warning=warning,
        total_fetched=len(all_articles),
        queries_tried=queries,
    )
    _cache_set(cache_key, result)
    return result
