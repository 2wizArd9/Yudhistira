#!/usr/bin/env python3
"""
Interactive setup: create .env with NEWS_API_KEY and/or GNEWS_API_KEY.

Usage:
  python setup_api_keys.py

Get free API keys:
  GNews:   https://gnews.io/
  NewsAPI: https://newsapi.org/
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
EXAMPLE_PATH = ROOT / ".env.example"


def main() -> None:
    print("=" * 60)
    print("  Veritas AI — Live News API Setup")
    print("=" * 60)
    print()
    print("Add at least ONE key for live fact verification.")
    print("  GNews:   https://gnews.io/")
    print("  NewsAPI: https://newsapi.org/")
    print()

    existing: dict[str, str] = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip().strip('"').strip("'")
        print(f"Found existing {ENV_PATH} — press Enter to keep current values.")
        print()

    def prompt(name: str, hint: str) -> str:
        current = existing.get(name, "")
        masked = f"{current[:4]}...{current[-4:]}" if len(current) > 8 else ("(set)" if current else "(empty)")
        val = input(f"  {name} [{masked}]: ").strip()
        return val if val else current

    gnews = prompt("GNEWS_API_KEY", "GNews API key")
    newsapi = prompt("NEWS_API_KEY", "NewsAPI key")

    if not gnews and not newsapi:
        print()
        print("ERROR: At least one API key is required for live verification.")
        print("Re-run this script after obtaining a key.")
        return

    lines = [
        "# Veritas AI — Live news verification (auto-generated)",
        "# Do not commit this file to public repositories.",
        "",
    ]
    if gnews:
        lines.append(f"GNEWS_API_KEY={gnews}")
    if newsapi:
        lines.append(f"NEWS_API_KEY={newsapi}")
    lines.append("")

    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")
    print()
    print(f"Saved configuration to: {ENV_PATH}")
    print()
    print("Testing configuration...")
    test_keys()


def test_keys() -> None:
    from api_config import ensure_env_loaded, get_api_config, log_startup_status
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ensure_env_loaded()
    log_startup_status()
    cfg = get_api_config(reload=True)

    if not cfg.is_live_enabled:
        print("WARNING: Keys were saved but did not pass validation (placeholder text?).")
        return

    try:
        from news_client import search_live_news
        result = search_live_news("NASA space mission launch")
        print(f"Test search: fetched {result.total_fetched} articles, "
              f"unique {len(result.articles)}, APIs={result.api_used}")
        if result.warning and not result.articles:
            print(f"Warning: {result.warning}")
        elif result.articles:
            print(f"Sample: {result.articles[0].source} — {result.articles[0].title[:60]}...")
            print("SUCCESS: Live verification is working.")
    except Exception as e:
        print(f"Test search failed: {e}")


if __name__ == "__main__":
    main()
