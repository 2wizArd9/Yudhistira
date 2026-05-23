"""
Secure API key configuration for live news verification.

Loads NEWS_API_KEY and GNEWS_API_KEY from:
  1. Environment variables (already set)
  2. Project-root .env file (via python-dotenv)

Never logs or exposes full API keys.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("veritas.api")

# Project root (directory containing front.py)
PROJECT_ROOT = Path(__file__).resolve().parent
ENV_FILE = PROJECT_ROOT / ".env"

PLACEHOLDER_PATTERNS = (
    "your_",
    "your-",
    "xxx",
    "paste_",
    "insert_",
    "api_key_here",
    "changeme",
    "replace_me",
    "<",
    ">",
)


@dataclass(frozen=True)
class ApiConfig:
    news_api_key: str
    gnews_api_key: str
    env_loaded: bool
    env_path: str

    @property
    def has_newsapi(self) -> bool:
        return _is_valid_key(self.news_api_key)

    @property
    def has_gnews(self) -> bool:
        return _is_valid_key(self.gnews_api_key)

    @property
    def is_live_enabled(self) -> bool:
        return self.has_newsapi or self.has_gnews

    def masked_status(self) -> dict[str, str]:
        return {
            "NEWS_API_KEY": _mask(self.news_api_key),
            "GNEWS_API_KEY": _mask(self.gnews_api_key),
            "live_verification": "enabled" if self.is_live_enabled else "disabled",
            "env_file": self.env_path,
        }


def _mask(key: str) -> str:
    if not key:
        return "(not set)"
    if len(key) <= 8:
        return "****"
    return f"{key[:4]}...{key[-4:]}"


def _is_valid_key(key: str) -> bool:
    if not key or len(key.strip()) < 8:
        return False
    lower = key.lower().strip()
    for pat in PLACEHOLDER_PATTERNS:
        if pat in lower:
            return False
    return True


def ensure_env_loaded() -> bool:
    """Load .env from project root. Returns True if file was found."""
    if "NEWS_API_KEY" in os.environ and "GNEWS_API_KEY" in os.environ:
        return ENV_FILE.exists()

    try:
        from dotenv import load_dotenv
        loaded = load_dotenv(ENV_FILE, override=False)
        if loaded:
            logger.info("[API Config] Loaded environment from %s", ENV_FILE)
        elif ENV_FILE.exists():
            load_dotenv(ENV_FILE, override=True)
            logger.info("[API Config] Loaded environment from %s (override)", ENV_FILE)
        return ENV_FILE.exists()
    except ImportError:
        if ENV_FILE.exists():
            _load_env_manual(ENV_FILE)
            logger.info("[API Config] Loaded .env manually (python-dotenv not installed)")
            return True
        return False


def _load_env_manual(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_config(reload: bool = False) -> ApiConfig:
    """Return current API configuration (loads .env on first call)."""
    if reload or not getattr(get_api_config, "_cached", None):
        ensure_env_loaded()
        cfg = ApiConfig(
            news_api_key=os.environ.get("NEWS_API_KEY", "").strip(),
            gnews_api_key=os.environ.get("GNEWS_API_KEY", "").strip(),
            env_loaded=ENV_FILE.exists(),
            env_path=str(ENV_FILE),
        )
        get_api_config._cached = cfg  # type: ignore[attr-defined]
        return cfg
    return get_api_config._cached  # type: ignore[attr-defined]


def log_startup_status() -> None:
    """Print API configuration status at startup (masked)."""
    cfg = get_api_config()
    status = cfg.masked_status()
    logger.info("[API Config] NEWS_API_KEY=%s", status["NEWS_API_KEY"])
    logger.info("[API Config] GNEWS_API_KEY=%s", status["GNEWS_API_KEY"])
    logger.info("[API Config] Live verification: %s", status["live_verification"])
    if not cfg.is_live_enabled:
        logger.warning(
            "[API Config] No valid API keys. Copy .env.example to .env and add keys. "
            "Run: python setup_api_keys.py"
        )
    else:
        providers = []
        if cfg.has_gnews:
            providers.append("GNews")
        if cfg.has_newsapi:
            providers.append("NewsAPI")
        logger.info("[API Config] Active providers: %s", ", ".join(providers))
