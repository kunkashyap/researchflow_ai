"""
utils/helpers.py
General-purpose utility functions used across the application.
"""

import re
import hashlib
import unicodedata
from datetime import datetime
from typing import Any
from urllib.parse import urlparse


# ---------------------------------------------------------------------------
# URL Utilities
# ---------------------------------------------------------------------------

# Private IP ranges and localhost patterns that must never be scraped
_PRIVATE_IP_PATTERNS = re.compile(
    r"""^(
        localhost
        | 127\.
        | 10\.
        | 192\.168\.
        | 172\.(1[6-9]|2\d|3[01])\.
        | \[::1\]
        | ::1
    )""",
    re.VERBOSE | re.IGNORECASE,
)


def is_safe_url(url: str) -> bool:
    """
    Return True if the URL is safe to scrape.

    Rejects:
    - Non-HTTP(S) schemes
    - Localhost / loopback addresses
    - Private IP ranges (10.x, 192.168.x, 172.16-31.x)

    This protects against SSRF (Server-Side Request Forgery) attacks.
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        hostname = parsed.hostname or ""
        if _PRIVATE_IP_PATTERNS.match(hostname):
            return False
        return True
    except Exception:
        return False


def deduplicate_urls(sources: list[dict]) -> list[dict]:
    """
    Remove duplicate sources based on normalised URL.

    Normalisation strips trailing slashes and query strings to catch
    near-identical URLs pointing to the same resource.
    """
    seen: set[str] = set()
    unique: list[dict] = []
    for source in sources:
        url = source.get("url", "").rstrip("/").split("?")[0]
        if url and url not in seen:
            seen.add(url)
            unique.append(source)
    return unique


def normalise_url(url: str) -> str:
    """Return a canonical form of the URL for deduplication."""
    return url.strip().rstrip("/").split("?")[0].split("#")[0]


# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------

def clean_text(text: str, max_chars: int = 50_000) -> str:
    """
    Normalise and truncate extracted web text.

    Steps:
    1. Unicode normalisation (NFC)
    2. Replace Windows/Mac line endings with Unix
    3. Collapse runs of blank lines to a single blank line
    4. Strip leading/trailing whitespace
    5. Truncate to max_chars to prevent LLM context overflow
    """
    if not text:
        return ""
    # Unicode normalisation
    text = unicodedata.normalize("NFC", text)
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip surrounding whitespace
    text = text.strip()
    # Truncate
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated for length]"
    return text


def truncate_text(text: str, max_chars: int) -> str:
    """Simple truncation with ellipsis."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


# ---------------------------------------------------------------------------
# Identifier Utilities
# ---------------------------------------------------------------------------

def generate_research_id(topic: str) -> str:
    """
    Generate a unique research ID from the topic and current timestamp.
    Format: RES_{YYYYMMDD_HHMMSS}_{6-char-hash}
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hash_part = hashlib.md5(f"{topic}{timestamp}".encode()).hexdigest()[:6].upper()
    return f"RES_{timestamp}_{hash_part}"


def source_id_from_index(index: int) -> str:
    """Return a source identifier like S1, S2, …"""
    return f"S{index + 1}"


# ---------------------------------------------------------------------------
# Formatting Utilities
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Convert a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    """Safely traverse a nested dict/object using a sequence of keys."""
    current = obj
    for key in keys:
        try:
            if isinstance(current, dict):
                current = current[key]
            else:
                current = getattr(current, key)
        except (KeyError, AttributeError, TypeError):
            return default
    return current
