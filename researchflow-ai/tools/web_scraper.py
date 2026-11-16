"""
tools/web_scraper.py
BeautifulSoup-based web content extractor for the Reader Agent.

Security notes:
- All scraped content is treated as UNTRUSTED DATA.
- Content is never executed, evaluated or passed as instructions to agents
  without first being clearly framed as "research evidence."
- Localhost and private IPs are blocked before any HTTP request is made.
- robots.txt is checked to respect website access policies.
- Request timeouts and content size limits prevent resource exhaustion.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from models.schemas import ExtractedDocument, ResearchSource
from utils.config import config
from utils.helpers import clean_text, is_safe_url
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.scraper")

# User-Agent: identifies us as a research tool, not a browser
_USER_AGENT = (
    "ResearchFlowAI/1.0 (Autonomous Research Assistant; "
    "contact: research@researchflow.ai)"
)

# HTML elements that contain noise rather than content
_NOISE_TAGS = {
    "script", "style", "nav", "header", "footer", "aside",
    "form", "button", "input", "select", "iframe", "noscript",
    "advertisement", "ads", "cookie-banner",
}

# Heading tags to extract as structured headings
_HEADING_TAGS = {"h1", "h2", "h3", "h4"}


def _build_session() -> requests.Session:
    """
    Build a requests.Session with:
    - Automatic retries for transient 5xx errors (NOT for 4xx — those are real failures)
    - Custom User-Agent
    - Connection pooling
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=1.0,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": _USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    })
    return session


def _can_fetch(url: str) -> bool:
    """
    Check robots.txt to determine whether we are permitted to fetch this URL.

    Returns True if robots.txt allows access or if robots.txt cannot be fetched.
    Returns False if robots.txt explicitly disallows our User-Agent.

    This is a best-effort check: failures (timeout, parse error) default to allowed.
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(_USER_AGENT, url)
    except Exception:
        # Failing to read robots.txt should not block scraping
        return True


def _extract_content(html: str, url: str) -> tuple[str, list[str]]:
    """
    Parse HTML with BeautifulSoup and extract clean readable text.

    Returns:
        (extracted_text, headings_list)
    """
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    # Remove noise elements in-place
    for tag in _NOISE_TAGS:
        for element in soup.find_all(tag):
            element.decompose()

    # Remove comment nodes
    for comment in soup.find_all(string=lambda text: isinstance(text, str) and "<!--" in text):
        comment.extract()

    # Extract headings (preserve structure)
    headings: list[str] = []
    for heading in soup.find_all(_HEADING_TAGS):
        text = heading.get_text(strip=True)
        if text and len(text) < 300:
            headings.append(text)

    # Extract main text
    # Prefer <article>, <main>, <div role="main"> for content-rich areas
    main_content = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", role="main")
        or soup.find("div", class_=re.compile(r"content|article|post|entry", re.I))
        or soup.body
        or soup
    )

    raw_text = main_content.get_text(separator="\n") if main_content else soup.get_text(separator="\n")
    cleaned = clean_text(raw_text, max_chars=config.SCRAPE_MAX_CONTENT_BYTES // 2)

    return cleaned, headings


def scrape_url(source: ResearchSource) -> ExtractedDocument:
    """
    Attempt to scrape a single URL and return an ExtractedDocument.

    Pipeline:
    1. Validate URL is safe (SSRF protection)
    2. Check robots.txt
    3. Fetch with timeout + size limit
    4. Parse with BeautifulSoup
    5. Fall back to Tavily content if scraping fails
    """
    url = source.url
    source_id = source.source_id

    # --- Safety check: reject localhost and private IPs ---
    if not is_safe_url(url):
        logger.warning("Blocked unsafe URL: %s", url)
        return ExtractedDocument(
            source_id=source_id,
            title=source.title,
            url=url,
            extraction_status="failed",
            failure_reason="URL rejected: localhost or private IP range",
        )

    # --- robots.txt check ---
    if not _can_fetch(url):
        logger.info("robots.txt disallows: %s", url)
        # Use Tavily fallback content if available
        if source.tavily_content:
            return ExtractedDocument(
                source_id=source_id,
                title=source.title,
                url=url,
                extracted_text=clean_text(source.tavily_content),
                extraction_status="fallback",
                failure_reason="robots.txt disallowed direct access; using Tavily content",
            )
        return ExtractedDocument(
            source_id=source_id,
            title=source.title,
            url=url,
            extraction_status="failed",
            failure_reason="robots.txt disallowed access and no Tavily fallback available",
        )

    # --- HTTP fetch ---
    session = _build_session()
    try:
        response = session.get(
            url,
            timeout=config.SCRAPE_TIMEOUT,
            stream=True,  # Stream so we can check size before full download
        )
        response.raise_for_status()

        # Content-type check: only process HTML
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "text/plain" not in content_type:
            raise ValueError(f"Non-HTML content type: {content_type}")

        # Read up to SCRAPE_MAX_CONTENT_BYTES
        content_bytes = b""
        for chunk in response.iter_content(chunk_size=4096):
            content_bytes += chunk
            if len(content_bytes) > config.SCRAPE_MAX_CONTENT_BYTES:
                logger.info("Content size limit reached for %s, truncating", url)
                break

        # Detect encoding
        encoding = response.encoding or "utf-8"
        html = content_bytes.decode(encoding, errors="replace")

        extracted_text, headings = _extract_content(html, url)

        if not extracted_text.strip():
            raise ValueError("Extraction produced empty content")

        return ExtractedDocument(
            source_id=source_id,
            title=source.title,
            url=url,
            extracted_text=extracted_text,
            headings=headings,
            extraction_status="success",
            content_length=len(extracted_text),
        )

    except requests.exceptions.Timeout:
        failure = f"Request timed out after {config.SCRAPE_TIMEOUT}s"
    except requests.exceptions.TooManyRedirects:
        failure = "Too many redirects"
    except requests.exceptions.HTTPError as exc:
        failure = f"HTTP error: {exc.response.status_code}"
    except requests.exceptions.ConnectionError:
        failure = "Connection error (DNS failure or connection refused)"
    except requests.exceptions.RequestException as exc:
        failure = f"Request error: {exc}"
    except ValueError as exc:
        failure = str(exc)
    except Exception as exc:
        failure = f"Unexpected error: {exc}"

    logger.warning("Scraping failed for %s: %s", url, failure)

    # --- Fallback to Tavily content ---
    if source.tavily_content and source.tavily_content.strip():
        logger.info("Using Tavily fallback content for %s", url)
        return ExtractedDocument(
            source_id=source_id,
            title=source.title,
            url=url,
            extracted_text=clean_text(source.tavily_content),
            extraction_status="fallback",
            failure_reason=failure,
            content_length=len(source.tavily_content),
        )

    return ExtractedDocument(
        source_id=source_id,
        title=source.title,
        url=url,
        extraction_status="failed",
        failure_reason=failure,
    )


def scrape_sources_concurrently(
    sources: list[ResearchSource],
    max_workers: int = None,
    delay_between: float = 0.3,
) -> tuple[list[ExtractedDocument], list[str]]:
    """
    Scrape multiple sources concurrently using a thread pool.

    Args:
        sources:        List of ResearchSource objects to scrape.
        max_workers:    Maximum concurrent threads (defaults to config value).
        delay_between:  Rate limiting delay between individual requests.

    Returns:
        (extracted_documents, failed_source_ids)
    """
    if max_workers is None:
        max_workers = config.MAX_CONCURRENT_SCRAPERS

    extracted: list[ExtractedDocument] = []
    failed_ids: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_source = {
            executor.submit(scrape_url, source): source
            for source in sources
        }
        for future in as_completed(future_to_source):
            source = future_to_source[future]
            try:
                doc = future.result()
                extracted.append(doc)
                if doc.extraction_status == "failed":
                    failed_ids.append(doc.source_id)
            except Exception as exc:
                logger.error("Unexpected error scraping %s: %s", source.url, exc)
                failed_ids.append(source.source_id)
                extracted.append(
                    ExtractedDocument(
                        source_id=source.source_id,
                        title=source.title,
                        url=source.url,
                        extraction_status="failed",
                        failure_reason=str(exc),
                    )
                )

    # Restore original order (concurrent futures return in completion order)
    source_order = {s.source_id: i for i, s in enumerate(sources)}
    extracted.sort(key=lambda d: source_order.get(d.source_id, 999))

    return extracted, failed_ids
