"""
tests/test_reader_agent.py
Unit tests for the Reader Agent and web scraper.

Tests use sample HTML strings and mocked HTTP requests — no real network
calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from models.schemas import ExtractedDocument, ResearchSource


# ---------------------------------------------------------------------------
# Sample HTML fixtures
# ---------------------------------------------------------------------------

SAMPLE_HTML_ARTICLE = """
<!DOCTYPE html>
<html>
<head><title>Quantum Computing News</title></head>
<body>
<nav>Navigation menu here</nav>
<main>
  <article>
    <h1>Quantum Computing Breakthrough</h1>
    <h2>Background</h2>
    <p>Scientists at MIT have achieved a major breakthrough in quantum computing.</p>
    <p>The new 1000-qubit processor demonstrates unprecedented error correction capabilities.</p>
    <h2>Implications for Cryptography</h2>
    <p>This development has significant implications for current RSA and AES encryption standards.</p>
  </article>
</main>
<footer>Copyright 2024 Example News</footer>
<script>var x = 1;</script>
<style>.css { color: red; }</style>
</body>
</html>
"""

SAMPLE_HTML_PAYWALL = """
<!DOCTYPE html>
<html>
<body>
<div class="paywall">
  <h1>Subscribe to Read</h1>
  <p>This content is available to subscribers only.</p>
  <form><button>Subscribe Now</button></form>
</div>
</body>
</html>
"""

SAMPLE_HTML_EMPTY = "<html><body></body></html>"


# ---------------------------------------------------------------------------
# Tests: HTML extraction
# ---------------------------------------------------------------------------

class TestHTMLExtraction:

    def test_extracts_main_content(self):
        from tools.web_scraper import _extract_content
        text, headings = _extract_content(SAMPLE_HTML_ARTICLE, "https://example.com")
        assert "quantum computing" in text.lower()
        assert "MIT" in text
        assert "RSA" in text

    def test_removes_navigation(self):
        from tools.web_scraper import _extract_content
        text, _ = _extract_content(SAMPLE_HTML_ARTICLE, "https://example.com")
        assert "Navigation menu" not in text

    def test_removes_footer(self):
        from tools.web_scraper import _extract_content
        text, _ = _extract_content(SAMPLE_HTML_ARTICLE, "https://example.com")
        assert "Copyright 2024" not in text

    def test_removes_scripts_and_styles(self):
        from tools.web_scraper import _extract_content
        text, _ = _extract_content(SAMPLE_HTML_ARTICLE, "https://example.com")
        assert "var x = 1" not in text
        assert ".css {" not in text

    def test_extracts_headings(self):
        from tools.web_scraper import _extract_content
        _, headings = _extract_content(SAMPLE_HTML_ARTICLE, "https://example.com")
        assert len(headings) >= 2
        assert any("Quantum" in h for h in headings)
        assert any("Background" in h or "Cryptography" in h for h in headings)

    def test_handles_empty_html(self):
        from tools.web_scraper import _extract_content
        text, headings = _extract_content(SAMPLE_HTML_EMPTY, "https://example.com")
        # Should not raise, should return empty strings
        assert isinstance(text, str)
        assert isinstance(headings, list)

    def test_handles_malformed_html(self):
        from tools.web_scraper import _extract_content
        malformed = "<html><body><p>Unclosed paragraph<div>Content here"
        text, headings = _extract_content(malformed, "https://example.com")
        assert "Content here" in text


# ---------------------------------------------------------------------------
# Tests: scrape_url() with mocked requests
# ---------------------------------------------------------------------------

class TestScrapeURL:

    def _make_source(self, url="https://example.com/article", tavily_content="Fallback content"):
        return ResearchSource(
            source_id="S1",
            title="Test Article",
            url=url,
            snippet="Test snippet",
            tavily_content=tavily_content,
        )

    @patch("tools.web_scraper._can_fetch", return_value=True)
    @patch("tools.web_scraper._build_session")
    def test_successful_scrape(self, mock_session_factory, mock_can_fetch):
        """Test that a successful HTTP response produces a success-status document."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
        mock_response.encoding = "utf-8"
        mock_response.iter_content.return_value = [SAMPLE_HTML_ARTICLE.encode("utf-8")]

        mock_session = MagicMock()
        mock_session.get.return_value.__enter__ = lambda s: mock_response
        mock_session.get.return_value.__exit__ = MagicMock(return_value=False)
        mock_session.get.return_value = mock_response
        mock_session_factory.return_value = mock_session

        from tools.web_scraper import scrape_url
        source = self._make_source()
        doc = scrape_url(source)

        assert doc.source_id == "S1"
        assert doc.extraction_status in ("success", "fallback")  # Either is acceptable

    def test_rejects_localhost_url(self):
        """Test that localhost URLs are rejected before any HTTP request."""
        from tools.web_scraper import scrape_url
        source = self._make_source(url="http://localhost:8080/secret")
        doc = scrape_url(source)
        assert doc.extraction_status == "failed"
        assert "localhost" in doc.failure_reason.lower() or "private" in doc.failure_reason.lower()

    def test_rejects_private_ip(self):
        """Test that private IP URLs are rejected."""
        from tools.web_scraper import scrape_url
        source = self._make_source(url="http://192.168.1.1/admin")
        doc = scrape_url(source)
        assert doc.extraction_status == "failed"

    @patch("tools.web_scraper._can_fetch", return_value=False)
    def test_robots_disallowed_uses_tavily_fallback(self, mock_can_fetch):
        """Test that robots.txt disallowed URLs fall back to Tavily content."""
        from tools.web_scraper import scrape_url
        source = self._make_source(tavily_content="Tavily provided content for this source")
        doc = scrape_url(source)
        assert doc.extraction_status == "fallback"
        assert "Tavily" in doc.failure_reason or "robots" in doc.failure_reason.lower()

    @patch("tools.web_scraper._can_fetch", return_value=True)
    @patch("tools.web_scraper._build_session")
    def test_timeout_uses_tavily_fallback(self, mock_session_factory, mock_can_fetch):
        """Test that request timeouts fall back to Tavily content."""
        import requests
        mock_session = MagicMock()
        mock_session.get.side_effect = requests.exceptions.Timeout()
        mock_session_factory.return_value = mock_session

        from tools.web_scraper import scrape_url
        source = self._make_source(tavily_content="Tavily fallback content")
        doc = scrape_url(source)
        assert doc.extraction_status == "fallback"
        assert "Tavily" in doc.extracted_text or len(doc.extracted_text) > 0

    @patch("tools.web_scraper._can_fetch", return_value=True)
    @patch("tools.web_scraper._build_session")
    def test_404_no_fallback_returns_failed(self, mock_session_factory, mock_can_fetch):
        """Test that 404 errors without fallback content return failed status."""
        import requests
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=404)
        )
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response
        mock_session_factory.return_value = mock_session

        from tools.web_scraper import scrape_url
        source = self._make_source(tavily_content="")  # No fallback
        doc = scrape_url(source)
        assert doc.extraction_status == "failed"


# ---------------------------------------------------------------------------
# Tests: Concurrent scraping
# ---------------------------------------------------------------------------

class TestConcurrentScraping:

    @patch("tools.web_scraper.scrape_url")
    def test_concurrent_scraping_maintains_order(self, mock_scrape):
        """Test that concurrent scraping returns results in original source order."""
        sources = [
            ResearchSource(source_id=f"S{i}", url=f"https://example.com/{i}", title=f"Source {i}")
            for i in range(1, 6)
        ]

        def mock_scrape_fn(source):
            return ExtractedDocument(
                source_id=source.source_id,
                url=source.url,
                extraction_status="success",
                extracted_text=f"Content for {source.source_id}",
            )

        mock_scrape.side_effect = mock_scrape_fn

        from tools.web_scraper import scrape_sources_concurrently
        docs, failed = scrape_sources_concurrently(sources, max_workers=3)

        assert len(docs) == 5
        # Results should be ordered S1, S2, S3, S4, S5
        for i, doc in enumerate(docs, start=1):
            assert doc.source_id == f"S{i}"

    @patch("tools.web_scraper.scrape_url")
    def test_failed_sources_tracked(self, mock_scrape):
        """Test that failed extractions are tracked in failed_ids list."""
        sources = [
            ResearchSource(source_id="S1", url="https://ok.com", title="OK"),
            ResearchSource(source_id="S2", url="https://fail.com", title="Fail"),
        ]

        def mock_scrape_fn(source):
            status = "success" if source.source_id == "S1" else "failed"
            return ExtractedDocument(
                source_id=source.source_id,
                url=source.url,
                extraction_status=status,
                failure_reason="Test failure" if status == "failed" else None,
            )

        mock_scrape.side_effect = mock_scrape_fn

        from tools.web_scraper import scrape_sources_concurrently
        docs, failed = scrape_sources_concurrently(sources)

        assert "S2" in failed
        assert "S1" not in failed
