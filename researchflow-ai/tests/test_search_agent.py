"""
tests/test_search_agent.py
Unit tests for the Search Agent.

All tests use mocked external APIs — no real Tavily or OpenAI calls are made.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import ResearchConfiguration, ResearchSource, SearchAgentOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config():
    return ResearchConfiguration(
        research_topic="Quantum computing applications in cryptography",
        search_depth="standard",
        max_sources=5,
        max_revisions=2,
    )


@pytest.fixture
def mock_tavily_results():
    return [
        {
            "title": "Quantum Cryptography Overview",
            "url": "https://example.com/quantum-crypto",
            "content": "Quantum computing poses significant risks to classical cryptography...",
            "score": 0.95,
            "published_date": "2024-01-15",
        },
        {
            "title": "Post-Quantum Encryption Standards",
            "url": "https://nist.gov/pqc",
            "content": "NIST has finalised post-quantum cryptographic standards...",
            "score": 0.92,
            "published_date": "2024-03-01",
        },
        {
            "title": "RSA Vulnerability to Quantum Attacks",
            "url": "https://research.example.org/rsa-quantum",
            "content": "RSA-2048 encryption could be broken by a sufficiently large quantum computer...",
            "score": 0.88,
            "published_date": None,
        },
    ]


# ---------------------------------------------------------------------------
# Tests: Query generation parsing
# ---------------------------------------------------------------------------

class TestQueryParsing:
    """Test the _parse_query_list helper function."""

    def test_parse_valid_json_array(self):
        from agents.search_agent import _parse_query_list
        output = '["quantum encryption", "post-quantum cryptography", "NIST PQC standards"]'
        result = _parse_query_list(output, 3)
        assert len(result) == 3
        assert "quantum encryption" in result

    def test_parse_json_with_markdown_fence(self):
        from agents.search_agent import _parse_query_list
        output = '```json\n["query 1", "query 2"]\n```'
        result = _parse_query_list(output, 2)
        assert len(result) == 2
        assert result[0] == "query 1"

    def test_parse_regex_fallback(self):
        from agents.search_agent import _parse_query_list
        output = 'Here are the queries: "quantum computing security" and "cryptography quantum"'
        result = _parse_query_list(output, 2)
        assert len(result) >= 1
        assert any("quantum" in q for q in result)

    def test_parse_respects_num_queries_cap(self):
        from agents.search_agent import _parse_query_list
        output = '["q1", "q2", "q3", "q4", "q5"]'
        result = _parse_query_list(output, 3)
        assert len(result) == 3

    def test_parse_last_resort_fallback(self):
        from agents.search_agent import _parse_query_list
        # Completely unparseable output
        result = _parse_query_list("I cannot generate queries right now.", 3)
        assert isinstance(result, list)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Tests: SearchAgent.run() with mocked dependencies
# ---------------------------------------------------------------------------

class TestSearchAgent:

    @patch("agents.search_agent.TavilySearchTool")
    @patch("agents.search_agent.ChatOpenAI")
    def test_successful_search(self, mock_llm_class, mock_tavily_class, sample_config, mock_tavily_results):
        """Test that SearchAgent correctly orchestrates query generation and Tavily search."""
        # Mock LLM to return 4 queries as JSON
        mock_llm = MagicMock()
        mock_llm_class.return_value = mock_llm
        mock_chain_output = '["query 1", "query 2", "query 3", "query 4"]'

        # Mock LCEL chain
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = mock_chain_output

        # Mock Tavily
        mock_tavily = MagicMock()
        mock_tavily_class.return_value = mock_tavily

        sources = [
            ResearchSource(
                source_id="S1",
                title=r["title"],
                url=r["url"],
                snippet=r["content"][:200],
                relevance_score=r["score"],
            )
            for r in mock_tavily_results
        ]
        mock_tavily.search_multiple.return_value = sources

        # Import after patching
        from agents.search_agent import SearchAgent
        agent = SearchAgent()
        agent._query_chain = mock_chain

        result = agent.run(sample_config)

        assert isinstance(result, SearchAgentOutput)
        assert result.status == "completed"
        assert len(result.sources) == 3
        assert result.sources[0].source_id == "S1"

    @patch("agents.search_agent.TavilySearchTool")
    @patch("agents.search_agent.ChatOpenAI")
    def test_tavily_failure_returns_failed_output(self, mock_llm_class, mock_tavily_class, sample_config):
        """Test that Tavily API failures are handled gracefully."""
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = '["query 1"]'

        mock_tavily = MagicMock()
        mock_tavily_class.return_value = mock_tavily
        mock_tavily.search_multiple.side_effect = Exception("Tavily API unavailable")

        from agents.search_agent import SearchAgent
        agent = SearchAgent()
        agent._query_chain = mock_chain

        result = agent.run(sample_config)

        assert result.status == "failed"
        assert "Tavily API unavailable" in result.error
        assert len(result.sources) == 0


# ---------------------------------------------------------------------------
# Tests: URL deduplication
# ---------------------------------------------------------------------------

class TestURLDeduplication:

    def test_deduplicate_removes_exact_duplicates(self):
        from utils.helpers import deduplicate_urls
        sources = [
            {"url": "https://example.com/page"},
            {"url": "https://example.com/page"},
            {"url": "https://other.com/page"},
        ]
        result = deduplicate_urls(sources)
        assert len(result) == 2

    def test_deduplicate_normalises_trailing_slash(self):
        from utils.helpers import deduplicate_urls
        sources = [
            {"url": "https://example.com/page"},
            {"url": "https://example.com/page/"},
        ]
        result = deduplicate_urls(sources)
        assert len(result) == 1

    def test_deduplicate_normalises_query_string(self):
        from utils.helpers import deduplicate_urls
        sources = [
            {"url": "https://example.com/page?ref=twitter"},
            {"url": "https://example.com/page?ref=google"},
        ]
        result = deduplicate_urls(sources)
        assert len(result) == 1

    def test_deduplicate_preserves_order(self):
        from utils.helpers import deduplicate_urls
        sources = [
            {"url": "https://a.com"},
            {"url": "https://b.com"},
            {"url": "https://c.com"},
        ]
        result = deduplicate_urls(sources)
        assert [r["url"] for r in result] == ["https://a.com", "https://b.com", "https://c.com"]

    def test_safe_url_rejects_localhost(self):
        from utils.helpers import is_safe_url
        assert not is_safe_url("http://localhost:8080/admin")
        assert not is_safe_url("http://127.0.0.1/secret")

    def test_safe_url_rejects_private_ip(self):
        from utils.helpers import is_safe_url
        assert not is_safe_url("http://192.168.1.1/api")
        assert not is_safe_url("http://10.0.0.1/data")
        assert not is_safe_url("http://172.16.0.1/internal")

    def test_safe_url_accepts_public_https(self):
        from utils.helpers import is_safe_url
        assert is_safe_url("https://www.example.com/article")
        assert is_safe_url("https://arxiv.org/abs/2024.12345")

    def test_safe_url_rejects_non_http_schemes(self):
        from utils.helpers import is_safe_url
        assert not is_safe_url("ftp://example.com/file")
        assert not is_safe_url("file:///etc/passwd")
