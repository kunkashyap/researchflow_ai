"""
agents/search_agent.py
Search Agent — LCEL pipeline that generates search queries and retrieves sources.

Architecture:
    query_generation_chain: prompt | llm | json_parser
        ↓ (list of queries)
    TavilySearchTool.search_multiple()
        ↓ (list of ResearchSource)
    SearchAgentOutput (Pydantic model)

The LLM is used only for QUERY GENERATION, not for fabricating results.
All actual search results come from the Tavily API.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from models.schemas import ResearchConfiguration, SearchAgentOutput
from prompts.search_prompts import search_query_generation_prompt
from tools.tavily_search import TavilySearchTool
from utils.config import config
from utils.logging_config import setup_logging

logger = setup_logging("researchflow.search_agent")


def _parse_query_list(llm_output: str, num_queries: int) -> list[str]:
    """
    Parse a JSON array of strings from LLM output.
    Falls back to regex extraction if JSON parsing fails.
    """
    # Try direct JSON parse
    try:
        # Strip markdown code fences if present
        cleaned = re.sub(r"```(?:json)?", "", llm_output).strip().strip("`")
        queries = json.loads(cleaned)
        if isinstance(queries, list) and all(isinstance(q, str) for q in queries):
            return queries[:num_queries]
    except json.JSONDecodeError:
        pass

    # Regex fallback: extract quoted strings
    logger.warning("JSON parse failed for query list; using regex fallback")
    matches = re.findall(r'"([^"]{5,200})"', llm_output)
    if matches:
        return matches[:num_queries]

    # Last resort: return the topic as a single query
    logger.error("Could not parse query list from LLM output: %s", llm_output[:200])
    return [llm_output.strip()[:200]]


class SearchAgent:
    """
    Search Agent — generates targeted queries and retrieves web sources.

    LCEL chain:
        search_query_generation_prompt | llm | StrOutputParser()
        → _parse_query_list()
        → TavilySearchTool.search_multiple()
        → SearchAgentOutput
    """

    def __init__(self, model_name: Optional[str] = None):
        self.llm = ChatOpenAI(
            model=model_name or config.OPENAI_MODEL,
            temperature=0.3,  # Low temperature for precise query generation
            openai_api_key=config.OPENAI_API_KEY,
        )
        self.search_tool = TavilySearchTool()
        # LCEL chain for query generation
        self._query_chain = search_query_generation_prompt | self.llm | StrOutputParser()

    def run(self, research_config: ResearchConfiguration) -> SearchAgentOutput:
        """
        Execute the full search pipeline.

        Args:
            research_config: User-specified research configuration.

        Returns:
            SearchAgentOutput containing the discovered sources.
        """
        topic = research_config.research_topic
        num_queries = research_config.num_queries
        max_sources = research_config.max_sources

        logger.info(
            "SearchAgent starting: topic='%s', queries=%d, max_sources=%d",
            topic, num_queries, max_sources
        )

        # --- Step 1: Generate targeted search queries via LLM ---
        try:
            raw_llm_output = self._query_chain.invoke({
                "research_topic": topic,
                "num_queries": num_queries,
            })
            queries = _parse_query_list(raw_llm_output, num_queries)
            logger.info("Generated %d search queries: %s", len(queries), queries)
        except Exception as exc:
            logger.error("Query generation failed: %s", exc)
            # Fall back to searching with the raw topic
            queries = [topic]

        # --- Step 2: Execute Tavily searches ---
        try:
            sources = self.search_tool.search_multiple(
                queries=queries,
                search_depth=research_config.tavily_search_depth,
                max_results_per_query=max(3, max_sources // len(queries) + 1),
                total_max_results=max_sources,
            )
        except Exception as exc:
            logger.error("Tavily search failed: %s", exc)
            return SearchAgentOutput(
                research_topic=topic,
                search_queries=queries,
                sources=[],
                status="failed",
                error=str(exc),
            )

        logger.info("SearchAgent complete: %d sources found", len(sources))
        return SearchAgentOutput(
            research_topic=topic,
            search_queries=queries,
            sources=sources,
            status="completed",
        )
