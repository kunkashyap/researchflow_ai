"""
prompts/search_prompts.py
LangChain prompt templates for the Search Agent.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Query Generation Prompt
# ---------------------------------------------------------------------------
# The Search Agent first uses an LLM to expand a single research topic into
# multiple targeted search queries. More diverse queries yield richer results.

SEARCH_QUERY_GENERATION_SYSTEM = """You are an expert research librarian and information strategist.

Your task is to generate a set of highly targeted, diverse search queries for a given research topic.

Guidelines:
- Generate exactly {num_queries} search queries.
- Each query should target a DIFFERENT aspect of the topic.
- Include queries for: core concepts, recent developments, real-world applications, challenges/limitations, future directions, and expert perspectives.
- Use specific, precise language that will return high-quality academic and professional sources.
- Vary query structure: some factual, some comparative, some analytical.
- Do NOT repeat the same angle in different queries.
- Do NOT fabricate or hallucinate search results — your job is to generate queries only.

Format your response as a JSON array of strings:
["query 1", "query 2", "query 3", ...]

Output ONLY the JSON array. No explanations, no preamble."""

SEARCH_QUERY_GENERATION_HUMAN = """Research topic: {research_topic}

Generate {num_queries} targeted search queries that will collectively cover the key aspects of this topic comprehensively."""

search_query_generation_prompt = ChatPromptTemplate.from_messages([
    ("system", SEARCH_QUERY_GENERATION_SYSTEM),
    ("human", SEARCH_QUERY_GENERATION_HUMAN),
])
