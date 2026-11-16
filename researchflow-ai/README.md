# ResearchFlow AI
## Autonomous Multi-Agent Research Assistant

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://python.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)

> Enter any research topic. Four specialised AI agents collaborate to search the web, extract content, write a comprehensive report, and critically evaluate it — returning a professionally structured, citation-rich research document.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [How the Agents Communicate](#how-the-agents-communicate)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Environment Configuration](#environment-configuration)
6. [Running the Application](#running-the-application)
7. [Running Tests](#running-tests)
8. [Project Structure](#project-structure)
9. [Agent Details](#agent-details)
10. [Known Limitations](#known-limitations)

---

## System Architecture

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR                            │
│  Coordinates all agents, manages state, handles revisions   │
└───────────────┬────────────────────────────────────────────┘
                │
    ┌───────────▼───────────┐
    │    SEARCH AGENT       │  LLM query expansion → Tavily Search API
    │  (search_agent.py)    │  → structured ResearchSource list
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │    READER AGENT       │  requests + BeautifulSoup → HTML extraction
    │  (reader_agent.py)    │  → ExtractedDocument list
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │    WRITER AGENT       │  OpenAI LLM → structured Markdown report
    │  (writer_agent.py)    │  with [S1], [S2] citations
    └───────────┬───────────┘
                │
    ┌───────────▼───────────┐
    │    CRITIC AGENT       │  OpenAI LLM → 100-point scored evaluation
    │  (critic_agent.py)    │  → revision recommendations
    └───────────┬───────────┘
                │
      [Not Approved & revisions_remaining > 0]
                │
    ┌───────────▼───────────┐
    │    WRITER AGENT       │  Revises report based on Critic feedback
    │    (revision mode)    │
    └───────────┬───────────┘
                │
       [Loop until approved OR max_revisions reached]
                │
    ┌───────────▼───────────┐
    │    FINAL REPORT       │  → SQLite persistence
    │    + CRITIC SCORE     │  → Streamlit dashboard
    └───────────────────────┘
```

---

## How the Agents Communicate

All agents communicate through **typed Pydantic models** passed by the **Orchestrator**. No agent directly calls another agent.

| From | To | Data Model |
|---|---|---|
| User Input | Orchestrator | `ResearchConfiguration` |
| Orchestrator → Search Agent | Search Agent | `ResearchConfiguration` |
| Search Agent → Orchestrator | Orchestrator | `SearchAgentOutput` |
| Orchestrator → Reader Agent | Reader Agent | `SearchAgentOutput` |
| Reader Agent → Orchestrator | Orchestrator | `ReaderAgentOutput` |
| Orchestrator → Writer Agent | Writer Agent | `SearchAgentOutput + ReaderAgentOutput` |
| Writer Agent → Orchestrator | Orchestrator | `WriterAgentOutput` |
| Orchestrator → Critic Agent | Critic Agent | `WriterAgentOutput + sources + documents` |
| Critic Agent → Orchestrator | Orchestrator | `CriticAgentOutput` |
| Orchestrator → Writer Agent (revision) | Writer Agent | `CriticAgentOutput + current report` |

The Orchestrator stores everything in a `ResearchState` instance and persists it to SQLite at completion.

---

## Prerequisites

- Python 3.11 or higher
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Tavily API key ([get one here](https://app.tavily.com/sign-in))

---

## Installation

### 1. Clone or download the project

```bash
# If using git:
git clone <repo-url>
cd researchflow-ai

# Or navigate to the project directory directly
cd researchflow-ai
```

### 2. Create a Python virtual environment

```bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: This installs ~15 packages. The core dependencies are `langchain`, `langchain-openai`, `tavily-python`, `requests`, `beautifulsoup4`, `streamlit`, `pydantic`, and `python-dotenv`.

---

## Environment Configuration

### 1. Copy the example environment file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### 2. Edit `.env` with your API keys

```env
OPENAI_API_KEY=sk-...your-openai-key...
TAVILY_API_KEY=tvly-...your-tavily-key...

# Optional overrides
OPENAI_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

> ⚠️ **Never commit your `.env` file.** It is included in `.gitignore`.

### 3. Verify your keys are loaded correctly

```bash
python -c "from utils.config import config; errors = config.validate_api_keys(); print('OK' if not errors else errors)"
```

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open at **http://localhost:8501**.

### Using the Dashboard

1. Enter a research topic in the text area
2. Configure research depth, sources, revision attempts and approval threshold
3. Click **🚀 Start Research**
4. Watch the four agents work in real-time
5. Review the results in the tabbed output panel
6. Download the Markdown report
7. Browse previous sessions in the **Research History** tab

---

## Running Tests

```bash
# Run all tests (no API keys required — all external calls are mocked)
pytest tests/ -v

# Run a specific test file
pytest tests/test_search_agent.py -v
pytest tests/test_reader_agent.py -v
pytest tests/test_writer_agent.py -v
pytest tests/test_critic_agent.py -v
pytest tests/test_orchestrator.py -v

# Run with coverage report
pip install pytest-cov
pytest tests/ -v --cov=. --cov-report=term-missing
```

> All tests use mocked APIs and do NOT require real API keys or internet access.

---

## Project Structure

```
researchflow-ai/
├── app.py                         # Streamlit dashboard (entry point)
├── requirements.txt               # Python dependencies
├── .env.example                   # Environment variable template
├── .gitignore                     # Excludes .env, DB, logs, virtualenv
├── README.md                      # This file
│
├── agents/                        # Four specialised AI agents
│   ├── search_agent.py            # LLM query generation + Tavily search
│   ├── reader_agent.py            # BeautifulSoup content extraction
│   ├── writer_agent.py            # Report generation + revision (LCEL)
│   └── critic_agent.py            # Quality scoring + feedback (LCEL)
│
├── orchestrator/
│   └── research_orchestrator.py   # Central pipeline coordinator
│
├── memory/
│   ├── state.py                   # ResearchState: shared mutable container
│   └── database.py                # SQLite persistence layer
│
├── tools/
│   ├── tavily_search.py           # Tavily SDK wrapper (multi-query)
│   └── web_scraper.py             # BeautifulSoup scraper (concurrent)
│
├── prompts/
│   ├── search_prompts.py          # Search Agent LangChain prompts
│   ├── reader_prompts.py          # Reader Agent summarisation prompts
│   ├── writer_prompts.py          # Writer Agent initial + revision prompts
│   └── critic_prompts.py          # Critic Agent evaluation prompt
│
├── models/
│   └── schemas.py                 # All Pydantic v2 output schemas
│
├── utils/
│   ├── config.py                  # Settings + API key validation
│   ├── logging_config.py          # Structured logging setup
│   └── helpers.py                 # URL safety, dedup, text cleaning
│
├── database/                      # SQLite DB (created at runtime)
├── exports/                       # Saved report files
├── logs/                          # Daily log files
└── tests/                         # pytest test suite (5 files, ~50 tests)
```

---

## Agent Details

### Search Agent (`agents/search_agent.py`)
- **LLM task**: Generates 2–6 targeted search queries from the research topic using `gpt-4o-mini` with temperature 0.3
- **Tool**: Tavily Python SDK — executes all queries, deduplicates by URL, filters private IPs
- **Output**: `SearchAgentOutput` with ranked `ResearchSource` objects

### Reader Agent (`agents/reader_agent.py`)
- **Primary tool**: `requests` + `BeautifulSoup4` — concurrent scraping via `ThreadPoolExecutor`
- **Safety**: Checks `robots.txt`, rejects localhost/private IPs, enforces 500KB content limit and 15s timeout
- **Fallback**: Uses Tavily's pre-fetched content when direct scraping fails
- **LLM task** (optional): Condenses documents exceeding 8,000 characters to prevent Writer context overflow

### Writer Agent (`agents/writer_agent.py`)
- **Mode 1 — Initial**: `writer_initial_prompt | ChatOpenAI | StrOutputParser()` — generates full 12-section report
- **Mode 2 — Revision**: `writer_revision_prompt | ChatOpenAI | StrOutputParser()` — incorporates Critic feedback
- Source content is explicitly labelled as "UNTRUSTED WEB CONTENT" to mitigate prompt injection

### Critic Agent (`agents/critic_agent.py`)
- **LCEL chain**: `critic_prompt | ChatOpenAI | StrOutputParser()` — returns structured JSON
- **Scoring**: 6 weighted categories totalling 100 points; score is computed from category sum (not LLM self-report) to prevent inflation
- **Approval**: Sets `approved = True` only if computed score ≥ threshold

---

## Known Limitations

1. **OpenAI context window**: Very long research sessions (many sources + multiple revisions) may approach the model's token limit. `gpt-4o` has a larger context window than `gpt-4o-mini` and can be configured via `OPENAI_MODEL`.

2. **Streamlit threading**: The agent pipeline runs in a background thread. Progress updates require manual page refreshes (every 2 seconds automatically) due to Streamlit's execution model.

3. **JavaScript-heavy sites**: BeautifulSoup cannot execute JavaScript. Dynamic single-page apps (React, Vue) may yield empty or partial extraction — Tavily fallback content is used in these cases.

4. **Paywalls and CAPTCHAs**: The Reader Agent does not bypass access restrictions as a matter of policy. Paywalled content is gracefully handled via Tavily fallback snippets.

5. **PDF export**: The current implementation exports Markdown reports. PDF generation requires `reportlab` and is not fully implemented in this version.

6. **Rate limits**: Tavily and OpenAI both have rate limits. For deep research with many sources, delays between queries are built in. If you encounter 429 errors, reduce `max_sources` or use a plan with higher rate limits.
