"""
app.py
ResearchFlow AI — Streamlit Dashboard
=======================================
Entry point for the web interface.

Run with:
    streamlit run app.py

The dashboard provides:
- Research configuration panel
- Live agent workflow progress display
- Tabbed results (Report, Sources, Extracted Info, Critic Evaluation, Logs)
- Research history browser
- Markdown report download
"""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# ---- Page config (must be first Streamlit call) ----
st.set_page_config(
    page_title="ResearchFlow AI",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/researchflow-ai",
        "Report a bug": "https://github.com/researchflow-ai/issues",
        "About": "ResearchFlow AI — Autonomous Multi-Agent Research Assistant",
    },
)

from memory.database import get_all_sessions, get_session_by_id
from models.schemas import ResearchConfiguration
from orchestrator.research_orchestrator import ResearchOrchestrator
from utils.config import config

# ---------------------------------------------------------------------------
# Custom CSS — Premium dark-theme design
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ---- Root theme variables ---- */
:root {
    --bg-primary: #0d1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2128;
    --bg-hover: #21262d;
    --border: #30363d;
    --accent-blue: #58a6ff;
    --accent-purple: #bc8cff;
    --accent-green: #3fb950;
    --accent-orange: #ff7b72;
    --accent-yellow: #e3b341;
    --text-primary: #e6edf3;
    --text-secondary: #8b949e;
    --text-muted: #484f58;
    --gradient-hero: linear-gradient(135deg, #1a1f35 0%, #0d1117 50%, #1a1535 100%);
}

/* ---- Global resets ---- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

.stApp {
    background: var(--bg-primary);
}

/* ---- Hide default Streamlit chrome ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--accent-blue) !important;
}

/* ---- Hero header ---- */
.hero-container {
    background: var(--gradient-hero);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(88,166,255,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(188,140,255,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #58a6ff 0%, #bc8cff 50%, #79c0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: var(--text-secondary);
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(63,185,80,0.15);
    border: 1px solid rgba(63,185,80,0.4);
    color: var(--accent-green);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

/* ---- Cards ---- */
.card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s, transform 0.2s;
}

.card:hover {
    border-color: rgba(88,166,255,0.4);
    transform: translateY(-1px);
}

/* ---- Agent status cards ---- */
.agent-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 12px;
    transition: all 0.3s ease;
}

.agent-card.running {
    border-color: var(--accent-blue);
    background: rgba(88,166,255,0.06);
    animation: pulse-border 2s infinite;
}

.agent-card.completed {
    border-color: var(--accent-green);
    background: rgba(63,185,80,0.06);
}

.agent-card.failed {
    border-color: var(--accent-orange);
    background: rgba(255,123,114,0.06);
}

.agent-card.pending {
    opacity: 0.5;
}

@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(88,166,255,0.2); }
    50% { box-shadow: 0 0 0 4px rgba(88,166,255,0.1); }
}

.agent-icon {
    font-size: 1.4rem;
    min-width: 32px;
    text-align: center;
}

.agent-name {
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-primary);
}

.agent-status-text {
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin-top: 2px;
}

.status-badge {
    margin-left: auto;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.badge-running { background: rgba(88,166,255,0.2); color: var(--accent-blue); }
.badge-completed { background: rgba(63,185,80,0.2); color: var(--accent-green); }
.badge-failed { background: rgba(255,123,114,0.2); color: var(--accent-orange); }
.badge-pending { background: rgba(72,79,88,0.3); color: var(--text-muted); }
.badge-skipped { background: rgba(227,179,65,0.2); color: var(--accent-yellow); }

/* ---- Score ring ---- */
.score-container {
    text-align: center;
    padding: 1.5rem;
}

.score-ring {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 110px;
    height: 110px;
    border-radius: 50%;
    border: 4px solid;
    margin-bottom: 0.5rem;
}

.score-ring.high { border-color: var(--accent-green); }
.score-ring.medium { border-color: var(--accent-yellow); }
.score-ring.low { border-color: var(--accent-orange); }

.score-number {
    font-size: 2rem;
    font-weight: 800;
    line-height: 1;
}

.score-label {
    font-size: 0.65rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---- Metric cards ---- */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
}

.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    flex: 1;
    min-width: 120px;
    text-align: center;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--accent-blue);
    line-height: 1;
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---- Source chips ---- */
.source-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.3);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: var(--accent-blue);
    margin: 2px;
    text-decoration: none;
}

/* ---- Report display ---- */
.report-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem;
    line-height: 1.8;
}

.report-container h1 { color: var(--accent-blue); font-size: 1.8rem; }
.report-container h2 { color: var(--text-primary); border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }
.report-container h3 { color: var(--accent-purple); }
.report-container code { background: var(--bg-secondary); padding: 2px 6px; border-radius: 4px; font-family: 'JetBrains Mono', monospace; font-size: 0.85em; }

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: 10px;
    padding: 4px;
    border: 1px solid var(--border);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: var(--text-secondary);
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 16px;
    transition: all 0.2s;
}

.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #1a4fa0 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.01em !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.4) !important;
}

.stButton > button:active { transform: translateY(0) !important; }

/* ---- Inputs ---- */
.stTextInput input, .stTextArea textarea, .stSelectbox select {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
}

.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 2px rgba(88,166,255,0.15) !important;
}

/* ---- Progress bar ---- */
.stProgress > div > div > div { background: linear-gradient(90deg, #58a6ff, #bc8cff) !important; }

/* ---- History table ---- */
.history-row {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    cursor: pointer;
    transition: border-color 0.2s;
}

.history-row:hover { border-color: rgba(88,166,255,0.4); }

/* ---- Section dividers ---- */
.section-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-muted);
    margin: 1.5rem 0 0.75rem 0;
}

/* ---- Spinner animation ---- */
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.spinning { display: inline-block; animation: spin 1s linear infinite; }

/* ---- Approval banner ---- */
.approval-banner {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.75rem 1.25rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    font-weight: 600;
    font-size: 0.9rem;
}

.approval-banner.approved {
    background: rgba(63,185,80,0.12);
    border: 1px solid rgba(63,185,80,0.4);
    color: var(--accent-green);
}

.approval-banner.not-approved {
    background: rgba(227,179,65,0.12);
    border: 1px solid rgba(227,179,65,0.4);
    color: var(--accent-yellow);
}
</style>
"""

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def init_session_state():
    """Initialise Streamlit session state variables on first run."""
    defaults = {
        "research_result": None,
        "is_running": False,
        "agent_statuses": {
            "search_agent": "pending",
            "reader_agent": "pending",
            "writer_agent": "pending",
            "critic_agent": "pending",
        },
        "agent_messages": {
            "search_agent": "",
            "reader_agent": "",
            "writer_agent": "",
            "critic_agent": "",
        },
        "current_view": "research",   # research | history
        "history_session_id": None,
        "run_start_time": None,
        "execution_logs": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Agent display helpers
# ---------------------------------------------------------------------------

AGENT_META = {
    "search_agent": {"icon": "🔍", "name": "Search Agent", "desc": "Generating queries & searching the web"},
    "reader_agent": {"icon": "📖", "name": "Reader Agent", "desc": "Extracting content from sources"},
    "writer_agent": {"icon": "✍️", "name": "Writer Agent", "desc": "Composing research report"},
    "critic_agent": {"icon": "🎯", "name": "Critic Agent", "desc": "Evaluating report quality"},
}


def render_agent_card(agent_key: str, status: str, message: str = ""):
    """Render an individual agent status card."""
    meta = AGENT_META[agent_key]
    badge_class = f"badge-{status}"
    card_class = f"agent-card {status}"

    status_icon = {
        "pending": "○",
        "running": "◉",
        "completed": "✓",
        "failed": "✗",
        "skipped": "—",
    }.get(status, "○")

    display_message = message or meta["desc"]

    st.markdown(f"""
    <div class="{card_class}">
        <span class="agent-icon">{meta['icon']}</span>
        <div>
            <div class="agent-name">{meta['name']}</div>
            <div class="agent-status-text">{display_message}</div>
        </div>
        <span class="status-badge {badge_class}">{status_icon} {status.upper()}</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Score display
# ---------------------------------------------------------------------------

def render_score_ring(score: int, threshold: int):
    """Render the circular score indicator."""
    if score >= threshold:
        ring_class = "high"
        color = "#3fb950"
    elif score >= threshold * 0.7:
        ring_class = "medium"
        color = "#e3b341"
    else:
        ring_class = "low"
        color = "#ff7b72"

    st.markdown(f"""
    <div class="score-container">
        <div class="score-ring {ring_class}">
            <span class="score-number" style="color: {color};">{score}</span>
            <span class="score-label">/ 100</span>
        </div>
        <div style="color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;">Quality Score</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Research runner (runs in background thread)
# ---------------------------------------------------------------------------

def run_research_in_thread(research_config: ResearchConfiguration):
    """
    Execute the research pipeline in a background thread.
    Progress updates are written to st.session_state which Streamlit
    can read on rerun cycles.
    """
    def progress_callback(agent_name: str, status: str, message: str):
        st.session_state.agent_statuses[agent_name] = status
        st.session_state.agent_messages[agent_name] = message

    orchestrator = ResearchOrchestrator()

    try:
        result = orchestrator.run(
            research_config=research_config,
            progress_callback=progress_callback,
        )
        st.session_state.research_result = result
        # Copy execution logs
        if result.execution_logs:
            st.session_state.execution_logs = result.execution_logs
    except Exception as exc:
        st.session_state.research_result = None
        st.session_state.agent_statuses = {k: "failed" for k in AGENT_META}
        st.session_state.agent_messages["search_agent"] = str(exc)
    finally:
        st.session_state.is_running = False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar():
    """Render the sidebar navigation and status."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #58a6ff;">ResearchFlow AI</div>
            <div style="font-size: 0.75rem; color: #8b949e;">Multi-Agent Research</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navigation
        st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)

        if st.button("🔬 New Research", use_container_width=True):
            st.session_state.current_view = "research"
            st.session_state.history_session_id = None

        if st.button("📚 Research History", use_container_width=True):
            st.session_state.current_view = "history"

        st.divider()

        # Agent status (while running)
        if st.session_state.is_running or st.session_state.research_result:
            st.markdown('<div class="section-title">Agent Pipeline</div>', unsafe_allow_html=True)
            for agent_key in AGENT_META:
                status = st.session_state.agent_statuses.get(agent_key, "pending")
                meta = AGENT_META[agent_key]
                icon = {"pending": "○", "running": "◉", "completed": "✓", "failed": "✗"}.get(status, "○")
                color = {
                    "pending": "#484f58",
                    "running": "#58a6ff",
                    "completed": "#3fb950",
                    "failed": "#ff7b72",
                }.get(status, "#484f58")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;'
                    f'font-size:0.85rem;color:{color};">'
                    f'{meta["icon"]} {meta["name"]}'
                    f'<span style="margin-left:auto;font-size:0.7rem;">{icon}</span></div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        # API key status
        st.markdown('<div class="section-title">System Status</div>', unsafe_allow_html=True)
        errors = config.validate_api_keys()
        if errors:
            st.markdown(
                '<div style="color:#ff7b72;font-size:0.8rem;">⚠️ API Keys Missing</div>',
                unsafe_allow_html=True
            )
            for err in errors:
                st.caption(err)
        else:
            st.markdown(
                '<div style="color:#3fb950;font-size:0.8rem;">✓ API Keys Configured</div>',
                unsafe_allow_html=True
            )

        st.caption(f"Model: `{config.OPENAI_MODEL}`")


# ---------------------------------------------------------------------------
# Main Research Panel
# ---------------------------------------------------------------------------

def render_research_panel():
    """Render the main research configuration and results panel."""

    # Hero header
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">⚡ Powered by GPT + Tavily</div>
        <h1 class="hero-title">ResearchFlow AI</h1>
        <p class="hero-subtitle">
            Autonomous multi-agent research assistant. Enter any topic and four specialised AI agents
            will search the web, extract content, write a comprehensive report, and critically evaluate it.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # API key validation warning
    errors = config.validate_api_keys()
    if errors:
        st.error("⚠️ **API Keys Required** — Please configure your `.env` file before starting research.")
        for err in errors:
            st.caption(f"• {err}")
        st.code("cp .env.example .env\n# Then edit .env with your API keys")
        return

    # Research configuration form
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown('<div class="section-title">Research Topic</div>', unsafe_allow_html=True)
            research_topic = st.text_area(
                "Enter your research question or topic",
                placeholder="e.g. 'The impact of large language models on software engineering productivity'",
                height=80,
                key="topic_input",
                label_visibility="collapsed",
            )

        with col2:
            st.markdown('<div class="section-title">Research Depth</div>', unsafe_allow_html=True)
            search_depth = st.selectbox(
                "Depth",
                options=["quick", "standard", "deep"],
                index=1,
                label_visibility="collapsed",
                key="depth_select",
                help="Quick: 2 queries, 5 sources | Standard: 4 queries, 8 sources | Deep: 6 queries, 15 sources",
            )

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        max_sources = st.slider("Max Sources", min_value=3, max_value=20, value=8, key="max_sources")
    with col_b:
        max_revisions = st.slider("Max Revisions", min_value=0, max_value=5, value=2, key="max_revisions")
    with col_c:
        approval_threshold = st.slider("Approval Threshold", min_value=50, max_value=95, value=70, key="threshold",
                                        help="Minimum Critic score (0-100) required to approve the report")
    with col_d:
        date_range = st.text_input(
            "Date Range (optional)",
            placeholder="e.g. 2023 to 2024",
            key="date_range",
            label_visibility="visible",
        )

    # Start button
    st.markdown("<br>", unsafe_allow_html=True)
    btn_col, _, _ = st.columns([1, 2, 2])
    with btn_col:
        start_clicked = st.button(
            "🚀 Start Research",
            disabled=st.session_state.is_running,
            use_container_width=True,
            key="start_btn",
        )

    if start_clicked and research_topic.strip():
        # Reset state
        st.session_state.research_result = None
        st.session_state.execution_logs = []
        st.session_state.agent_statuses = {k: "pending" for k in AGENT_META}
        st.session_state.agent_messages = {k: "" for k in AGENT_META}
        st.session_state.is_running = True
        st.session_state.run_start_time = time.time()

        research_config = ResearchConfiguration(
            research_topic=research_topic.strip(),
            search_depth=search_depth,
            max_sources=max_sources,
            max_revisions=max_revisions,
            approval_threshold=approval_threshold,
            date_range=date_range.strip() if date_range.strip() else None,
        )

        thread = threading.Thread(
            target=run_research_in_thread,
            args=(research_config,),
            daemon=True,
        )
        thread.start()
        st.rerun()

    elif start_clicked and not research_topic.strip():
        st.warning("Please enter a research topic before starting.")

    # --- Live Progress Display ---
    if st.session_state.is_running or st.session_state.research_result:
        st.markdown("---")
        st.markdown('<div class="section-title">Agent Workflow</div>', unsafe_allow_html=True)

        prog_col, time_col = st.columns([3, 1])

        with prog_col:
            statuses = st.session_state.agent_statuses
            messages = st.session_state.agent_messages

            # Progress bar
            completed_count = sum(1 for s in statuses.values() if s == "completed")
            total_agents = len(AGENT_META)
            progress_pct = completed_count / total_agents

            st.progress(progress_pct)

            # Agent cards
            for agent_key in AGENT_META:
                render_agent_card(
                    agent_key,
                    statuses.get(agent_key, "pending"),
                    messages.get(agent_key, ""),
                )

        with time_col:
            if st.session_state.run_start_time:
                elapsed = time.time() - st.session_state.run_start_time
                minutes = int(elapsed // 60)
                seconds = int(elapsed % 60)
                st.metric("Elapsed", f"{minutes:02d}:{seconds:02d}")

        # Auto-refresh while running
        if st.session_state.is_running:
            time.sleep(2)
            st.rerun()

    # --- Results Panel ---
    if st.session_state.research_result:
        render_results_panel(st.session_state.research_result)


# ---------------------------------------------------------------------------
# Results Panel
# ---------------------------------------------------------------------------

def render_results_panel(result):
    """Render the tabbed research results view."""
    st.markdown("---")
    st.markdown('<div class="section-title">Research Results</div>', unsafe_allow_html=True)

    # Quick metrics row
    if result.search_output and result.reader_output:
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Sources Found", len(result.search_output.sources) if result.search_output else 0)
        with m2:
            st.metric("Successfully Read",
                      result.reader_output.successful_extractions if result.reader_output else 0)
        with m3:
            st.metric("Report Words",
                      result.writer_output.word_count if result.writer_output else 0)
        with m4:
            st.metric("Quality Score", f"{result.final_score}/100")
        with m5:
            st.metric("Revisions", result.revision_count)

    # Status banner
    if result.status == "completed":
        approved = result.critic_output and result.critic_output.approved
        banner_class = "approved" if approved else "not-approved"
        banner_icon = "✅" if approved else "⚠️"
        banner_text = "Report Approved" if approved else f"Below Threshold (threshold: {result.configuration.approval_threshold})"
        st.markdown(f"""
        <div class="approval-banner {banner_class}">
            {banner_icon} {banner_text} — Score: {result.final_score}/100 | Revisions: {result.revision_count}
        </div>
        """, unsafe_allow_html=True)
    elif result.status == "failed":
        st.error(f"❌ Research failed: {result.error_message}")

    # Tabs
    tab_report, tab_sources, tab_extracted, tab_critic, tab_logs = st.tabs([
        "📄 Final Report",
        "🌐 Sources",
        "📋 Extracted Info",
        "🎯 Critic Evaluation",
        "📊 Execution Logs",
    ])

    # ---- Tab 1: Final Report ----
    with tab_report:
        if result.final_report:
            # Download button
            dl_col1, dl_col2, _ = st.columns([1, 1, 3])
            with dl_col1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=result.final_report,
                    file_name=f"research_{result.research_id}.md",
                    mime="text/markdown",
                )
            with dl_col2:
                # Save to exports directory
                _save_report_to_disk(result)
                st.caption(f"ID: `{result.research_id}`")

            st.markdown('<div class="report-container">', unsafe_allow_html=True)
            st.markdown(result.final_report)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No report generated yet.")

    # ---- Tab 2: Sources ----
    with tab_sources:
        if result.search_output and result.search_output.sources:
            st.markdown(f"**{len(result.search_output.sources)} sources discovered using {len(result.search_output.search_queries)} search queries**")
            st.caption("Search queries: " + " | ".join(f'"{q}"' for q in result.search_output.search_queries[:6]))
            st.markdown("---")
            for source in result.search_output.sources:
                with st.expander(f"[{source.source_id}] {source.title or source.url}", expanded=False):
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.markdown(f"🔗 [{source.url}]({source.url})")
                        if source.published_date:
                            st.caption(f"Published: {source.published_date}")
                        st.markdown(source.snippet)
                    with col_b:
                        st.metric("Relevance", f"{source.relevance_score:.2f}")
        else:
            st.info("No sources available.")

    # ---- Tab 3: Extracted Information ----
    with tab_extracted:
        if result.reader_output and result.reader_output.extracted_documents:
            docs = result.reader_output.extracted_documents
            success = [d for d in docs if d.extraction_status in ("success", "fallback")]
            failed = [d for d in docs if d.extraction_status == "failed"]

            st.markdown(f"**{len(success)} successfully extracted, {len(failed)} failed**")

            for doc in docs:
                status_icon = {"success": "✅", "fallback": "🔁", "failed": "❌", "partial": "⚠️"}.get(
                    doc.extraction_status, "?"
                )
                label = f"{status_icon} [{doc.source_id}] {doc.title or doc.url}"
                with st.expander(label, expanded=False):
                    st.caption(f"Status: {doc.extraction_status} | Characters: {doc.content_length:,}")
                    if doc.failure_reason:
                        st.warning(f"Note: {doc.failure_reason}")
                    if doc.headings:
                        st.markdown("**Headings:** " + " › ".join(doc.headings[:8]))
                    if doc.extracted_text:
                        st.text_area(
                            "Extracted Content",
                            value=doc.extracted_text[:3000] + ("..." if len(doc.extracted_text) > 3000 else ""),
                            height=200,
                            key=f"doc_{doc.source_id}",
                            disabled=True,
                        )
        else:
            st.info("No extracted documents available.")

    # ---- Tab 4: Critic Evaluation ----
    with tab_critic:
        if result.critic_output:
            critic = result.critic_output

            crit_col1, crit_col2 = st.columns([1, 2])
            with crit_col1:
                render_score_ring(critic.overall_score, result.configuration.approval_threshold)

                # Category score breakdown
                cats = critic.category_scores
                score_data = {
                    "Factual Accuracy": (cats.factual_accuracy, 25),
                    "Research Depth": (cats.research_depth, 20),
                    "Source Quality": (cats.source_quality, 20),
                    "Citation Quality": (cats.citation_quality, 15),
                    "Organisation": (cats.organization, 10),
                    "Readability": (cats.readability, 10),
                }
                for label, (score, max_score) in score_data.items():
                    pct = score / max_score
                    color = "#3fb950" if pct >= 0.7 else "#e3b341" if pct >= 0.5 else "#ff7b72"
                    st.markdown(
                        f'<div style="margin:4px 0;font-size:0.82rem;">'
                        f'<span style="color:var(--text-secondary);">{label}</span>'
                        f'<span style="float:right;color:{color};font-weight:600;">{score}/{max_score}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(pct)

            with crit_col2:
                if critic.strengths:
                    st.markdown("#### ✅ Strengths")
                    for s in critic.strengths:
                        st.markdown(f"• {s}")

                if critic.weaknesses:
                    st.markdown("#### ⚠️ Weaknesses")
                    for w in critic.weaknesses:
                        st.markdown(f"• {w}")

                if critic.unsupported_claims:
                    st.markdown("#### 🔍 Unsupported Claims")
                    for c in critic.unsupported_claims:
                        st.markdown(f"• _{c}_")

                if critic.missing_information:
                    st.markdown("#### 📌 Missing Information")
                    for m in critic.missing_information:
                        st.markdown(f"• {m}")

                if critic.revision_recommendations:
                    st.markdown("#### 💡 Recommendations")
                    for r in critic.revision_recommendations:
                        st.markdown(f"• {r}")
        else:
            st.info("No critic evaluation available.")

    # ---- Tab 5: Execution Logs ----
    with tab_logs:
        logs = st.session_state.execution_logs or (result.execution_logs if result else [])
        if logs:
            log_text = "\n".join(logs)
            st.text_area(
                "Agent Execution Logs",
                value=log_text,
                height=400,
                key="log_area",
                disabled=True,
            )
            st.download_button(
                "⬇️ Download Logs",
                data=log_text,
                file_name=f"logs_{result.research_id}.txt",
                mime="text/plain",
            )
        else:
            st.info("No execution logs available.")


def _save_report_to_disk(result) -> Optional[Path]:
    """Save the final report to the exports directory."""
    if not result.final_report:
        return None
    try:
        export_path = config.EXPORTS_DIR / f"{result.research_id}.md"
        export_path.write_text(result.final_report, encoding="utf-8")
        return export_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# History Panel
# ---------------------------------------------------------------------------

def render_history_panel():
    """Render the research history browser."""
    st.markdown("""
    <div class="hero-container">
        <h1 class="hero-title" style="font-size: 1.8rem;">📚 Research History</h1>
        <p class="hero-subtitle">Review and revisit your previous research sessions.</p>
    </div>
    """, unsafe_allow_html=True)

    sessions = get_all_sessions(limit=50)

    if not sessions:
        st.info("No research sessions found. Start your first research above!")
        return

    st.markdown(f"**{len(sessions)} sessions** found")

    for session in sessions:
        status_icon = {"completed": "✅", "failed": "❌", "running": "⏳", "pending": "⏳"}.get(
            session.get("status", "pending"), "?"
        )
        score = session.get("final_score", 0)
        score_color = "#3fb950" if score >= 70 else "#e3b341" if score >= 50 else "#ff7b72"

        col1, col2, col3 = st.columns([4, 1, 1])
        with col1:
            st.markdown(
                f'<div style="font-size:0.9rem;font-weight:600;">{status_icon} {session["research_topic"]}</div>'
                f'<div style="font-size:0.75rem;color:#8b949e;">'
                f'ID: {session["research_id"]} | Started: {session.get("started_at", "")[:19]}</div>',
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f'<div style="text-align:center;font-size:1.3rem;font-weight:700;color:{score_color};">{score}</div>'
                f'<div style="text-align:center;font-size:0.65rem;color:#8b949e;">SCORE</div>',
                unsafe_allow_html=True,
            )
        with col3:
            if st.button("Open", key=f"open_{session['research_id']}"):
                st.session_state.history_session_id = session["research_id"]

        st.markdown('<hr style="border-color:#21262d;margin:0.5rem 0;">', unsafe_allow_html=True)

    # Detail view for selected session
    if st.session_state.history_session_id:
        st.markdown("---")
        full_session = get_session_by_id(st.session_state.history_session_id)
        if full_session:
            st.markdown(f"### 📄 {full_session['research_topic']}")
            st.caption(f"ID: `{full_session['research_id']}` | Status: {full_session['status']}")

            if full_session.get("final_report"):
                st.download_button(
                    "⬇️ Download Report",
                    data=full_session["final_report"],
                    file_name=f"research_{full_session['research_id']}.md",
                    mime="text/markdown",
                )
                st.markdown(full_session["final_report"])
        else:
            st.error("Session not found in database.")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_session_state()
    render_sidebar()

    if st.session_state.current_view == "history":
        render_history_panel()
    else:
        render_research_panel()


if __name__ == "__main__":
    main()
