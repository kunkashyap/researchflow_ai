"""
app.py
ResearchFlow AI — Premium Autonomous Multi-Agent Research Assistant
====================================================================
Streamlit dashboard with a modern, cyber-aesthetic, high-contrast dark theme,
real-time visual pipeline telemetry, interactive scorecards, and tabbed results.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import html
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import streamlit as st

# ---- Page Configuration (Must be the very first Streamlit call) ----
st.set_page_config(
    page_title="ResearchFlow AI | Autonomous Multi-Agent Research",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/researchflow-ai",
        "Report a bug": "https://github.com/researchflow-ai/issues",
        "About": "ResearchFlow AI — Autonomous Multi-Agent Research Assistant powered by LangChain LCEL & Tavily",
    },
)

from memory.database import delete_session, get_all_sessions, get_session_by_id
from models.schemas import ResearchConfiguration
from orchestrator.research_orchestrator import ResearchOrchestrator
from utils.config import config

# ---------------------------------------------------------------------------
# High-End Modern Custom CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ==========================================================================
   Root Design System
   ========================================================================== */
:root {
    --bg-base: #080c14;
    --bg-surface: #0e1526;
    --bg-card: rgba(17, 26, 46, 0.75);
    --bg-card-hover: rgba(25, 38, 66, 0.85);
    --bg-glass: rgba(14, 21, 38, 0.6);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-medium: rgba(255, 255, 255, 0.15);
    --border-glow: rgba(56, 189, 248, 0.35);

    --accent-cyan: #38bdf8;
    --accent-blue: #3b82f6;
    --accent-indigo: #6366f1;
    --accent-purple: #a855f7;
    --accent-pink: #ec4899;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;

    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;

    --glow-cyan: 0 0 25px rgba(56, 189, 248, 0.25);
    --glow-purple: 0 0 30px rgba(168, 85, 247, 0.25);
    --glow-emerald: 0 0 25px rgba(16, 185, 129, 0.25);
}

/* Global Typography & Background */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary);
}

.stApp {
    background: radial-gradient(circle at 15% 15%, rgba(30, 41, 69, 0.45) 0%, transparent 45%),
                radial-gradient(circle at 85% 85%, rgba(45, 25, 65, 0.35) 0%, transparent 50%),
                var(--bg-base) !important;
}

/* Chrome Cleanup */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1350px !important;
}

/* ==========================================================================
   Sidebar Styling
   ========================================================================== */
[data-testid="stSidebar"] {
    background-color: #0b1120 !important;
    border-right: 1px solid var(--border-subtle) !important;
    backdrop-filter: blur(12px);
}

[data-testid="stSidebar"] hr {
    border-color: var(--border-subtle) !important;
    margin: 1.25rem 0 !important;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0.5rem 0 1.25rem 0;
}

.brand-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 12px;
    background: linear-gradient(135deg, #0284c7, #7c3aed);
    box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
    font-size: 1.35rem;
}

.brand-title {
    font-weight: 800;
    font-size: 1.2rem;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.brand-subtitle {
    font-size: 0.72rem;
    color: var(--accent-cyan);
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ==========================================================================
   Hero Section & Floating Mesh
   ========================================================================== */
.hero-glass {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(20, 20, 45, 0.65) 100%);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 20px;
    padding: 2.25rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(16px);
}

.hero-glass::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 350px;
    height: 100%;
    background: radial-gradient(circle at 80% 20%, rgba(56, 189, 248, 0.15) 0%, transparent 70%);
    pointer-events: none;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 9999px;
    padding: 5px 14px;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--accent-cyan);
    margin-bottom: 1rem;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
}

.hero-headline {
    font-size: 2.6rem;
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    margin: 0 0 0.75rem 0;
    background: linear-gradient(135deg, #ffffff 10%, #93c5fd 50%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-desc {
    color: var(--text-secondary);
    font-size: 1.05rem;
    max-width: 820px;
    line-height: 1.6;
    margin: 0;
}

/* ==========================================================================
   Multi-Agent Interactive Visual Pipeline
   ========================================================================== */
.pipeline-container {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(12px);
}

.pipeline-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    position: relative;
}

@media (max-width: 900px) {
    .pipeline-grid {
        grid-template-columns: 1fr;
    }
}

.agent-node {
    position: relative;
    background: rgba(14, 21, 38, 0.85);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.25rem 1.1rem;
    display: flex;
    flex-direction: column;
    gap: 10px;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
}

.agent-node:hover {
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.4);
}

/* Agent Node State Overrides */
.agent-node.running {
    border-color: var(--accent-cyan);
    background: linear-gradient(145deg, rgba(14, 21, 38, 0.95), rgba(14, 45, 75, 0.4));
    box-shadow: var(--glow-cyan);
    animation: node-pulse 2.2s infinite;
}

.agent-node.completed {
    border-color: rgba(16, 185, 129, 0.5);
    background: linear-gradient(145deg, rgba(14, 21, 38, 0.95), rgba(12, 45, 35, 0.35));
}

.agent-node.failed {
    border-color: rgba(244, 63, 94, 0.5);
    background: linear-gradient(145deg, rgba(14, 21, 38, 0.95), rgba(55, 20, 30, 0.35));
}

.agent-node.pending {
    opacity: 0.55;
}

@keyframes node-pulse {
    0%, 100% { box-shadow: 0 0 15px rgba(56, 189, 248, 0.2); }
    50% { box-shadow: 0 0 28px rgba(56, 189, 248, 0.45); }
}

.node-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.node-icon-wrapper {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.08);
}

.agent-node.running .node-icon-wrapper {
    background: rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.4);
}

.agent-node.completed .node-icon-wrapper {
    background: rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.4);
}

.node-badge {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 3px 8px;
    border-radius: 6px;
}

.badge-pending { background: rgba(100, 116, 139, 0.2); color: #94a3b8; }
.badge-running { background: rgba(56, 189, 248, 0.2); color: #38bdf8; }
.badge-completed { background: rgba(16, 185, 129, 0.2); color: #34d399; }
.badge-failed { background: rgba(244, 63, 94, 0.2); color: #fb7185; }

.node-name {
    font-weight: 700;
    font-size: 0.95rem;
    color: var(--text-primary);
}

.node-desc {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.4;
    min-height: 34px;
}

/* ==========================================================================
   Stat & Metric Cards
   ========================================================================== */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 1.75rem;
}

.stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.25rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    transition: transform 0.2s, border-color 0.2s;
}

.stat-card:hover {
    transform: translateY(-2px);
    border-color: var(--border-medium);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-indigo));
}

.stat-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 6px;
}

.stat-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.stat-sub {
    font-size: 0.75rem;
    color: var(--accent-cyan);
    margin-top: 4px;
}

/* ==========================================================================
   Score Dial / Radial Gauge
   ========================================================================== */
.score-badge-card {
    background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(20, 20, 48, 0.8));
    border: 1px solid var(--border-medium);
    border-radius: 16px;
    padding: 1.75rem;
    text-align: center;
    position: relative;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
}

.radial-score {
    width: 120px;
    height: 120px;
    margin: 0 auto 1rem auto;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    background: radial-gradient(closest-side, #0b1120 78%, transparent 80% 100%),
                conic-gradient(var(--score-color, #10b981) calc(var(--score-pct, 75) * 1%), rgba(255,255,255,0.06) 0);
    box-shadow: 0 0 25px rgba(56, 189, 248, 0.2);
}

.radial-score-inner {
    text-align: center;
}

.radial-score-val {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--text-primary);
    line-height: 1;
}

.radial-score-denom {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    font-weight: 600;
}

/* ==========================================================================
   Report Container & Citations
   ========================================================================== */
.report-paper {
    background: rgba(11, 17, 32, 0.75);
    border: 1px solid var(--border-medium);
    border-radius: 18px;
    padding: 2.5rem 3rem;
    line-height: 1.85;
    font-size: 1.02rem;
    color: #e2e8f0;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
    backdrop-filter: blur(14px);
}

.report-paper h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #f8fafc;
    border-bottom: 2px solid rgba(56, 189, 248, 0.3);
    padding-bottom: 0.75rem;
    margin-top: 0;
    margin-bottom: 1.5rem;
    letter-spacing: -0.02em;
}

.report-paper h2 {
    font-size: 1.45rem;
    font-weight: 700;
    color: var(--accent-cyan);
    margin-top: 2rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

.report-paper h3 {
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--accent-purple);
    margin-top: 1.25rem;
}

.report-paper p {
    margin-bottom: 1.25rem;
}

.report-paper blockquote {
    border-left: 3px solid var(--accent-cyan);
    padding-left: 1rem;
    margin-left: 0;
    color: var(--text-secondary);
    background: rgba(56, 189, 248, 0.04);
    border-radius: 0 8px 8px 0;
    padding-top: 0.5rem;
    padding-bottom: 0.5rem;
}

.report-paper code {
    font-family: 'JetBrains Mono', monospace;
    background: rgba(30, 41, 59, 0.8);
    color: var(--accent-cyan);
    padding: 2px 7px;
    border-radius: 5px;
    font-size: 0.88em;
    border: 1px solid rgba(255, 255, 255, 0.05);
}

/* ==========================================================================
   Streamlit Component Overrides
   ========================================================================== */
/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(14, 21, 38, 0.8) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    padding: 6px !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(99, 102, 241, 0.2)) !important;
    color: #ffffff !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
}

/* Primary Button */
.stButton > button {
    background: linear-gradient(135deg, #0284c7 0%, #2563eb 50%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 0.98rem !important;
    padding: 0.75rem 2rem !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    box-shadow: 0 4px 20px -2px rgba(2, 132, 199, 0.5) !important;
    transition: all 0.25s ease !important;
    letter-spacing: 0.01em !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px -4px rgba(2, 132, 199, 0.7) !important;
    border-color: rgba(255, 255, 255, 0.3) !important;
}

.stButton > button:active {
    transform: translateY(0) !important;
}

/* Secondary / Download Buttons */
.stDownloadButton > button {
    background: rgba(30, 41, 59, 0.8) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid rgba(56, 189, 248, 0.3) !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
}

.stDownloadButton > button:hover {
    background: rgba(56, 189, 248, 0.15) !important;
    border-color: var(--accent-cyan) !important;
    color: #ffffff !important;
    box-shadow: 0 0 15px rgba(56, 189, 248, 0.2) !important;
}

/* Input Fields */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: rgba(14, 21, 38, 0.8) !important;
    border: 1px solid var(--border-medium) !important;
    border-radius: 10px !important;
    color: #f8fafc !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.15) !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    margin-top: 10px;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(14, 21, 38, 0.6) !important;
    border-radius: 10px !important;
    border: 1px solid var(--border-subtle) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%) !important;
    border-radius: 9999px !important;
}

/* Suggestion Pills */
.suggestion-pill {
    display: inline-block;
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid var(--border-subtle);
    border-radius: 9999px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    margin: 4px 4px 6px 0;
    transition: all 0.2s;
    cursor: default;
}

.suggestion-pill:hover {
    border-color: var(--accent-cyan);
    color: var(--accent-cyan);
    background: rgba(56, 189, 248, 0.08);
}
</style>
"""

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    """Initialize Streamlit session state on first run."""
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
        "current_view": "research",  # "research" | "history"
        "history_session_id": None,
        "run_start_time": None,
        "execution_logs": [],
        "prefill_topic": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Agent Pipeline Metadata
# ---------------------------------------------------------------------------
AGENT_META = {
    "search_agent": {
        "icon": "🔍",
        "name": "Search Agent",
        "desc": "Generating targeted queries & retrieving web sources",
        "tech": "OpenAI LCEL + Tavily Search",
    },
    "reader_agent": {
        "icon": "📖",
        "name": "Reader Agent",
        "desc": "Concurrently scraping & extracting clean HTML content",
        "tech": "Requests + BeautifulSoup4",
    },
    "writer_agent": {
        "icon": "✍️",
        "name": "Writer Agent",
        "desc": "Synthesizing structured report with [S1] citations",
        "tech": "LangChain LCEL Synthesis Chain",
    },
    "critic_agent": {
        "icon": "🎯",
        "name": "Critic Agent",
        "desc": "Scoring 6 categories & directing revision loops",
        "tech": "100-Point Academic Rubric",
    },
}


# ---------------------------------------------------------------------------
# Background Research Runner
# ---------------------------------------------------------------------------
def run_research_in_thread(research_config: ResearchConfiguration) -> None:
    """Execute the multi-agent pipeline in a daemon thread."""
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
        if result.execution_logs:
            st.session_state.execution_logs = result.execution_logs
    except Exception as exc:
        st.session_state.research_result = None
        st.session_state.agent_statuses = {k: "failed" for k in AGENT_META}
        st.session_state.agent_messages["search_agent"] = str(exc)
    finally:
        st.session_state.is_running = False


# ---------------------------------------------------------------------------
# Component: Modern Multi-Agent Pipeline Visualization
# ---------------------------------------------------------------------------
def render_pipeline_visualization() -> None:
    """Render the 4 connected agent nodes with real-time glowing states."""
    st.markdown('<div class="pipeline-container">', unsafe_allow_html=True)
    st.markdown('<div class="pipeline-grid">', unsafe_allow_html=True)

    statuses = st.session_state.agent_statuses
    messages = st.session_state.agent_messages

    for key, meta in AGENT_META.items():
        status = statuses.get(key, "pending")
        message = messages.get(key, "") or meta["desc"]
        badge_text = status.upper()

        st.markdown(
            f"""
            <div class="agent-node {status}">
                <div class="node-header">
                    <div class="node-icon-wrapper">{meta['icon']}</div>
                    <span class="node-badge badge-{status}">{badge_text}</span>
                </div>
                <div>
                    <div class="node-name">{meta['name']}</div>
                    <div style="font-size: 0.68rem; color: var(--accent-cyan); font-weight: 600; margin-bottom: 4px;">{meta['tech']}</div>
                    <div class="node-desc">{html.escape(message)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('</div></div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Component: Circular Radial Score Badge
# ---------------------------------------------------------------------------
def render_score_badge(score: int, threshold: int, approved: bool) -> None:
    """Render an aesthetic radial score badge."""
    pct = max(0, min(100, score))
    color = "#10b981" if approved else ("#f59e0b" if score >= threshold * 0.75 else "#f43f5e")

    st.markdown(
        f"""
        <div class="score-badge-card" style="--score-pct: {pct}; --score-color: {color};">
            <div class="radial-score">
                <div class="radial-score-inner">
                    <div class="radial-score-val" style="color: {color};">{score}</div>
                    <div class="radial-score-denom">OUT OF 100</div>
                </div>
            </div>
            <div style="font-weight: 700; font-size: 1.1rem; color: #f8fafc; margin-bottom: 4px;">
                {'✅ APPROVED' if approved else '⚠️ REVISION FLAG'}
            </div>
            <div style="font-size: 0.8rem; color: var(--text-secondary);">
                Threshold required: <strong>{threshold} pts</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Component: Sidebar
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    """Render the sleek dark sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="brand-icon">🔬</div>
                <div>
                    <div class="brand-title">ResearchFlow AI</div>
                    <div class="brand-subtitle">Autonomous Multi-Agent</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Primary Navigation
        st.markdown('<div class="stat-label">Workspace</div>', unsafe_allow_html=True)
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("🔬 New", use_container_width=True):
                st.session_state.current_view = "research"
                st.session_state.history_session_id = None
                st.rerun()
        with col_nav2:
            if st.button("📚 History", use_container_width=True):
                st.session_state.current_view = "history"
                st.rerun()

        st.divider()

        # Agent Live Telemetry
        st.markdown('<div class="stat-label">Active Agents</div>', unsafe_allow_html=True)
        for key, meta in AGENT_META.items():
            status = st.session_state.agent_statuses.get(key, "pending")
            status_dot = {
                "pending": "⚪",
                "running": "🔵",
                "completed": "🟢",
                "failed": "🔴",
            }.get(status, "⚪")
            st.markdown(
                f"""
                <div style="display:flex; justify-content:space-between; align-items:center; padding: 5px 0; font-size: 0.85rem;">
                    <span>{meta['icon']} <strong>{meta['name']}</strong></span>
                    <span style="font-size: 0.72rem; color: var(--text-secondary);">{status_dot} {status.upper()}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.divider()

        # API & Environment Configuration
        st.markdown('<div class="stat-label">System Diagnostics</div>', unsafe_allow_html=True)
        api_errors = config.validate_api_keys()
        if api_errors:
            st.markdown(
                """
                <div style="background: rgba(244,63,94,0.12); border: 1px solid rgba(244,63,94,0.4); border-radius: 8px; padding: 10px; margin-bottom: 8px;">
                    <div style="color: #fb7185; font-weight: 700; font-size: 0.8rem;">⚠️ Keys Missing</div>
                    <div style="font-size: 0.72rem; color: #cbd5e1; margin-top: 4px;">Update researchflow-ai/.env</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.4); border-radius: 8px; padding: 8px 12px; margin-bottom: 8px;">
                    <span style="color: #34d399; font-weight: 700; font-size: 0.82rem;">✓ Ready for Autonomous Research</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="font-size: 0.75rem; color: var(--text-muted); line-height: 1.5; margin-top: 8px;">
                LLM Engine: <code style="color:var(--accent-cyan);">{config.OPENAI_MODEL}</code><br>
                Search Engine: <code style="color:var(--accent-cyan);">Tavily Python SDK</code>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# View: Main Research Panel
# ---------------------------------------------------------------------------
def render_research_panel() -> None:
    """Render the primary research input and live execution panel."""
    # Hero Glass Header
    st.markdown(
        """
        <div class="hero-glass">
            <div class="hero-pill">⚡ LangChain LCEL • OpenAI • Tavily API • Streamlit</div>
            <h1 class="hero-headline">Autonomous Multi-Agent Researcher</h1>
            <p class="hero-desc">
                Deploy an autonomous swarm of four specialized AI agents to discover, extract, synthesize,
                and critically peer-review rigorous research reports with fully verifiable citations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # API Warning banner if keys missing
    api_errors = config.validate_api_keys()
    if api_errors:
        st.error("⚠️ **API Keys Required**: Please configure `OPENAI_API_KEY` and `TAVILY_API_KEY` in `researchflow-ai/.env` to start live runs.")
        return

    # Configuration Container
    with st.container():
        st.markdown('<div class="stat-label">Research Directive</div>', unsafe_allow_html=True)

        # Quick topic suggestions
        st.markdown(
            """
            <div style="margin-bottom: 8px;">
                <span class="suggestion-pill">💡 Quantum computing impact on RSA cryptography</span>
                <span class="suggestion-pill">💡 Solid-state battery commercialization timeline</span>
                <span class="suggestion-pill">💡 AI agent workflows in clinical medicine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        research_topic = st.text_area(
            "Topic input",
            placeholder="e.g., 'Recent breakthroughs in nuclear fusion energy and commercial reactor viability'",
            height=85,
            label_visibility="collapsed",
            key="topic_input_box",
        )

        # Parameters Row
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        with col_p1:
            search_depth = st.selectbox(
                "Search Depth",
                options=["quick", "standard", "deep"],
                index=1,
                help="Quick: 2 queries, 5 sources | Standard: 4 queries, 8 sources | Deep: 6 queries, 15 sources",
            )
        with col_p2:
            max_sources = st.slider("Max Sources", min_value=3, max_value=20, value=8)
        with col_p3:
            max_revisions = st.slider("Max Revisions", min_value=0, max_value=4, value=2,
                                      help="Maximum Critic ↔ Writer revision loops")
        with col_p4:
            approval_threshold = st.slider("Approval Threshold", min_value=50, max_value=95, value=70,
                                           help="Minimum Critic score to pass peer review")

        # Start button
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            start_clicked = st.button(
                "🚀 Launch Research",
                disabled=st.session_state.is_running,
                use_container_width=True,
            )

        if start_clicked:
            if not research_topic.strip():
                st.warning("Please provide a research topic or question before starting.")
            else:
                # Reset run state
                st.session_state.research_result = None
                st.session_state.execution_logs = []
                st.session_state.agent_statuses = {k: "pending" for k in AGENT_META}
                st.session_state.agent_messages = {k: "" for k in AGENT_META}
                st.session_state.is_running = True
                st.session_state.run_start_time = time.time()

                config_obj = ResearchConfiguration(
                    research_topic=research_topic.strip(),
                    search_depth=search_depth,
                    max_sources=max_sources,
                    max_revisions=max_revisions,
                    approval_threshold=approval_threshold,
                )

                thread = threading.Thread(
                    target=run_research_in_thread,
                    args=(config_obj,),
                    daemon=True,
                )
                thread.start()
                st.rerun()

    # Live Workflow Visualization
    if st.session_state.is_running or st.session_state.research_result:
        st.markdown("---")
        st.markdown('<div class="stat-label">Agent Swarm Live Orchestration</div>', unsafe_allow_html=True)
        render_pipeline_visualization()

        # Progress telemetry
        if st.session_state.is_running:
            completed_count = sum(1 for s in st.session_state.agent_statuses.values() if s == "completed")
            progress_val = completed_count / len(AGENT_META)
            st.progress(progress_val)

            elapsed_str = ""
            if st.session_state.run_start_time:
                elapsed = time.time() - st.session_state.run_start_time
                elapsed_str = f" • Elapsed: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

            st.caption(f"⚡ Swarm active{elapsed_str} — Auto-refreshing every 2 seconds...")
            time.sleep(2)
            st.rerun()

    # Results Display
    if st.session_state.research_result:
        render_results_panel(st.session_state.research_result)


# ---------------------------------------------------------------------------
# View: Results Dashboard
# ---------------------------------------------------------------------------
def render_results_panel(result) -> None:
    """Render the aesthetic results dashboard with metrics, tabs, and export."""
    st.markdown("---")
    st.markdown('<div class="stat-label">Synthesized Intelligence & Peer Review</div>', unsafe_allow_html=True)

    # High-level Metrics Row
    st.markdown('<div class="stats-grid">', unsafe_allow_html=True)

    sources_count = len(result.search_output.sources) if result.search_output else 0
    read_count = result.reader_output.successful_extractions if result.reader_output else 0
    word_count = result.writer_output.word_count if result.writer_output else 0
    critic_score = result.final_score
    revisions = result.revision_count

    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-label">Sources Discovered</div>
            <div class="stat-value">{sources_count}</div>
            <div class="stat-sub">🌐 Live Web URLs</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Content Extracted</div>
            <div class="stat-value">{read_count}</div>
            <div class="stat-sub">📄 Clean Scrapes</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Report Length</div>
            <div class="stat-value">{word_count:,}</div>
            <div class="stat-sub">✍️ Words Synthesized</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Critic Score</div>
            <div class="stat-value" style="color: {'#10b981' if critic_score >= 70 else '#f59e0b'};">{critic_score}</div>
            <div class="stat-sub">🎯 100-Point Evaluation</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Revisions Run</div>
            <div class="stat-value">{revisions}</div>
            <div class="stat-sub">🔄 Feedback Loops</div>
        </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabbed Interface
    tab_report, tab_sources, tab_extracted, tab_critic, tab_logs = st.tabs([
        "📄 Research Report",
        "🌐 Discovered Sources",
        "📋 Extracted Documents",
        "🎯 Critic Evaluation",
        "📊 Execution Telemetry",
    ])

    # ---- Tab 1: Research Report ----
    with tab_report:
        if result.final_report:
            col_d1, col_d2 = st.columns([1, 4])
            with col_d1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=result.final_report,
                    file_name=f"researchflow_{result.research_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_d2:
                st.caption(f"Session ID: `{result.research_id}` • Completed: `{result.completed_at or 'Just now'}`")

            # Report paper wrapper
            st.markdown('<div class="report-paper">', unsafe_allow_html=True)
            st.markdown(result.final_report)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No report generated.")

    # ---- Tab 2: Sources ----
    with tab_sources:
        if result.search_output and result.search_output.sources:
            st.markdown(
                f"**{len(result.search_output.sources)} sources retrieved across {len(result.search_output.search_queries)} generated queries**"
            )
            for q in result.search_output.search_queries:
                st.markdown(f'<span class="suggestion-pill">🔍 {html.escape(q)}</span>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            for source in result.search_output.sources:
                with st.expander(f"[{source.source_id}] {source.title or source.url}", expanded=False):
                    c_src1, c_src2 = st.columns([4, 1])
                    with c_src1:
                        st.markdown(f"🔗 [Direct Link]({source.url})")
                        if source.published_date:
                            st.caption(f"Published: {source.published_date}")
                        st.markdown(source.snippet)
                    with c_src2:
                        st.metric("Relevance", f"{source.relevance_score:.2f}")
        else:
            st.info("No sources recorded.")

    # ---- Tab 3: Extracted Documents ----
    with tab_extracted:
        if result.reader_output and result.reader_output.extracted_documents:
            docs = result.reader_output.extracted_documents
            succ = sum(1 for d in docs if d.extraction_status in ("success", "fallback"))
            fail = sum(1 for d in docs if d.extraction_status == "failed")
            st.markdown(f"**Extraction Summary**: `{succ} Successful / Fallback` • `{fail} Failed`")

            for doc in docs:
                status_icon = "✅" if doc.extraction_status in ("success", "fallback") else "❌"
                with st.expander(f"{status_icon} [{doc.source_id}] {doc.title or doc.url}", expanded=False):
                    st.caption(f"Status: `{doc.extraction_status}` • Character Count: `{doc.content_length:,}`")
                    if doc.failure_reason:
                        st.warning(f"Fallback reason: {doc.failure_reason}")
                    if doc.headings:
                        st.markdown("**Identified Headings:** " + " • ".join(f"`{h}`" for h in doc.headings[:6]))
                    if doc.extracted_text:
                        st.text_area(
                            "Cleaned Content Preview",
                            value=doc.extracted_text[:2500] + ("..." if len(doc.extracted_text) > 2500 else ""),
                            height=180,
                            disabled=True,
                            key=f"ext_{doc.source_id}",
                        )
        else:
            st.info("No extracted documents recorded.")

    # ---- Tab 4: Critic Evaluation ----
    with tab_critic:
        if result.critic_output:
            crit = result.critic_output
            col_cr1, col_cr2 = st.columns([1, 2])

            with col_cr1:
                render_score_badge(
                    score=crit.overall_score,
                    threshold=result.configuration.approval_threshold,
                    approved=crit.approved,
                )

                st.markdown("<br>", unsafe_allow_html=True)
                cats = crit.category_scores
                cat_weights = [
                    ("Factual Accuracy", cats.factual_accuracy, 25),
                    ("Research Depth", cats.research_depth, 20),
                    ("Source Quality", cats.source_quality, 20),
                    ("Citation Quality", cats.citation_quality, 15),
                    ("Organization", cats.organization, 10),
                    ("Readability", cats.readability, 10),
                ]

                for label, score, max_s in cat_weights:
                    ratio = score / max_s
                    bar_color = "#10b981" if ratio >= 0.75 else ("#f59e0b" if ratio >= 0.55 else "#f43f5e")
                    st.markdown(
                        f"""
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin: 6px 0 2px 0;">
                            <span style="color:var(--text-secondary);">{label}</span>
                            <span style="font-weight:700; color:{bar_color};">{score}/{max_s}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.progress(ratio)

            with col_cr2:
                if crit.strengths:
                    st.markdown("#### 🌟 Key Strengths")
                    for s in crit.strengths:
                        st.markdown(f"• {s}")

                if crit.weaknesses:
                    st.markdown("#### ⚠️ Identified Weaknesses")
                    for w in crit.weaknesses:
                        st.markdown(f"• {w}")

                if crit.unsupported_claims:
                    st.markdown("#### 🔍 Claims Lacking Verification")
                    for c in crit.unsupported_claims:
                        st.markdown(f"• _{c}_")

                if crit.revision_recommendations:
                    st.markdown("#### 💡 Targeted Recommendations")
                    for r in crit.revision_recommendations:
                        st.markdown(f"• {r}")
        else:
            st.info("No critic output recorded.")

    # ---- Tab 5: Execution Telemetry ----
    with tab_logs:
        logs = st.session_state.execution_logs or (result.execution_logs if result else [])
        if logs:
            st.download_button(
                "⬇️ Download Session Logs",
                data="\n".join(logs),
                file_name=f"research_logs_{result.research_id}.txt",
                mime="text/plain",
            )
            st.text_area(
                "Console Log Stream",
                value="\n".join(logs),
                height=350,
                disabled=True,
            )
        else:
            st.info("No logs found for this run.")


# ---------------------------------------------------------------------------
# View: Research History Browser
# ---------------------------------------------------------------------------
def render_history_panel() -> None:
    """Render the research history browser with search, metrics, and details."""
    st.markdown(
        """
        <div class="hero-glass">
            <h1 class="hero-headline" style="font-size: 2rem;">📚 Research Archive</h1>
            <p class="hero-desc">
                Review previous autonomous research runs, inspect past reports, and export markdown files.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sessions = get_all_sessions(limit=50)
    if not sessions:
        st.info("No research sessions recorded yet in SQLite database.")
        return

    st.markdown(f"**Found {len(sessions)} archived sessions**")

    for sess in sessions:
        score = sess.get("final_score", 0)
        score_color = "#10b981" if score >= 70 else ("#f59e0b" if score >= 50 else "#f43f5e")
        sess_id = sess["research_id"]

        col_h1, col_h2, col_h3 = st.columns([4, 1, 1])
        with col_h1:
            st.markdown(
                f"""
                <div style="font-weight:700; font-size:1.05rem; color:#f8fafc;">
                    {html.escape(sess['research_topic'])}
                </div>
                <div style="font-size:0.75rem; color:var(--text-muted); margin-top:3px;">
                    ID: <code>{sess_id}</code> • Started: {sess.get('started_at', '')[:19]}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_h2:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <div style="font-size:1.4rem; font-weight:800; color:{score_color};">{score}</div>
                    <div style="font-size:0.65rem; color:var(--text-muted); text-transform:uppercase;">Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_h3:
            if st.button("Open Report", key=f"open_{sess_id}", use_container_width=True):
                st.session_state.history_session_id = sess_id

        st.markdown('<hr style="border-color: rgba(255,255,255,0.06); margin: 0.75rem 0;">', unsafe_allow_html=True)

    # Session Inspector
    if st.session_state.history_session_id:
        st.markdown("---")
        full_session = get_session_by_id(st.session_state.history_session_id)
        if full_session and full_session.get("final_report"):
            st.markdown(f"### 📄 {full_session['research_topic']}")
            col_b1, _ = st.columns([1, 4])
            with col_b1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=full_session["final_report"],
                    file_name=f"archive_{full_session['research_id']}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            st.markdown('<div class="report-paper">', unsafe_allow_html=True)
            st.markdown(full_session["final_report"])
            st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main App Entry Point
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
