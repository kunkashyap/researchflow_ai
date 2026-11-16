"""
utils/config.py
Application configuration and environment variable management.

Loads settings from the .env file and validates that required API keys
are present before any research session is started. This is the single
source-of-truth for configuration throughout the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve the project root (two levels up from this file: utils/ → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load .env from the project root
load_dotenv(PROJECT_ROOT / ".env")


class Config:
    """
    Centralised configuration object.
    All settings are read from environment variables so that no secrets
    are ever hardcoded in source files.
    """

    # --- OpenAI ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    # --- Tavily ---
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # --- Logging ---
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # --- Paths ---
    PROJECT_ROOT: Path = PROJECT_ROOT
    DATABASE_DIR: Path = PROJECT_ROOT / "database"
    EXPORTS_DIR: Path = PROJECT_ROOT / "exports"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

    # --- Research defaults ---
    DEFAULT_MAX_SOURCES: int = 8
    DEFAULT_MAX_REVISIONS: int = 2
    DEFAULT_SEARCH_DEPTH: str = "standard"   # quick | standard | deep
    APPROVAL_THRESHOLD: int = 70              # Critic score required for approval
    MAX_CONCURRENT_SCRAPERS: int = 5
    SCRAPE_TIMEOUT: int = 15                  # seconds per request
    SCRAPE_MAX_CONTENT_BYTES: int = 500_000   # 500 KB per page

    @classmethod
    def ensure_directories(cls) -> None:
        """Create required runtime directories if they don't exist."""
        for directory in [cls.DATABASE_DIR, cls.EXPORTS_DIR, cls.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def validate_api_keys(cls) -> list[str]:
        """
        Check that all required API keys are present and non-empty.
        Returns a list of error messages (empty list means all OK).
        """
        errors: list[str] = []
        if not cls.OPENAI_API_KEY or cls.OPENAI_API_KEY == "your_openai_api_key_here":
            errors.append(
                "OPENAI_API_KEY is missing. Please set it in your .env file."
            )
        if not cls.TAVILY_API_KEY or cls.TAVILY_API_KEY == "your_tavily_api_key_here":
            errors.append(
                "TAVILY_API_KEY is missing. Please set it in your .env file."
            )
        return errors


# Singleton instance used throughout the application
config = Config()

# Ensure runtime directories exist on import
config.ensure_directories()
