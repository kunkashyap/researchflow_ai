"""
tests/conftest.py
Shared pytest configuration and fixtures.

Sets up the Python path so that project modules can be imported
in tests without needing to install the project as a package.
"""

import sys
import os

# Add the project root to sys.path so imports like `from agents.search_agent import ...`
# work correctly regardless of how pytest is invoked.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set a dummy OPENAI_API_KEY and TAVILY_API_KEY so that config.py doesn't
# print warnings during test collection (tests mock the actual API calls).
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy-key-for-testing-only")
os.environ.setdefault("TAVILY_API_KEY", "tvly-test-dummy-key-for-testing-only")
os.environ.setdefault("LOG_LEVEL", "ERROR")  # Suppress log noise during tests
