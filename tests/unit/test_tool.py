"""
Unit tests for document_search tool.

Tests the tool directly without the server — no HTTP calls, no agent.
Uses testing=True to skip Langfuse logging.
"""
import sys
sys.path.insert(0, '../..')

from dotenv import load_dotenv
load_dotenv('../../.env')

import pytest
from tool import make_document_search

document_search = make_document_search(testing=True)


def test_tool_returns_results():
    """Tool should return non-empty text for a valid medical query."""
    result = document_search.invoke({"query": "HbA1c target for type 2 diabetes"})
    assert isinstance(result, str)
    assert len(result) > 0


def test_tool_handles_bad_query():
    """Tool should return a string even for a nonsense query."""
    result = document_search.invoke({"query": "zzzzz nonsense query xyzxyz"})
    assert isinstance(result, str)
