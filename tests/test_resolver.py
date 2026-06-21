"""Tests for entity resolution and alias merging."""

from rag_graph.resolver import EntityResolver, normalize_surface


def test_normalize_lowercases_and_strips_punct():
    assert normalize_surface("OpenAI") == "openai"
    assert normalize_surface("Open AI") == "ai open"  # tokens sorted
    assert normalize_surface("Open-AI!") == "ai open"


def test_resolver_merges_aliases():
    r = EntityResolver()
    a, is_new = r.resolve("OpenAI")
    assert is_new
    b, is_new = r.resolve("openai")
    assert not is_new
    assert a == b
    assert "OpenAI" in r.aliases(a)
    assert "openai" in r.aliases(a)


def test_resolver_separates_distinct():
    r = EntityResolver()
    a, _ = r.resolve("Anthropic")
    b, _ = r.resolve("OpenAI")
    assert a != b