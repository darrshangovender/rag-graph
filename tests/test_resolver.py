"""Tests for entity resolution and alias merging."""

import pytest

from rag_graph import GraphStore
from rag_graph.resolver import EntityResolver, normalize_key, normalize_surface, tokens_of


def test_unicode_punctuation_is_stripped_too():
    assert tokens_of("Acme — “Corp”") == ("acme", "corp")
    assert normalize_key("Acme\u2014Corp") == "acmecorp"


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


# ---- new -----------------------------------------------------------------------------------------

def test_open_ai_and_openai_are_one_entity_as_the_docs_claim():
    r = EntityResolver()
    a, _ = r.resolve("OpenAI")
    assert r.resolve("Open AI") == (a, False)
    assert r.resolve("OPEN-AI") == (a, False)
    assert r.resolve("open.ai") == (a, False)


def test_token_order_is_kept_by_default_so_ai_bias_and_bias_ai_stay_distinct():
    r = EntityResolver()
    a, _ = r.resolve("AI Bias")
    b, new = r.resolve("Bias AI")
    assert new and a != b


def test_sort_tokens_option_merges_reordered_names():
    r = EntityResolver(sort_tokens=True)
    a, _ = r.resolve("John Smith")
    assert r.resolve("Smith John") == (a, False)
    assert r.resolve("Smith, John") == (a, False)


def test_unicode_case_folding():
    r = EntityResolver()
    a, _ = r.resolve("Straße")
    assert r.resolve("STRASSE") == (a, False)
    eid, _ = r.resolve("École")
    assert r.resolve("école") == (eid, False)
    assert r.resolve("Zoë") != r.resolve("Zoe")             # accents are kept, not stripped


@pytest.mark.parametrize("name", ["", "   ", "!!!", "—", "..."])
def test_a_name_with_no_word_characters_is_rejected(name):
    with pytest.raises(ValueError, match="no word characters"):
        EntityResolver().resolve(name)


def test_lookup_never_creates():
    r = EntityResolver()
    assert r.lookup("Nobody") is None and len(r) == 0
    eid, _ = r.resolve("Somebody")
    assert r.lookup("somebody") == eid and len(r) == 1
    assert r.lookup("!!!") is None


def test_ids_are_sequential_from_one():
    r = EntityResolver()
    assert [r.resolve(n)[0] for n in ("A1", "B2", "C3")] == [1, 2, 3]


def test_canonical_is_the_first_surface_form_seen():
    r = EntityResolver()
    eid, _ = r.resolve("Acme Corp")
    r.resolve("ACME CORP")
    assert r.canonical(eid) == "Acme Corp" and r.aliases(eid) == {"Acme Corp", "ACME CORP"}


def test_version_counts_new_entities_and_new_aliases_only():
    r = EntityResolver()
    v0 = r.version
    r.resolve("Acme")
    r.resolve("Acme")                    # known surface form: no change
    v1 = r.version
    r.resolve("ACME")                    # new alias
    assert v0 < v1 < r.version
    stable = r.version
    r.resolve("ACME")
    assert r.version == stable


def test_surface_forms_lists_canonical_and_aliases():
    r = EntityResolver()
    r.resolve("Acme")
    r.resolve("ACME")
    assert sorted(f for f, _ in r.surface_forms()) == ["ACME", "Acme"]
    assert r.entity_ids() == [1]


def test_tokens_and_keys():
    assert tokens_of("Open-AI, Inc.") == ("open", "ai", "inc")
    assert normalize_key("Open-AI") == "openai"
    assert normalize_key("B A", sort_tokens=True) == "ab"
    assert normalize_key("!!!") == ""


def test_from_store_continues_ids_instead_of_reissuing_them(tmp_path):
    """The resolver started at id 1 in every process, so ingesting into an existing database
    reissued ids that belonged to other entities and overwrote them."""
    store = GraphStore(tmp_path / "kg.db")
    store.upsert_entity(entity_id=1, canonical="Acme Corp", type_="Organisation",
                        aliases=["Acme Corp", "ACME"])
    store.upsert_entity(entity_id=7, canonical="Ann Lee", type_="Person", aliases=["Ann Lee"])
    r = EntityResolver.from_store(store)
    assert r.lookup("acme") == 1 and r.lookup("ann lee") == 7
    new_id, is_new = r.resolve("Brand New Thing")
    assert is_new and new_id == 8
    assert r.canonical(1) == "Acme Corp" and r.aliases(1) == {"Acme Corp", "ACME"}


def test_from_store_on_an_empty_store_starts_at_one(tmp_path):
    assert EntityResolver.from_store(GraphStore(tmp_path / "e.db")).resolve("A1") == (1, True)
