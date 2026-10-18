"""Rule-based and LLM extractors."""

from __future__ import annotations

import json

import pytest

from rag_graph import EntityRelationExtractor, Extraction, RuleBasedExtractor
from rag_graph.extractor import _parse_json


def triples(text):
    ex = RuleBasedExtractor().extract(text)
    return {(t.subject, t.predicate, t.object) for t in ex.triples}


def types(text):
    return {e.name: e.type for e in RuleBasedExtractor().extract(text).entities}


# ---- rule-based -------------------------------------------------------------------------------

def test_founded_active_and_passive():
    assert triples("Ann Lee founded Acme Corp in 2010.") == {("Ann Lee", "founded", "Acme Corp")}
    assert triples("Acme Corp was founded by Ann Lee.") == {("Ann Lee", "founded", "Acme Corp")}


def test_co_founded_and_hyphen_spelling():
    assert triples("Bo Chan co-founded Delta Works.") == {("Bo Chan", "founded", "Delta Works")}
    assert triples("Bo Chan cofounded Delta Works.") == {("Bo Chan", "founded", "Delta Works")}


def test_lists_of_founders_with_and_without_the_oxford_comma():
    want = {("A One", "founded", "Zed Corp"), ("B Two", "founded", "Zed Corp"),
            ("C Three", "founded", "Zed Corp")}
    assert triples("Zed Corp was founded by A One, B Two, and C Three.") == want
    assert triples("Zed Corp was founded by A One, B Two and C Three.") == want
    assert triples("A One, B Two and C Three founded Zed Corp.") == want
    assert triples("A One and B Two founded Zed Corp.") == {("A One", "founded", "Zed Corp"),
                                                           ("B Two", "founded", "Zed Corp")}


def test_a_leading_clause_is_not_mistaken_for_a_founder_list():
    t = triples("After leaving Old Lab, Mustafa Said co-founded Inflect AI in 2022.")
    assert t == {("Mustafa Said", "founded", "Inflect AI")}


def test_relative_clause_subject():
    t = triples("Demis Roe is a neuroscientist who co-founded DeepWell.")
    assert t == {("Demis Roe", "founded", "DeepWell")}


def test_acquired_active_and_passive():
    assert triples("Big Co acquired Small Co in 2019.") == {("Big Co", "acquired", "Small Co")}
    assert triples("Small Co was acquired by Big Co.") == {("Big Co", "acquired", "Small Co")}


def test_headquartered_with_an_adjective_phrase():
    t = triples("Kalden Labs is a software company headquartered in Telmouth.")
    assert t == {("Kalden Labs", "headquartered in", "Telmouth")}
    assert triples("Kalden Labs is headquartered in Telmouth.") == t


def test_makes_and_located_in_and_ceo():
    assert triples("Kalden Labs makes Aurex-4.") == {("Kalden Labs", "makes", "Aurex-4")}
    assert triples("Telmouth is a port in Veldor.") == {("Telmouth", "located in", "Veldor")}
    assert triples("Ann Lee is the CEO of Acme Corp.") == {("Ann Lee", "leads", "Acme Corp")}


def test_entity_types_follow_the_pattern():
    assert types("Ann Lee founded Acme Corp.") == {"Ann Lee": "Person", "Acme Corp": "Organisation"}
    assert types("Acme Corp is headquartered in Portsea.")["Portsea"] == "Place"


def test_evidence_is_the_sentence_trimmed_to_25_words():
    ex = RuleBasedExtractor().extract("Ann Lee founded Acme Corp in 2010.")
    assert ex.triples[0].evidence == "Ann Lee founded Acme Corp in 2010."
    long_sentence = "Ann Lee founded Acme Corp " + "and so on " * 30 + "."
    assert len(RuleBasedExtractor().extract(long_sentence).triples[0].evidence.split()) <= 25


def test_several_sentences_and_duplicates():
    text = ("Ann Lee founded Acme Corp. Ann Lee founded Acme Corp.\n"
            "Big Co acquired Acme Corp. Acme Corp makes Gizmo-2.")
    ex = RuleBasedExtractor().extract(text)
    assert len(ex.triples) == 3 and len(ex.entities) == 4


def test_no_match_gives_an_empty_extraction():
    ex = RuleBasedExtractor().extract("It rained all week and the harbour was grey.")
    assert ex.entities == [] and ex.triples == []
    assert RuleBasedExtractor().extract("").triples == []


def test_a_subject_equal_to_the_object_is_dropped():
    assert triples("Acme Corp acquired Acme Corp.") == set()


def test_a_capitalised_sentence_opener_becomes_part_of_the_name():
    """Documented blind spot: the rules cannot tell 'Yesterday' from a first name."""
    assert triples("Yesterday Ann Lee founded Acme Corp.") == {
        ("Yesterday Ann Lee", "founded", "Acme Corp")}


def test_lowercase_names_are_invisible_to_the_rules():
    """Documented blind spot: patterns need capitalised names."""
    assert triples("ann lee founded acme corp.") == set()


def test_unicode_names():
    t = triples("Zoë Müller founded Café Noir in 2001.")
    assert ("Zoë Müller", "founded", "Café Noir") in t


def test_custom_patterns():
    from rag_graph import Pattern

    ex = RuleBasedExtractor((Pattern(r"(?P<s>[A-Z]\w+) reports to (?P<o>[A-Z]\w+)", "reports to",
                                    "Person", "Person"),))
    t = ex.extract("Alice reports to Bob.").triples
    assert [(x.subject, x.predicate, x.object) for x in t] == [("Alice", "reports to", "Bob")]


def test_rule_based_extraction_is_deterministic():
    text = "Ann Lee founded Acme Corp. Big Co acquired Acme Corp."
    assert RuleBasedExtractor().extract(text) == RuleBasedExtractor().extract(text)


# ---- LLM extractor with a fake transport ----------------------------------------------------------------

GOOD = json.dumps({
    "entities": [{"name": "Ann Lee", "type": "Person"}, {"name": "Acme", "type": "Organisation"}],
    "triples": [{"subject": "Ann Lee", "predicate": "founded", "object": "Acme", "evidence": "x"}],
})


def llm(reply):
    seen = []

    def chat(system, user, max_tokens):
        seen.append((system, user, max_tokens))
        return reply

    return EntityRelationExtractor(model="claude-sonnet-4-5", chat=chat), seen


def test_llm_extractor_parses_valid_json_and_sends_the_chunk():
    ex, seen = llm(GOOD)
    out = ex.extract("some chunk")
    assert isinstance(out, Extraction) and out.triples[0].predicate == "founded"
    assert seen[0][1] == "some chunk" and "strict JSON" in seen[0][0] and seen[0][2] == 1024


def test_llm_json_inside_prose_and_fences():
    ex, _ = llm("Sure! ```json\n" + GOOD + "\n``` hope that helps {not json}")
    assert len(ex.extract("x").entities) == 2


def test_the_first_object_wins_over_trailing_braces():
    """A greedy brace-to-brace regex swallowed the trailing prose and failed to parse."""
    assert _parse_json('{"a": 1} then {"b": 2}') == {"a": 1}
    assert _parse_json('lead-in {"a": {"b": [1, 2]}} done {x}') == {"a": {"b": [1, 2]}}


@pytest.mark.parametrize("reply", ["", "no json at all", "[1, 2]", "{broken"])
def test_llm_reply_without_json_raises(reply):
    ex, _ = llm(reply)
    with pytest.raises(ValueError, match="no JSON"):
        ex.extract("x")


def test_llm_reply_with_the_wrong_schema_raises():
    ex, _ = llm(json.dumps({"entities": [{"name": "A", "type": "Spaceship"}], "triples": []}))
    with pytest.raises(ValueError, match="invalid schema"):
        ex.extract("x")


def test_blank_entity_or_triple_fields_are_rejected():
    for bad in ({"entities": [{"name": "  ", "type": "Person"}]},
                {"triples": [{"subject": "A", "predicate": "", "object": "B"}]}):
        ex, _ = llm(json.dumps(bad))
        with pytest.raises(ValueError, match="invalid schema"):
            ex.extract("x")


def test_names_are_whitespace_normalised():
    ex, _ = llm(json.dumps({"entities": [{"name": " Ann\n  Lee ", "type": "Person"}]}))
    assert ex.extract("x").entities[0].name == "Ann Lee"


def test_missing_keys_default_to_empty():
    ex, _ = llm("{}")
    assert ex.extract("x") == Extraction()


def test_provider_is_inferred_from_the_model_prefix_and_no_sdk_is_touched_at_construction():
    assert EntityRelationExtractor("claude-sonnet-4-5", chat=lambda *a: "{}").provider == "anthropic"
    assert EntityRelationExtractor("gpt-4o", chat=lambda *a: "{}").provider == "openai"
    EntityRelationExtractor("claude-sonnet-4-5")          # builds a lazy chat; imports nothing yet


def test_a_pathologically_nested_reply_is_a_value_error():
    ex, _ = llm('{"a":' * 5000)
    with pytest.raises(ValueError):
        ex.extract("x")


def test_a_relation_wrapped_across_lines_is_still_extracted():
    """Single newlines used to end a sentence, so a hard-wrapped paragraph lost its relations."""
    text = "In 2011 the engineer Ann Lee founded\nAcme Corp in a garage. Zenith Ltd\nacquired Acme Corp."
    assert triples(text) == {("Ann Lee", "founded", "Acme Corp"),
                             ("Zenith Ltd", "acquired", "Acme Corp")}


def test_lines_without_punctuation_are_not_glued_into_one_name():
    """A heading or list item still ends a sentence, so names on adjacent lines stay apart."""
    text = "# Acme Corp\nAnn Lee founded Acme Corp.\n- Zenith Ltd acquired Acme Corp\n- Bo Chen founded Zenith Ltd"
    assert triples(text) == {("Ann Lee", "founded", "Acme Corp"),
                             ("Zenith Ltd", "acquired", "Acme Corp"),
                             ("Bo Chen", "founded", "Zenith Ltd")}


# ---- hostile input: the patterns must stay linear ---------------------------------------------

@pytest.mark.parametrize("text", [
    "Alpha " * 3000 + "ends.",                                   # was ~50 s: cubic backtracking
    "A" + "-A" * 4000 + " founded",                              # one unbreakable token
    ", ".join(["Alpha"] * 3000) + " and Beta founded",           # a very long founder list
    "Ann is " * 2000 + "who",                                    # lazy gap rescanned per "is"
], ids=["capitalised-run", "hyphen-token", "long-list", "repeated-is"])
def test_extraction_time_is_bounded_on_hostile_text(text):
    """The unbounded name/list/gap repeats made extraction time explode with input length (4,000
    capitalised words took 121 s). The extractor also runs on the user's question."""
    import time

    start = time.perf_counter()
    RuleBasedExtractor().extract(text)
    assert time.perf_counter() - start < 15


def test_a_founder_list_of_25_names_is_still_extracted_whole():
    names = [f"Person{chr(65 + i)}" for i in range(25)]
    got = triples(", ".join(names[:-1]) + f", and {names[-1]} founded Acme Corp.")
    assert got == {(n, "founded", "Acme Corp") for n in names}
