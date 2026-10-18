"""Tests for the heading-aware chunker."""

import pytest

from rag_graph.chunker import chunk_document, split_sentences


def test_no_headings_is_one_chunk():
    out = chunk_document("Just plain text. Another sentence.")
    assert len(out) == 1
    assert out[0].heading_path == []


def test_one_heading_per_chunk():
    text = "# A\nText under A\n## B\nText under B\n## C\nText under C"
    out = chunk_document(text)
    assert len(out) == 3
    assert out[0].heading_path == ["A"]
    assert out[1].heading_path == ["A", "B"]
    assert out[2].heading_path == ["A", "C"]


def test_oversize_section_splits_on_paragraphs():
    para = "lorem ipsum dolor sit amet " * 50
    text = f"# H\n{para}\n\n{para}\n\n{para}"
    out = chunk_document(text, max_chars=500, min_chars=100)
    assert len(out) >= 2
    assert all(c.heading_path == ["H"] for c in out)


# ---- new -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "\n\n\n", "\t \r\n"])
def test_blank_input_yields_no_chunks(text):
    """It used to yield one empty chunk that was then embedded and stored."""
    assert chunk_document(text) == []


def test_text_before_the_first_heading_is_a_prelude_chunk():
    out = chunk_document("Intro line.\n\n# Title\nBody.")
    assert [(c.heading_path, c.text) for c in out] == [([], "Intro line."), (["Title"], "Body.")]


def test_a_heading_with_no_body_adds_no_chunk_but_stays_in_the_path():
    out = chunk_document("# Parent\n## Child\nChild text")
    assert len(out) == 1 and out[0].heading_path == ["Parent", "Child"]


def test_heading_levels_pop_correctly():
    text = "# A\na\n### Deep\nd\n## Mid\nm\n# Z\nz"
    paths = [c.heading_path for c in chunk_document(text)]
    assert paths == [["A"], ["A", "Deep"], ["A", "Mid"], ["Z"]]


def test_headings_inside_code_fences_are_not_headings():
    text = "# Real\nbefore\n```python\n# not a heading\nx = 1\n```\nafter"
    out = chunk_document(text)
    assert len(out) == 1 and "# not a heading" in out[0].text


def test_an_unclosed_fence_swallows_the_rest_of_the_document():
    out = chunk_document("# A\ntext\n```\n# hidden\nmore")
    assert [c.heading_path for c in out] == [["A"]]


def test_tilde_fences_work_too():
    out = chunk_document("# A\n~~~\n# x\n~~~\nend")
    assert len(out) == 1


def test_closing_hashes_are_stripped_but_a_trailing_hash_in_the_title_is_kept():
    out = chunk_document("## Closed ##\nx\n# Learning C#\ny")
    assert [c.heading_path for c in out] == [["Closed"], ["Learning C#"]]


def test_crlf_line_endings():
    out = chunk_document("# A\r\nline one\r\n\r\nline two\r\n# B\r\nb")
    assert [c.heading_path for c in out] == [["A"], ["B"]]
    assert "\r" not in out[0].text


@pytest.mark.parametrize("max_chars", [50, 120, 400])
def test_no_chunk_exceeds_max_chars(max_chars):
    para = "alpha beta gamma delta epsilon. " * 40
    text = f"# H\n{para}\n\n{para}\n\n" + "x" * 1000
    for c in chunk_document(text, max_chars=max_chars, min_chars=10):
        assert len(c.text) <= max_chars


def test_a_single_run_on_paragraph_is_split_on_sentences_then_words():
    sentence = "word " * 100
    out = chunk_document(f"# H\n{sentence}", max_chars=80, min_chars=0)
    assert len(out) > 4 and all(len(c.text) <= 80 for c in out)
    assert " ".join(c.text for c in out).split() == sentence.split()


def test_splitting_never_loses_or_reorders_text():
    paras = [f"paragraph number {i} " + "filler " * 30 for i in range(12)]
    text = "# H\n" + "\n\n".join(paras)
    out = chunk_document(text, max_chars=300, min_chars=50)
    assert " ".join(" ".join(c.text.split()) for c in out) == " ".join(" ".join(p.split()) for p in paras)


def test_a_tiny_trailing_chunk_is_merged_into_the_previous_one():
    big = "x" * 100
    out = chunk_document(f"# H\n{big}\n\n{big}\n\ntail", max_chars=230, min_chars=50)
    assert out[-1].text.endswith("tail") and len(out) == 1 or "tail" in out[-1].text


def test_embed_text_prepends_the_breadcrumb():
    out = chunk_document("# Demis Hassabis\n## Career\nHe led the lab.")
    assert out[0].embed_text == "Demis Hassabis > Career\n\nHe led the lab."
    assert out[0].text == "He led the lab."


def test_embed_text_without_headings_is_the_text():
    assert chunk_document("plain")[0].embed_text == "plain"


@pytest.mark.parametrize("kw", [{"max_chars": 0}, {"min_chars": -1}, {"max_chars": 10, "min_chars": 20}])
def test_bad_limits_rejected(kw):
    with pytest.raises(ValueError):
        chunk_document("text", **kw)


def test_unicode_text_and_headings():
    out = chunk_document("# Ünïcode ✓\nTexte en français — ça marche.")
    assert out[0].heading_path == ["Ünïcode ✓"] and "ça" in out[0].text


def test_chunking_is_deterministic():
    text = "# A\n" + "sentence one. sentence two. " * 80
    assert chunk_document(text, max_chars=200) == chunk_document(text, max_chars=200)


# ---- sentence splitting (shared by the extractor and the extractive answerer) -----------------

def test_split_sentences_rejoins_soft_wraps_and_collapses_whitespace():
    assert split_sentences("A company that moves\n  city loses   people. Next one!\nAnd more") == [
        "A company that moves city loses people.", "Next one!", "And more"]


def test_split_sentences_keeps_block_boundaries():
    text = "# Title\nBody text\n\nPara two\n- item one\n- item\n  continued\n1. first\n> quote"
    assert split_sentences(text) == [
        "# Title", "Body text", "Para two", "- item one", "- item continued", "1. first", "> quote"]


@pytest.mark.parametrize("text", ["", "   ", "\n\n \n"])
def test_split_sentences_of_blank_text_is_empty(text):
    assert split_sentences(text) == []
