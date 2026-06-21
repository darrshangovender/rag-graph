"""Tests for the heading-aware chunker."""

from rag_graph.chunker import chunk_document


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