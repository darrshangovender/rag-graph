"""Heading-aware document chunker.

  - One chunk per heading section; the heading path (breadcrumb) travels with it.
  - Headings inside fenced code blocks are not headings.
  - A section over `max_chars` is split on paragraphs, then on sentences, then on words, and a
    single token longer than `max_chars` is cut, so no chunk ever exceeds `max_chars`.
  - `ChunkDraft.embed_text` prepends the breadcrumb: a chunk under "# Demis Hassabis" that never
    repeats the name still matches a query for it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# An optional closing run of #s needs whitespace before it ("# C#" is a heading about C#).
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*$", re.MULTILINE)
_FENCE_RE = re.compile(r"^[ \t]*(```|~~~)", re.MULTILINE)
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
# A line that starts a new block even without a blank line before it: heading, list item, quote,
# table row or code fence.
_BLOCK_START_RE = re.compile(r"^(?:#{1,6}[ \t]|[-*+][ \t]|\d+[.)][ \t]|>|\||```|~~~)")
_ORDERED_ITEM_RE = re.compile(r"\d+[.)] ")


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, rejoining hard-wrapped lines.

    A single newline inside a paragraph is a soft wrap, not a sentence end: "a company that moves
    city\\nloses people." is one sentence. Blank lines, heading lines and the start of a list item,
    quote, table row or fence still end the current block. Whitespace inside a sentence is
    collapsed to single spaces, so patterns written with one space match wrapped text.
    """
    blocks: list[str] = []
    current: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or _BLOCK_START_RE.match(line):
            if current:
                blocks.append(" ".join(current))
                current = []
        if not line:
            continue
        current.append(line)
        if HEADING_RE.match(line):          # a heading is a block on its own
            blocks.append(" ".join(current))
            current = []
    if current:
        blocks.append(" ".join(current))
    out: list[str] = []
    for block in blocks:
        block = " ".join(block.split())
        marker = _ORDERED_ITEM_RE.match(block)          # "1. first" is not "1." + "first"
        prefix = marker.group(0) if marker else ""
        sents = [s for s in _SENTENCE_RE.split(block[len(prefix):]) if s]
        if sents:
            sents[0] = prefix + sents[0]
        out.extend(sents)
    return out


@dataclass
class ChunkDraft:
    text: str
    heading_path: list[str]   # e.g. ["Architecture", "Retrieval"]

    @property
    def embed_text(self) -> str:
        """Breadcrumb + text: what should be embedded and shown to an extractor.

        The breadcrumb is its own paragraph (blank line), so sentence splitting never glues it to
        the first line of the body ("Zenith Ltd" + "Zenith Ltd is ...").
        """
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text


def chunk_document(text: str, *, max_chars: int = 1500, min_chars: int = 200) -> list[ChunkDraft]:
    """Split into heading sections, then by paragraph / sentence / word when oversize.

    Blank input yields no chunks (it used to yield one empty chunk that was embedded and stored).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if min_chars < 0 or min_chars > max_chars:
        raise ValueError("min_chars must be between 0 and max_chars")
    if not text or not text.strip():
        return []
    out: list[ChunkDraft] = []
    for section in _split_by_heading(text.replace("\r\n", "\n").replace("\r", "\n")):
        if len(section.text) <= max_chars:
            out.append(section)
            continue
        out.extend(
            ChunkDraft(text=piece, heading_path=section.heading_path)
            for piece in _pack(_atoms(section.text, max_chars), max_chars, min_chars)
        )
    return out


def _atoms(text: str, max_chars: int) -> list[str]:
    """Pieces no longer than max_chars: paragraphs, else sentences, else words."""
    atoms: list[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= max_chars:
            atoms.append(para)
            continue
        for sent in (s.strip() for s in _SENTENCE_RE.split(para)):
            if not sent:
                continue
            if len(sent) <= max_chars:
                atoms.append(sent)
                continue
            line = ""
            for word in sent.split():
                while len(word) > max_chars:          # one unbreakable token: cut it
                    if line:
                        atoms.append(line)
                        line = ""
                    atoms.append(word[:max_chars])
                    word = word[max_chars:]
                if line and len(line) + 1 + len(word) > max_chars:
                    atoms.append(line)
                    line = word
                else:
                    line = f"{line} {word}" if line else word
            if line:
                atoms.append(line)
    return atoms


def _pack(atoms: list[str], max_chars: int, min_chars: int) -> list[str]:
    """Greedily join atoms into chunks of at most max_chars; avoid a tiny trailing chunk."""
    chunks: list[str] = []
    buf = ""
    for a in atoms:
        joined = f"{buf}\n\n{a}" if buf else a
        if len(joined) <= max_chars:
            buf = joined
        else:
            chunks.append(buf)
            buf = a
    if buf:
        if chunks and len(buf) < min_chars and len(chunks[-1]) + 2 + len(buf) <= max_chars:
            chunks[-1] = f"{chunks[-1]}\n\n{buf}"
        else:
            chunks.append(buf)
    return chunks


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    """Character ranges inside ``` / ~~~ fences (an unclosed fence runs to the end)."""
    ranges, start = [], None
    for m in _FENCE_RE.finditer(text):
        if start is None:
            start = m.start()
        else:
            ranges.append((start, m.end()))
            start = None
    if start is not None:
        ranges.append((start, len(text)))
    return ranges


def _split_by_heading(text: str) -> list[ChunkDraft]:
    """One ChunkDraft per heading section; heading_path is the stack of containing headings."""
    fenced = _fenced_ranges(text)
    matches = [m for m in HEADING_RE.finditer(text)
               if not any(a <= m.start() < b for a, b in fenced)]
    if not matches:
        return [ChunkDraft(text=text.strip(), heading_path=[])]

    sections: list[ChunkDraft] = []
    path_stack: list[tuple[int, str]] = []

    if matches[0].start() > 0:
        prelude = text[:matches[0].start()].strip()
        if prelude:
            sections.append(ChunkDraft(text=prelude, heading_path=[]))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2).strip()
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():body_end].strip()
        if body:
            sections.append(ChunkDraft(text=body, heading_path=[t for _, t in path_stack]))
    return sections
