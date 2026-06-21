"""Heading-aware document chunker.

The naive recursive-character splitter doesn't respect document structure.
For markdown / HTML / heading-y plaintext, we want:
  - One chunk per heading section
  - If a section exceeds max_chars, split inside it on paragraphs
  - Carry the heading path forward as metadata so retrieval can show breadcrumbs

This is the single most underrated thing you can do to lift RAG quality — far
more leverage than fiddling with chunk_size in token units.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class ChunkDraft:
    text: str
    heading_path: list[str]   # e.g. ["Architecture", "Retrieval"]


def chunk_document(text: str, *, max_chars: int = 1500, min_chars: int = 200) -> list[ChunkDraft]:
    """Split into heading sections, then by paragraph when oversize."""
    sections = _split_by_heading(text)
    out: list[ChunkDraft] = []
    for section in sections:
        if len(section.text) <= max_chars:
            out.append(section)
            continue
        # Section too big — split on blank lines
        paras = [p.strip() for p in re.split(r"\n\s*\n", section.text) if p.strip()]
        buf = ""
        for p in paras:
            if len(buf) + len(p) + 2 > max_chars and len(buf) >= min_chars:
                out.append(ChunkDraft(text=buf.strip(), heading_path=section.heading_path))
                buf = p
            else:
                buf = (buf + "\n\n" + p) if buf else p
        if buf:
            out.append(ChunkDraft(text=buf.strip(), heading_path=section.heading_path))
    return out


def _split_by_heading(text: str) -> list[ChunkDraft]:
    """Walk through the text, producing one ChunkDraft per heading section.

    Heading path is the stack of containing headings, so a level-3 heading inside
    a level-2 inside a level-1 carries all three as its `heading_path`.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [ChunkDraft(text=text.strip(), heading_path=[])]

    sections: list[ChunkDraft] = []
    path_stack: list[tuple[int, str]] = []  # (level, title)

    # Anything before the first heading
    if matches[0].start() > 0:
        prelude = text[:matches[0].start()].strip()
        if prelude:
            sections.append(ChunkDraft(text=prelude, heading_path=[]))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        title = m.group(2)
        # Pop the stack to this level
        while path_stack and path_stack[-1][0] >= level:
            path_stack.pop()
        path_stack.append((level, title))

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append(ChunkDraft(text=body, heading_path=[t for _, t in path_stack]))

    return sections