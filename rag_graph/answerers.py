"""Answerers: turn retrieved passages and graph facts into an answer.

  * `ExtractiveAnswerer` - offline, deterministic: picks the passage sentences that share the most
    words with the question and quotes them with their [cN] labels, followed by the graph path.
    It is a baseline that shows what was retrieved, NOT a language model: it cannot reason or
    paraphrase.
  * `LLMAnswerer` - an LLM reads the same material (optional extra, or supply `chat=`).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

from .chunker import split_sentences
from .extractor import default_chat

ANSWER_SYSTEM = """\
You answer questions from supplied context. Rules:
  - Use ONLY facts present in the context.
  - Cite chunks inline like [c1], [c2].
  - The context may include "Graph facts": relations between entities with the chunk they came from.
    Use them to connect facts that sit in different chunks, and cite the chunk.
  - If the context does not answer the question, say so plainly. Do not guess.
"""

_WORD = re.compile(r"[^\W_]+")
_STOP = frozenset(
    "a an and are as at be by did do does for from has have how in is it its of on or that the "
    "their this to was were what when where which who whom whose why with".split()
)


@dataclass
class Passage:
    label: str            # "c1", "c2", ...
    doc_id: str
    breadcrumb: str
    text: str


@dataclass
class AnswerInput:
    question: str
    passages: list[Passage] = field(default_factory=list)
    facts: list[str] = field(default_factory=list)      # "A --founded--> B [c2]"

    def render(self) -> str:
        """The context block an LLM reads."""
        lines = []
        for p in self.passages:
            crumb = p.breadcrumb or "(no heading)"
            lines.append(f"[{p.label}] doc={p.doc_id} | {crumb}\n{p.text}\n")
        if self.facts:
            lines.append("Graph facts:\n" + "\n".join(f"  {f}" for f in self.facts) + "\n")
        return "\n".join(lines)


class Answerer(Protocol):
    def answer(self, inp: AnswerInput) -> str: ...


class ExtractiveAnswerer:
    """Quote the best-matching sentences; no generation."""

    def __init__(self, max_sentences: int = 3) -> None:
        if max_sentences < 1:
            raise ValueError("max_sentences must be >= 1")
        self.max_sentences = max_sentences

    def answer(self, inp: AnswerInput) -> str:
        q = {w for w in _WORD.findall(inp.question.casefold()) if w not in _STOP}
        scored: list[tuple[int, int, int, str, str]] = []
        for rank, p in enumerate(inp.passages):
            for pos, sent in enumerate(split_sentences(p.text)):
                overlap = len(q & {w for w in _WORD.findall(sent.casefold())})
                if overlap:
                    scored.append((-overlap, rank, pos, sent, p.label))
        if not scored:
            return "The retrieved context does not answer this question."
        scored.sort()
        picked = scored[: self.max_sentences]
        picked.sort(key=lambda t: (t[1], t[2]))           # present in retrieval order
        text = " ".join(f"{sent} [{label}]" for _, _, _, sent, label in picked)
        if inp.facts:
            text += "\nGraph path: " + "; ".join(inp.facts[:4])
        return text


class LLMAnswerer:
    """An LLM answers from the same material. Optional: needs a provider SDK or `chat=`."""

    def __init__(self, model: str = "claude-sonnet-4-5", *, max_tokens: int = 1024,
                 chat: Callable[[str, str, int], str] | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._chat = chat or default_chat(model)

    def answer(self, inp: AnswerInput) -> str:
        user = f"Question: {inp.question}\n\nContext:\n{inp.render()}"
        return self._chat(ANSWER_SYSTEM, user, self.max_tokens)
