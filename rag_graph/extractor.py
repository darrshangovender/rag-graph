"""Entity and relation extraction.

Two extractors share one interface, `extract(text) -> Extraction`:

  * `RuleBasedExtractor` - deterministic regex patterns over simple declarative sentences
    ("A founded B", "B was founded by A, C and D", "A acquired B", ...). No model, no key, no
    network: this is what the tests, the demo and the benchmark use. It is precise on text
    that follows its patterns and blind to everything else.
  * `EntityRelationExtractor` - an LLM in strict-JSON mode (Anthropic or OpenAI), validated with
    Pydantic. Pass `chat=` to supply your own transport (or a fake in tests).

Anything that does not match the schema raises: better to fail loudly than to insert junk into
the graph.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator

from .chunker import split_sentences

EntityType = Literal[
    "Person", "Organisation", "Place", "Concept", "Date", "Event", "Product", "Other",
]


class Entity(BaseModel):
    name: str
    type: EntityType

    @field_validator("name")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("entity name is blank")
        return v


class Triple(BaseModel):
    subject: str         # entity name as it appears in `entities`
    predicate: str       # short verb phrase, e.g. "founded", "acquired", "works for"
    object: str          # entity name as it appears in `entities`
    evidence: str = ""   # short quote from the chunk supporting this triple

    @field_validator("subject", "predicate", "object")
    @classmethod
    def _non_blank(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("triple field is blank")
        return v


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)


class Extractor(Protocol):
    def extract(self, text: str) -> Extraction: ...


# ---- rule-based -----------------------------------------------------------------------------

# Every repeat is bounded. With unbounded repeats a run of capitalised words, or one 10,000
# character hyphenated token, took time cubic/quadratic in its length (4,000 capitalised words:
# 121 s; 5,000 "A-A-A..." characters: 72 s), and the extractor also runs on the user's question.
# A name is at most 8 capitalised words of at most 60 characters; a list at most 30 names.
_WORD = r"[A-Z][\w'’\-]{0,59}"
_NAME = rf"{_WORD}(?:[ ]{_WORD}){{0,7}}"
_LIST = rf"{_NAME}(?:,[ ]{_NAME}){{0,29}},?[ ]and[ ]{_NAME}|{_NAME}"
# Room for a clause between "is" and the verb ("Ann is a quiet engineer who founded ...").
_GAP = r"[^.\n]{0,120}?"


@dataclass(frozen=True)
class Pattern:
    """`regex` has named groups `s` (subject, may be a list) and `o` (object)."""
    regex: str
    predicate: str
    subject_type: EntityType = "Other"
    object_type: EntityType = "Other"


DEFAULT_PATTERNS: tuple[Pattern, ...] = (
    Pattern(rf"(?P<s>{_LIST})[ ](?:co-?)?founded[ ](?P<o>{_NAME})", "founded", "Person",
            "Organisation"),
    Pattern(rf"(?P<s>{_NAME})[ ]is[ ]{_GAP}\bwho[ ](?:co-?)?founded[ ](?P<o>{_NAME})",
            "founded", "Person", "Organisation"),
    Pattern(rf"(?P<o>{_NAME})[ ]was[ ](?:co-?)?founded[ ]by[ ](?P<s>{_LIST})",
            "founded", "Person", "Organisation"),
    Pattern(rf"(?P<s>{_NAME})[ ]acquired[ ](?P<o>{_NAME})", "acquired", "Organisation",
            "Organisation"),
    Pattern(rf"(?P<o>{_NAME})[ ]was[ ]acquired[ ]by[ ](?P<s>{_NAME})", "acquired",
            "Organisation", "Organisation"),
    Pattern(rf"(?P<s>{_NAME})[ ]is[ ](?:a[ ]{_GAP}[ ])?headquartered[ ]in[ ](?P<o>{_NAME})",
            "headquartered in", "Organisation", "Place"),
    Pattern(rf"(?P<s>{_NAME})[ ]makes[ ](?:the[ ]|a[ ]|an[ ])?(?:product[ ])?(?P<o>{_NAME})",
            "makes", "Organisation", "Product"),
    Pattern(rf"(?P<s>{_NAME})[ ]is[ ]a[ ](?:[a-z\-]+[ ])?(?:city|town|port|village)[ ]in[ ]"
            rf"(?P<o>{_NAME})", "located in", "Place", "Place"),
    Pattern(rf"(?P<s>{_NAME})[ ]is[ ]the[ ](?:CEO|chief executive)[ ]of[ ](?P<o>{_NAME})",
            "leads", "Person", "Organisation"),
    Pattern(rf"(?P<s>{_NAME})[ ]joined[ ](?P<o>{_NAME})[ ]as[ ]CEO\b", "leads", "Person",
            "Organisation"),
)



def _split_names(group: str) -> list[str]:
    parts = re.split(r",\s*(?:and\s+)?|\s+and\s+", group)
    return [p.strip() for p in parts if p.strip()]


class RuleBasedExtractor:
    """Deterministic pattern extractor. See the module docstring for what it can and cannot see."""

    def __init__(self, patterns: tuple[Pattern, ...] = DEFAULT_PATTERNS) -> None:
        self._patterns = [(re.compile(p.regex), p) for p in patterns]

    def extract(self, text: str) -> Extraction:
        entities: dict[str, EntityType] = {}
        triples: list[Triple] = []
        seen: set[tuple[str, str, str]] = set()
        for sentence in split_sentences(text):
            for rx, pat in self._patterns:
                for m in rx.finditer(sentence):
                    for subj in _split_names(m.group("s")):
                        obj = m.group("o").strip()
                        if subj == obj:
                            continue
                        entities.setdefault(subj, pat.subject_type)
                        entities.setdefault(obj, pat.object_type)
                        key = (subj, pat.predicate, obj)
                        if key not in seen:
                            seen.add(key)
                            triples.append(Triple(
                                subject=subj, predicate=pat.predicate, object=obj,
                                evidence=" ".join(sentence.split()[:25]),
                            ))
        return Extraction(
            entities=[Entity(name=n, type=t) for n, t in entities.items()], triples=triples,
        )


# ---- LLM -----------------------------------------------------------------------------------------

SYSTEM = """\
You extract a knowledge graph from a document chunk.

Output strict JSON matching this schema:
{
  "entities": [{"name": "...",
                "type": "Person|Organisation|Place|Concept|Date|Event|Product|Other"}],
  "triples":  [{"subject": "...", "predicate": "...", "object": "...", "evidence": "short quote"}]
}

Rules:
  - Entity names appear verbatim as they do in the chunk (don't normalise).
  - Predicates are short verb phrases: "founded", "acquired", "works for", "located in".
  - Every triple's subject and object must appear in `entities`.
  - Every triple's evidence is a short literal quote from the chunk (<= 25 words).
  - Skip facts you are not confident about. Quality > quantity.
"""

#: chat(system, user, max_tokens) -> the model's text reply
ChatFn = Callable[[str, str, int], str]


def default_chat(model: str) -> ChatFn:
    """An SDK-backed `chat` for `model` (claude-* -> Anthropic, otherwise OpenAI). Built lazily,
    and the client is created once and reused."""
    state: dict[str, object] = {}

    def chat(system: str, user: str, max_tokens: int) -> str:
        if model.startswith("claude-"):
            if "client" not in state:
                from anthropic import Anthropic
                state["client"] = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = state["client"].messages.create(  # type: ignore[attr-defined]
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(getattr(b, "text", "") for b in resp.content)
        if "client" not in state:
            from openai import OpenAI
            state["client"] = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = state["client"].chat.completions.create(  # type: ignore[attr-defined]
            model=model, max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""

    return chat


class EntityRelationExtractor:
    """LLM extractor: strict JSON, Pydantic-validated.

    Optional: needs the anthropic or openai extra, or your own `chat`."""

    def __init__(self, model: str = "claude-sonnet-4-5", *, max_tokens: int = 1024,
                 chat: ChatFn | None = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.provider = "anthropic" if model.startswith("claude-") else "openai"
        self._chat = chat or default_chat(model)

    def extract(self, chunk_text: str) -> Extraction:
        raw = self._chat(SYSTEM, chunk_text, self.max_tokens)
        data = _parse_json(raw)
        try:
            return Extraction(**data)
        except ValidationError as e:
            raise ValueError(f"Extractor returned invalid schema: {e}") from e


def _parse_json(text: str) -> dict:
    """The first JSON object in `text`. (A greedy brace-to-brace regex used to swallow trailing
    prose containing another `}` and then fail to parse a perfectly good leading object.)"""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, RecursionError):
        pass
    decoder = json.JSONDecoder()
    idx, tries = text.find("{"), 0
    while idx != -1 and tries < 200:
        try:
            obj, _ = decoder.raw_decode(text, idx)
        except (json.JSONDecodeError, RecursionError):
            idx, tries = text.find("{", idx + 1), tries + 1
            continue
        if isinstance(obj, dict):
            return obj
        idx = text.find("{", idx + 1)
    raise ValueError(f"no JSON in extractor output: {text[:200]!r}")
