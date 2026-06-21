"""LLM-based entity + relation extraction.

Given a text chunk, return:
  - Entities: typed (Person, Org, Concept, Place, Date, Other)
  - Relations: (subject, predicate, object) triples grounded in the text

Pydantic schema validation gates the LLM output. Anything that does not match
the schema raises — better to fail loudly than silently insert junk into the
knowledge graph.
"""

from __future__ import annotations

import json
import os
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


EntityType = Literal["Person", "Organisation", "Place", "Concept", "Date", "Event", "Product", "Other"]


class Entity(BaseModel):
    name: str
    type: EntityType


class Triple(BaseModel):
    subject: str         # entity name as it appears in `entities`
    predicate: str       # short verb phrase, e.g. "founded", "acquired", "works for"
    object: str          # entity name as it appears in `entities`
    evidence: str = ""   # short quote from the chunk supporting this triple


class Extraction(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    triples: list[Triple] = Field(default_factory=list)


SYSTEM = """\
You extract a knowledge graph from a document chunk.

Output strict JSON matching this schema:
{
  "entities": [{"name": "...", "type": "Person|Organisation|Place|Concept|Date|Event|Product|Other"}],
  "triples":  [{"subject": "...", "predicate": "...", "object": "...", "evidence": "short quote"}]
}

Rules:
  - Entity names appear verbatim as they do in the chunk (don't normalise).
  - Predicates are short verb phrases: "founded", "acquired", "works for", "located in".
  - Every triple's subject and object must appear in `entities`.
  - Every triple's evidence is a short literal quote from the chunk (<= 25 words).
  - Skip facts you are not confident about. Quality > quantity.
"""


class EntityRelationExtractor:
    """Provider-portable extractor. Default model is Sonnet-4-5 (cheap + reliable)."""

    def __init__(self, model: str = "claude-sonnet-4-5", *, max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self.provider = "anthropic" if model.startswith("claude-") else "openai"

    def extract(self, chunk_text: str) -> Extraction:
        raw = self._call_llm(chunk_text)
        data = _parse_json(raw)
        try:
            return Extraction(**data)
        except ValidationError as e:
            raise ValueError(f"Extractor returned invalid schema: {e}") from e

    def _call_llm(self, chunk_text: str) -> str:
        if self.provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            resp = client.messages.create(
                model=self.model, max_tokens=self.max_tokens,
                system=SYSTEM, messages=[{"role": "user", "content": chunk_text}],
            )
            return resp.content[0].text
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": chunk_text}],
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        raise ValueError(f"no JSON in extractor output: {text[:200]!r}")
    return json.loads(m.group(0))