"""Entity resolver — alias merging via surface-form normalisation.

The single biggest quality lift on a knowledge-graph RAG system is NOT
fancier extraction; it's making sure "OpenAI", "Open AI", and "openai" all
map to one node.

Strategy:
  1. Normalise: lowercase, strip punctuation, sort tokens (for things like
     "John Smith" vs "Smith, John").
  2. Alias lookup: if the normalised form matches an existing entity, attach
     as alias. Else, create a new entity.

For trickier cases (two unrelated "Apple"s), an LLM-based disambiguation hook
can be plugged in here. Default is the cheap deterministic path.
"""

from __future__ import annotations

import re
import string


_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")


def normalize_surface(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, sort tokens.

    Sorting tokens is the subtle one — it makes "John Smith" and "Smith John"
    collide, which is usually what you want for person names but NOT what you
    want for "AI Bias" vs "Bias AI". Tune per-domain by passing sort_tokens=False
    when the order is semantically significant.
    """
    s = name.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = " ".join(s.split())
    tokens = sorted(s.split())
    return " ".join(tokens)


class EntityResolver:
    """Holds the canonical name -> id mapping, plus the alias index."""

    def __init__(self) -> None:
        self._normalized_to_id: dict[str, int] = {}
        self._id_to_canonical: dict[int, str] = {}
        self._id_to_aliases: dict[int, set[str]] = {}
        self._next_id: int = 1

    def resolve(self, surface_name: str, *, entity_type: str | None = None) -> tuple[int, bool]:
        """Return (entity_id, is_new). Attach as alias if existing canonical matches."""
        norm = normalize_surface(surface_name)
        if norm in self._normalized_to_id:
            eid = self._normalized_to_id[norm]
            self._id_to_aliases[eid].add(surface_name)
            return eid, False

        eid = self._next_id
        self._next_id += 1
        self._normalized_to_id[norm] = eid
        self._id_to_canonical[eid] = surface_name
        self._id_to_aliases[eid] = {surface_name}
        return eid, True

    def canonical(self, eid: int) -> str:
        return self._id_to_canonical[eid]

    def aliases(self, eid: int) -> set[str]:
        return set(self._id_to_aliases.get(eid, ()))