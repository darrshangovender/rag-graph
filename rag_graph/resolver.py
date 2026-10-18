"""Entity resolver: map surface forms to entity ids.

Strategy (deterministic, no model):
  1. Normalise the surface form to a key: case-fold, replace punctuation with spaces,
     collapse whitespace. "Open-AI!", "open ai" and "OPEN AI" share a key.
  2. By default the key then drops the spaces ("openai"), so "OpenAI" and "Open AI" merge.
     Token ORDER is kept: "AI Bias" and "Bias AI" are different things.
  3. `sort_tokens=True` additionally ignores token order, which suits person names written
     "Smith John" / "John Smith" but falsely merges "Bank of America" and "America of Bank".
     It is off by default.

The resolver's state is rebuilt from the store with `EntityResolver.from_store`. It used to live
only in memory and restart at id 1 in every process, so ingesting into an existing database reissued
ids that already belonged to other entities and silently corrupted the graph.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

# Letters and digits in any script; everything else separates words.
_WORD_RE = re.compile(r"[^\W_]+")


def normalize_surface(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace, SORT tokens (the 0.1.0 key).

    Kept for backwards compatibility. Sorting tokens makes "John Smith" and "Smith John" collide,
    which is wanted for person names and wrong for "AI Bias" vs "Bias AI"; the resolver no longer
    uses this by default (see `normalize_key`).
    """
    return " ".join(sorted(tokens_of(name)))


def tokens_of(name: str) -> tuple[str, ...]:
    """Case-folded word tokens of a surface form, punctuation removed, order kept."""
    return tuple(_WORD_RE.findall(name.casefold()))


def normalize_key(name: str, *, sort_tokens: bool = False) -> str:
    """The key under which a surface form is looked up. Empty string for a name with no word."""
    toks = tokens_of(name)
    if sort_tokens:
        toks = tuple(sorted(toks))
    return "".join(toks)


class _EntityRows(Protocol):
    def all_entities(self) -> list[dict]: ...


class EntityResolver:
    """Canonical name -> id mapping plus the alias index."""

    def __init__(self, *, sort_tokens: bool = False) -> None:
        self.sort_tokens = sort_tokens
        self._key_to_id: dict[str, int] = {}
        self._id_to_canonical: dict[int, str] = {}
        self._id_to_aliases: dict[int, set[str]] = {}
        self._next_id: int = 1
        self.version = 0          # bumped whenever a surface form or entity is added

    @classmethod
    def from_store(cls, store: _EntityRows, *, sort_tokens: bool = False) -> EntityResolver:
        """Rebuild the resolver from persisted entities so ids continue where they left off."""
        r = cls(sort_tokens=sort_tokens)
        for row in store.all_entities():
            eid, canonical = int(row["id"]), row["canonical"]
            r._id_to_canonical[eid] = canonical
            r._id_to_aliases[eid] = set(row["aliases"]) | {canonical}
            for surface in r._id_to_aliases[eid]:
                key = r._key(surface)
                if key:
                    r._key_to_id.setdefault(key, eid)
            r._next_id = max(r._next_id, eid + 1)
        return r

    def _key(self, name: str) -> str:
        return normalize_key(name, sort_tokens=self.sort_tokens)

    def resolve(self, surface_name: str, *, entity_type: str | None = None) -> tuple[int, bool]:
        """Return (entity_id, is_new). A known surface form is attached as an alias."""
        key = self._key(surface_name)
        if not key:
            raise ValueError(f"entity name has no word characters: {surface_name!r}")
        if key in self._key_to_id:
            eid = self._key_to_id[key]
            if surface_name not in self._id_to_aliases[eid]:
                self._id_to_aliases[eid].add(surface_name)
                self.version += 1
            return eid, False
        eid = self._next_id
        self._next_id += 1
        self._key_to_id[key] = eid
        self._id_to_canonical[eid] = surface_name
        self._id_to_aliases[eid] = {surface_name}
        self.version += 1
        return eid, True

    def lookup(self, surface_name: str) -> int | None:
        """The id for a surface form WITHOUT creating it."""
        return self._key_to_id.get(self._key(surface_name))

    def canonical(self, eid: int) -> str:
        return self._id_to_canonical[eid]

    def aliases(self, eid: int) -> set[str]:
        return set(self._id_to_aliases.get(eid, ()))

    def entity_ids(self) -> list[int]:
        return sorted(self._id_to_canonical)

    def surface_forms(self) -> Iterable[tuple[str, int]]:
        """Every known surface form with its entity id (canonical and aliases)."""
        for eid, forms in self._id_to_aliases.items():
            for form in forms:
                yield form, eid

    def __len__(self) -> int:
        return len(self._id_to_canonical)
