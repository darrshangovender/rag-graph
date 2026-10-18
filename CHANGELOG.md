# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Breaking
- `GraphRAG.ingest` returns an `IngestReport` (`int(report)` is the chunk count, the old return value).
  Re-ingesting a `doc_id` now replaces the earlier version instead of duplicating it.
- Constructor defaults are offline: `HashingEmbedder`, `RuleBasedExtractor`, `ExtractiveAnswerer`. Pass
  `OpenAIEmbedder()`, `extractor_model=` / `answer_model=` (or your own objects) for real models.
- The entity resolver no longer merges reordered names by default ("AI Bias" and "Bias AI" are distinct);
  `sort_tokens=True` restores it. "Open AI" and "OpenAI" now merge, which the 0.1.0 docs claimed but the
  code did not do. `numpy` is a new dependency.
- `cosine_similarity` and the store raise on a dimension mismatch instead of returning 0.0.

### Added
- Offline pipeline: `HashingEmbedder`, `RuleBasedExtractor` (regex patterns), `ExtractiveAnswerer`;
  `LLMAnswerer` and `EntityRelationExtractor(chat=...)` take an injectable transport.
- A bundled SYNTHETIC multi-hop benchmark (`data/`, `benchmarks/multihop.py`): 232 documents, 80 questions,
  vector-only vs graph retrieval by hops, question type, `alpha`, traversal depth and extraction recall.
- The retrieved graph facts (relation, endpoints, evidence, source chunk) are passed to the answerer.
  `AnswerResult` gained `graph_facts`, `cited`, `invalid_citations`; each source has a `cite` label.
- Bounded traversal (`max_entities`, `max_degree`, `max_facts`), `alpha` per call, optional
  `traversal="relation_guided"`.
- Atomic ingest per document, `IngestReport` (entities, relations, dropped triples, extraction errors).
- 216 tests (was 9).

### Fixed
- **Incremental ingest corrupted the graph**: the resolver restarted at id 1 in every process, so ingesting
  into an existing `kg.db` reissued ids that belonged to other entities and overwrote them while their edges
  still pointed at them. The resolver is now rebuilt from the store.
- **The graph half of retrieval was dead in a fresh process** (the surface index came from the in-memory
  resolver): a reopened database had no seeds and silently degraded to vector search.
- Source labels `[cN]` in the context were positional while sources carried database ids; sources now carry
  the same `cite` labels and answers are checked for markers that do not exist.
- Seeds were found by substring ("ann" matched inside "annual"); they are now whole-word and alias-aware.
- Chunker: blank input produced one empty chunk; headings inside code fences were split on; "# C#" lost its
  last character; a single oversize paragraph or word was never split; the breadcrumb was not embedded, so a
  page titled by a name that its body never repeats could not be found by that name.
- Unicode punctuation (em dash, curly quotes) was not stripped when normalising names.
- A failed ingest could leave half a document, and a failed extraction was swallowed by a bare `except`;
  both are now handled and reported.
- `_parse_json` used a greedy brace-to-brace regex; it now takes the first JSON object.
- **Two `GraphRAG` objects on one file overwrote each other's entities**: each built its resolver once, at
  construction, so both issued the same next entity id (the restart bug again, with two live writers), and a
  long-lived reader never saw entities added later. The resolver is now rebuilt when `PRAGMA data_version`
  shows another connection has written, inside the write lock for `ingest` and before every `retrieve`.
- An entity name with no letter or digit (a model returned "—") aborted and rolled back the whole document;
  it is now skipped and counted in `IngestReport.dropped_entities`.
- The rule-based extractor took time cubic in the length of a run of capitalised words (4,000 words: 121 s)
  and quadratic in a long hyphenated token (72 s); it also runs on the user's question. Names, lists and
  clause gaps are now length-bounded (name: 8 words of 60 characters; list: 30 names; gap: 120 characters).
- Edge duplicates, per-write commits and a shared unguarded connection: edges are unique per source chunk,
  writes are transactional, the connection is locked, WAL with a busy timeout.
- Lint: `ruff check .` is a required CI step with an explicit rule set.
- `.gitignore` excluded `data/make_data.py` (only data formats were re-included under `data/`), so a
  fresh clone failed at test collection and `make data` had nothing to run. A test now checks that no
  shipped file is ignored.

## [0.1.0] - 2026-10-09

### Added
- Heading-aware chunker, LLM entity and relation extraction, SQLite graph store, hybrid vector + graph
  retriever, `GraphRAG.ingest` / `.ask`.
