# Benchmark data (SYNTHETIC)

Everything in this folder is **SYNTHETIC**. The people, companies, products, cities and countries are
invented from syllable tables by `make_data.py` (any resemblance to a real name is coincidence), and
the facts are random. Nothing here is real, nothing comes from any client or employer, and the
corpus says nothing about the real world.

| file | what |
|---|---|
| `corpus.jsonl` | 232 short documents: 60 person pages, 60 company pages, 20 acquisition notices, 12 city pages, 60 product pages, 20 filler notes |
| `questions.jsonl` | 80 questions with `hops` (1, 2 or 3), `kind`, `gold_docs` (the documents that contain the evidence) and `answer` |
| `make_data.py` | the seeded generator; `python data/make_data.py` rewrites both files byte for byte (a test checks this) |

The sentences follow a few fixed templates ("X founded Y in 2011.", "Y is a software company
headquartered in Z.") on purpose, so the offline rule-based extractor can read them. The benchmark
therefore measures **retrieval**, not an LLM's extraction quality: extraction here is near-perfect, and the
ablation in the README shows what happens when it is not.

Each question needs `hops` documents. For a multi-hop question the later documents share no distinctive
word with the question (that is what makes it multi-hop), so a vector-only retriever cannot be expected to
find them. Real corpora are less clean; read the results as a demonstration of the mechanism, not a forecast.
