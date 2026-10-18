"""Shared fixtures: a tiny hand-written corpus whose graph is known."""

from __future__ import annotations

import pytest

from rag_graph import GraphRAG

#: doc id -> text. Graph: Ann Lee -founded-> Acme Corp <-acquired- Zenith Ltd -makes-> Widget-9;
#: Acme Corp -headquartered in-> Portsea -located in-> Norland.
DOCS = {
    "d-founder": "# Ann Lee\nAnn Lee is an engineer. Ann Lee founded Acme Corp in 2010.",
    "d-deal": "# Zenith acquires Acme\nZenith Ltd acquired Acme Corp in 2020.",
    "d-zenith": "# Zenith Ltd\nZenith Ltd is a retail company headquartered in Brightport. "
                "Zenith Ltd makes Widget-9.",
    "d-acme": "# Acme Corp\nAcme Corp is a software company headquartered in Portsea.",
    "d-city": "# Portsea\nPortsea is a port in Norland.",
    "d-noise": "# Weather\nIt rained all week and the harbour was grey.",
}


@pytest.fixture
def rag(tmp_path):
    g = GraphRAG(db_path=str(tmp_path / "kg.db"))
    yield g
    g.close()


@pytest.fixture
def loaded(rag):
    for doc_id, text in DOCS.items():
        rag.ingest(doc_id=doc_id, text=text)
    return rag
