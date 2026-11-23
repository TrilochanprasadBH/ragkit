"""Stage 08 reference: the service.

Everything the previous seven stages built, wrapped in something you can call.

Three things this stage adds that the earlier ones did not have:

**Per-stage timing.** "The request took 120ms" is useless. Knowing that 90ms of it was
embedding and 4ms was retrieval tells you what to fix. Each stage is timed separately
and reported together.

**Cost per request.** The corpus is embedded once; every query costs one query
embedding. That is a fraction of a cent at current prices, and it is still worth
recording, because the number that kills a product is not the cost of one request but
the cost of a million and nobody noticing.

**A manifest.** The index is only valid for the embedder that built it. Writing the
embedder name, its dimension, and the chunk count into a file next to the index means a
mismatch becomes a startup error instead of a silent quality collapse.

The design keeps the HTTP layer separate from the service. `Service.ask()` knows nothing
about requests, so it can be called from a test, a batch job, or any web framework.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import time
from dataclasses import dataclass, field

from .answer import refuse
from .grounding import Grounder
from .pack import pack
from .types import Chunk


@dataclass
class Service:
    chunks: list[Chunk]
    retriever: object            # anything with .search(query, k)
    embedder_name: str = "char-ngram"
    embedder_dim: int = 512
    budget: int = 180
    top_k: int = 5
    min_sufficiency: float = 0.5
    embed_price_per_mtok: float = 0.0
    usage_path: pathlib.Path | None = None

    grounder: Grounder = field(init=False)

    def __post_init__(self) -> None:
        self.grounder = Grounder(self.chunks, min_sufficiency=self.min_sufficiency)

    # -- the request path --------------------------------------------------

    def ask(self, question: str, log: bool = True) -> dict:
        """Answer one question, with timings, citations, and cost."""
        started = time.perf_counter()

        retrieved_at = time.perf_counter()
        hits = self.retriever.search(question, k=self.top_k)
        by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        docs = [by_id[chunk_id] for chunk_id, _ in hits if chunk_id in by_id]
        retrieved = time.perf_counter()

        packed = pack(question, docs, budget=self.budget)
        packed_at = time.perf_counter()

        result = self.grounder.answer(question, packed.text)
        answered_at = time.perf_counter()

        tokens = len(question.split())
        cost = tokens / 1_000_000 * self.embed_price_per_mtok

        timings = {
            "retrieve_ms": round((retrieved - retrieved_at) * 1000, 2),
            "pack_ms": round((packed_at - retrieved) * 1000, 2),
            "answer_ms": round((answered_at - packed_at) * 1000, 2),
            "total_ms": round((answered_at - started) * 1000, 2),
        }

        response = {
            "question": question,
            "answer": result.text,
            "refused": not result.grounded,
            "citations": [
                {"number": c.number, "chunk_id": c.chunk_id, "label": c.label}
                for c in packed.citations
            ],
            "cited": result.citation,
            "timings": timings,
            "cost_usd": round(cost, 8),
            "embedded_tokens": tokens,
            "pack_tokens": packed.tokens,
        }

        if log:
            self._log(response)
        return response

    # -- supporting material ----------------------------------------------

    def _log(self, response: dict) -> None:
        if self.usage_path is None:
            return
        path = pathlib.Path(self.usage_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "question_sha": hashlib.sha256(response["question"].encode()).hexdigest()[:12],
            "refused": response["refused"],
            "citations": len(response["citations"]),
            "timings": response["timings"],
            "cost_usd": response["cost_usd"],
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def manifest(self) -> dict:
        """What this index is, and what built it."""
        return {
            "embedder": self.embedder_name,
            "dimension": self.embedder_dim,
            "chunks": len(self.chunks),
            "documents": len({chunk.doc_id for chunk in self.chunks}),
            "budget_words": self.budget,
            "top_k": self.top_k,
            "min_sufficiency": self.min_sufficiency,
            "built_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        }

    def write_manifest(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.manifest(), indent=2) + "\n", encoding="utf-8")
