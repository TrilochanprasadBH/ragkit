"""Stage 08 — the service.

Read `stages/08_service/README.md` first.

You implement one method. Everything below it already exists: retrieval, packing, and
the sufficiency gate. This class composes them, times them separately, records what each
request cost, and writes a line to the usage log.
"""

from __future__ import annotations

import pathlib

from .grounding import Grounder
from .types import Chunk


class Service:
    """A callable documentation assistant."""

    def __init__(
        self,
        chunks: list[Chunk],
        retriever,
        embedder_name: str = "char-ngram",
        embedder_dim: int = 512,
        budget: int = 180,
        top_k: int = 5,
        min_sufficiency: float = 0.5,
        embed_price_per_mtok: float = 0.0,
        usage_path: pathlib.Path | None = None,
    ) -> None:
        """Store the configuration and build the grounder.

        Contract:
          - keep every argument on the instance under the same name
          - build `Grounder(chunks, min_sufficiency=min_sufficiency)` and store it as
            `self.grounder`
        """
        raise NotImplementedError("stage 08: implement Service.__init__")

    def ask(self, question: str, log: bool = True) -> dict:
        """Answer one question.

        Contract:
          - retrieve `top_k` chunks with `self.retriever.search(question, k=top_k)`
          - pack them with `pack(question, docs, budget=self.budget)`
          - answer through `self.grounder` so the sufficiency gate applies
          - time each of the three stages separately with `time.perf_counter`, and
            report milliseconds rounded to two decimals under the keys
            `retrieve_ms`, `pack_ms`, `answer_ms`, `total_ms`
          - cost = len(question.split()) / 1e6 * embed_price_per_mtok
          - return a dict with: question, answer, refused, citations, cited, timings,
            cost_usd, embedded_tokens, pack_tokens
          - `citations` is a list of dicts with `number`, `chunk_id`, `label`
          - when `log` is true, append one JSON line to `usage_path`
          - the log entry contains: at (ISO timestamp), question_sha (first 12 hex
            characters of the SHA-256 of the question), refused, citations, timings,
            cost_usd
          - do nothing when `usage_path` is None
        """
        raise NotImplementedError("stage 08: implement Service.ask")

    def manifest(self) -> dict:
        """Describe the index and the embedder that built it.

        Contract: keys `embedder`, `dimension`, `chunks`, `documents`,
        `budget_words`, `top_k`, `min_sufficiency`, `built_at`.
        """
        raise NotImplementedError("stage 08: implement Service.manifest")

    def write_manifest(self, path: pathlib.Path) -> None:
        """Write the manifest as indented JSON, creating the parent directory."""
        raise NotImplementedError("stage 08: implement Service.write_manifest")
