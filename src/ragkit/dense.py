"""Stage 04 — a dense index and rank fusion.

Read `stages/04_dense/README.md` first. Three ideas, in order: vectors for similarity,
rank fusion for combining retrievers, and measurement for deciding whether any of it
helped.
"""

from __future__ import annotations

from .types import Chunk


class DenseIndex:
    """One vector per chunk; queries score by cosine similarity."""

    def __init__(self, chunks: list[Chunk], embedder) -> None:
        """Embed each chunk, including its heading path.

        Contract:
          - the text embedded is `" ".join(chunk.headings) + " " + chunk.text`
          - vectors are stored in the same order as `chunks`
        """
        raise NotImplementedError("stage 04: implement DenseIndex.__init__")

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs, highest score first.

        Contract:
          - scores are cosine similarities
          - chunks scoring 0 or less are excluded
          - ties are broken by chunk_id
        """
        raise NotImplementedError("stage 04: implement DenseIndex.search")


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists: score(doc) = sum over lists of 1 / (k + rank).

    Contract:
      - `rank` starts at 1
      - a document absent from a list contributes nothing from that list
      - the result is sorted by score descending, ties broken by chunk_id
      - input scores are ignored entirely; only positions matter
    """
    raise NotImplementedError("stage 04: implement reciprocal_rank_fusion")


class HybridIndex:
    """Lexical and dense retrieval, fused by rank.

    Defaults: `candidates = 10`, `k = 10`. See the stage lesson for the measurements
    behind those numbers; the commonly quoted defaults do worse on this corpus.
    """

    def __init__(self, lexical, dense, candidates: int = 10, k: int = 10) -> None:
        self.lexical = lexical
        self.dense = dense
        self.candidates = candidates
        self.k = k

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Retrieve `candidates` from each side, fuse, return the top k."""
        raise NotImplementedError("stage 04: implement HybridIndex.search")
