"""Stage 03 — a field-weighted index.

Read `stages/03_analysis/README.md` first, particularly the section on what did and
did not help.

The formula is stage 02's BM25, with one change: a chunk's terms come from two fields,
and the heading field can be weighted separately from the body field.
"""

from __future__ import annotations

from .types import Chunk


class FieldedIndex:
    """BM25 over two fields: heading path and body."""

    def __init__(
        self,
        chunks: list[Chunk],
        heading_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Index the chunks.

        Contract:
          - tokenise the heading path with `analysis.tokenize`
          - tokenise the body with `analysis.tokenize`
          - store the term counts for each field separately
          - record document frequency over the union of both fields
          - weighted length = heading_weight * heading terms + body terms
          - weighted term frequency = heading_weight * heading count + body count
        """
        raise NotImplementedError("stage 03: implement FieldedIndex.__init__")

    def weighted_tf(self, chunk_id: str, term: str) -> float:
        """heading_weight * count in headings + count in body."""
        raise NotImplementedError("stage 03: implement weighted_tf")

    def idf(self, term: str) -> float:
        """log(1 + (N - df + 0.5) / (df + 0.5)), the same form as stage 02."""
        raise NotImplementedError("stage 03: implement idf")

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to k (chunk_id, score) pairs, highest first, ties by chunk_id."""
        raise NotImplementedError("stage 03: implement search")


def build_fielded_index(
    chunks: list[Chunk],
    heading_weight: float = 1.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> FieldedIndex:
    raise NotImplementedError("stage 03: implement build_fielded_index")
