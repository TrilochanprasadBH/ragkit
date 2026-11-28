"""Stage 09 — reranking, and when it is worth its cost.

Read `stages/09_rerank/README.md` first. Two ideas: score the query and the chunk *together*,
and find out whether the first stage left you anything to fix.
"""

from __future__ import annotations

from .types import Chunk


class TermStats:
    """Inverse document frequency over the indexed chunks."""

    def __init__(self, idf: dict[str, float], n_chunks: int) -> None:
        self.idf = idf
        self.n_chunks = n_chunks

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "TermStats":
        """Document frequency per term, then

            idf(term) = log(1 + (N - df + 0.5) / (df + 0.5))

        Contract:
          - a term is counted once per chunk, however often it appears
          - terms absent from the corpus get a weight of 1.0 from `weight`
        """
        raise NotImplementedError("stage 09: implement TermStats.from_chunks")

    def weight(self, term: str) -> float:
        """The IDF of `term`, or 1.0 when the corpus never saw it."""
        raise NotImplementedError("stage 09: implement TermStats.weight")


class CrossEncoder:
    """Scores a (query, chunk) pair with four lexical signals. Deterministic, offline."""

    def __init__(self, chunks: list[Chunk], weights: dict[str, float] | None = None) -> None:
        """Store chunks by id and compute their term statistics once.

        Contract:
          - `weights` defaults to `DEFAULT_WEIGHTS` below
          - `self.by_id[chunk_id]` resolves every chunk, for the reranker
        """
        raise NotImplementedError("stage 09: implement CrossEncoder.__init__")

    def score(self, query: str, chunk: Chunk) -> float:
        """A number in [0, 1]; higher means the chunk answers the query.

        Four signals, each in [0, 1], combined by the weights:

          coverage    share of the query's terms present in the chunk, weighted by IDF
          phrase      share of the query's adjacent pairs present adjacently in the chunk
          proximity   1 / (1 + span / 10), where span is the narrowest window containing
                      at least two distinct matched terms
          heading     share of the query's terms present in the chunk's heading path

        Contract:
          - an empty query scores 0.0
          - the same (query, chunk) always scores the same value
          - the score does not depend on which other chunks were passed to the constructor,
            except through `TermStats`
        """
        raise NotImplementedError("stage 09: implement CrossEncoder.score")


class RerankIndex:
    """Wrap a first stage: retrieve wide, rescore the pairs, blend, cut to k.

    See the stage lesson for the measurements behind `weight = 0.6` and `depth = 20`.
    """

    def __init__(
        self,
        first_stage,
        chunks: list[Chunk],
        weight: float = 0.6,
        depth: int = 20,
        encoder: CrossEncoder | None = None,
    ) -> None:
        """Contract:
          - `weight` outside [0, 1] raises ValueError
          - `depth` below 1 raises ValueError
          - `encoder` defaults to a CrossEncoder over `chunks`
        """
        raise NotImplementedError("stage 09: implement RerankIndex.__init__")

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Re-rank the first stage's top `depth` and return up to k pairs.

        Contract:
          - every returned chunk came from `first_stage.search(query, k=depth)`; the
            reranker can reorder, it cannot invent
          - pair scores are min-max normalised across the candidate set; when every
            candidate scores the same, the normalised value is 0 for all of them
          - the blended score is `weight * normalised + (1 - weight) * 1 / rank`,
            with rank starting at 1
          - sorted by blended score descending, ties broken by chunk_id
          - `weight = 0.0` returns exactly the first stage's order
        """
        raise NotImplementedError("stage 09: implement RerankIndex.search")


DEFAULT_WEIGHTS = {
    "coverage": 0.35,
    "phrase": 0.30,
    "proximity": 0.20,
    "heading": 0.15,
}
