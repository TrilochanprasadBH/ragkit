"""Stage 04 reference: a dense index and rank fusion.

Two independent pieces.

**DenseIndex** stores one vector per chunk and answers a query by cosine similarity.
It has no notion of terms, so it cannot prefer a chunk because it contains the word
`CORSMiddleware`. That is both the strength and the weakness: it matches meaning when
the words differ, and it misses exact identifiers that lexical search finds trivially.

**reciprocal_rank_fusion** merges several ranked lists into one. The formula is

    score(document) = sum over lists of 1 / (k + rank_in_that_list)

with `k = 60` in the common configuration.

Why rank fusion rather than score fusion: a BM25 score of 12.4 and a cosine similarity
of 0.71 are not on the same scale and do not mean the same thing. Averaging them is
category error. Ranks are comparable, and the `k` constant softens the advantage of
the very top position so that a document that appears at rank 2 in both lists beats
one that appears at rank 1 in one list and is absent from the other.
"""

from __future__ import annotations

from .embeddings import cosine
from .types import Chunk


class DenseIndex:
    def __init__(self, chunks: list[Chunk], embedder) -> None:
        self.chunks = list(chunks)
        self.embedder = embedder
        # Index the heading path as well: it is context we already have.
        self.vectors = embedder.embed(
            [" ".join([*chunk.headings, chunk.text]) for chunk in self.chunks]
        )

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        if not self.chunks:
            return []
        query_vector = self.embedder.embed([query])[0]
        scored = [
            (cosine(query_vector, vector), chunk.chunk_id)
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(chunk_id, score) for score, chunk_id in scored[:k] if score > 0]


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked lists into one, ordered by summed reciprocal rank."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for position, (chunk_id, _score) in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class HybridIndex:
    """Lexical and dense retrieval, fused by rank.

    `candidates` controls how deep each retriever goes before fusion. Retrieve wide,
    fuse, then cut to `k`: a document ranked 12th by BM25 can still win the fused
    ranking if the dense side ranks it 3rd.

    Both defaults are measured, not inherited. The widely quoted values are
    `candidates = 20, k = 60`; on this corpus those produce 83.3% recall, which is no
    better than the lexical baseline, because `k = 60` flattens the rank differences
    and lets a deep list of mediocre documents accumulate score. `k = 10` keeps the
    top of each list dominant. See the table in the stage 04 lesson.
    """

    def __init__(self, lexical, dense, candidates: int = 10, k: int = 10) -> None:
        self.lexical = lexical
        self.dense = dense
        self.candidates = candidates
        self.k = k

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        lexical_ranking = self.lexical.search(query, k=self.candidates)
        dense_ranking = self.dense.search(query, k=self.candidates)
        fused = reciprocal_rank_fusion([lexical_ranking, dense_ranking], k=self.k)
        return fused[:k]
