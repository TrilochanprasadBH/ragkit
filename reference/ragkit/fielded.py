"""Stage 03 reference: a field-weighted BM25 index (BM25F, simplified).

Stage 02 built one bag of words per chunk. But a chunk has two kinds of text: the
heading path, which says what the chunk is *about*, and the body, which says what it
*contains*. They are not equally informative.

This index scores them separately:

    weighted_tf(term) = heading_weight * tf_in_headings + tf_in_body
    weighted_len      = heading_weight * heading_len + body_len

Everything else is the stage 02 formula. With `heading_weight = 1.0` it degenerates
to plain BM25 over the concatenation, so the stage 02 numbers are a special case of
this one. That property is worth keeping: it means the comparison in the lesson is
between two settings of one parameter, not two different programs.

This is a simplified BM25F. The full version normalises each field by its own average
length. For two fields and a corpus this size, the extra machinery does not pay for
itself, and the simplification is easier to reason about.

An honest note on the default. `heading_weight = 1.0` means headings are indexed but
not boosted, which is plain BM25 over the concatenation. Raising it to 2.0 was the
obvious guess and it did **not** improve recall on this corpus; it moved MRR by about
0.01, which is noise. The default stays at 1.0 so that the number reported in the
lesson is not inflated by a knob that did nothing. The parameter exists because you
should be able to test the claim yourself.
"""

from __future__ import annotations

import math
from collections import Counter

from . import analysis
from .types import Chunk


class FieldedIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        heading_weight: float = 1.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = list(chunks)
        self.heading_weight = heading_weight
        self.k1 = k1
        self.b = b

        self.heading_tf: dict[str, Counter] = {}
        self.body_tf: dict[str, Counter] = {}
        self.lengths: dict[str, float] = {}
        self.df: Counter = Counter()

        for chunk in self.chunks:
            heading_terms = analysis.tokenize(" ".join(chunk.headings))
            body_terms = analysis.tokenize(chunk.text)

            heading_counts = Counter(heading_terms)
            body_counts = Counter(body_terms)

            self.heading_tf[chunk.chunk_id] = heading_counts
            self.body_tf[chunk.chunk_id] = body_counts
            self.lengths[chunk.chunk_id] = (
                heading_weight * len(heading_terms) + len(body_terms)
            )

            for term in set(heading_terms) | set(body_terms):
                self.df[term] += 1

        self.n = len(self.chunks)
        self.avg_length = sum(self.lengths.values()) / self.n if self.n else 0.0

    def weighted_tf(self, chunk_id: str, term: str) -> float:
        return (
            self.heading_weight * self.heading_tf[chunk_id].get(term, 0)
            + self.body_tf[chunk_id].get(term, 0)
        )

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        query_terms = analysis.tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = {}

        for term in set(query_terms):
            idf = self.idf(term)
            if idf <= 0:
                continue
            for chunk in self.chunks:
                tf = self.weighted_tf(chunk.chunk_id, term)
                if not tf:
                    continue
                length = self.lengths[chunk.chunk_id]
                normaliser = 1 - self.b + self.b * (
                    length / self.avg_length if self.avg_length else 0
                )
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + idf * (
                    tf * (self.k1 + 1) / (tf + self.k1 * normaliser)
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]


def build_fielded_index(
    chunks: list[Chunk],
    heading_weight: float = 1.0,
    k1: float = 1.5,
    b: float = 0.75,
) -> FieldedIndex:
    return FieldedIndex(chunks, heading_weight=heading_weight, k1=k1, b=b)
