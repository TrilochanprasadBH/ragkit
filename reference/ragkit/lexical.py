"""Reference implementation for stage 02: BM25.

Three decisions are worth understanding before reading the code.

**Index the heading path, not only the body.** A chunk from the section "Defaults"
says nothing about what it is about. The breadcrumb ("Query Parameters > Defaults")
supplies the vocabulary that a question would use. This is the cheapest form of the
"contextual chunk" idea that production systems use, and it costs nothing because
stage 01 already captured the headings.

**Use the BM25 idf, not the textbook idf.** `log((N - df + 0.5) / (df + 0.5) + 1)`
is bounded below by zero for very common terms, which stops them from contributing
negative scores. The older `log(N / df)` can go negative and makes long queries
unstable.

**Normalise by length.** Without the `b` term, a long chunk wins almost any query
simply because it contains more words. The length normalisation is what stops the
index from preferring verbose passages.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .types import Chunk

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in TOKEN.findall(text.casefold())
        if token not in STOP_WORDS
    ]


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b

        self.terms: dict[str, list[str]] = {}
        self.term_counts: dict[str, Counter] = {}
        self.lengths: dict[str, int] = {}
        self.df: Counter = Counter()

        for chunk in self.chunks:
            # The heading path is part of the searchable text on purpose.
            searchable = " ".join([*chunk.headings, chunk.text])
            tokens = tokenize(searchable)
            counts = Counter(tokens)

            self.terms[chunk.chunk_id] = tokens
            self.term_counts[chunk.chunk_id] = counts
            self.lengths[chunk.chunk_id] = len(tokens)

            for term in counts:
                self.df[term] += 1

        self.n = len(self.chunks)
        self.avg_length = (
            sum(self.lengths.values()) / self.n if self.n else 0.0
        )

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: dict[str, float] = {}

        for term in set(query_terms):
            idf = self.idf(term)
            if idf <= 0:
                continue
            for chunk_id, counts in self.term_counts.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                length = self.lengths[chunk_id]
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (length / self.avg_length if self.avg_length else 0)
                )
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * (
                    tf * (self.k1 + 1) / denominator
                )

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return ranked[:k]


def build_index(chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    return BM25Index(chunks, k1=k1, b=b)


STOP_WORDS = frozenset(
    """
    a an and are as at be but by for if in into is it its of on or such that the
    their then there these they this to was will with you your we our
    """.split()
)
