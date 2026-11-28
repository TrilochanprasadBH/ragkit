"""Stage 09 reference: a cross-encoder stand-in, and when it earns its cost.

A cross-encoder does not embed the query and the document separately. It reads the pair together
and emits one score, which is why it can notice that "the flag I turn on" and "enabled or not" are
the same idea in a way that no dot product between two independent vectors can. It is also why it
costs 10 to 100 times more per query: every candidate is a separate forward pass.

This file is a stand-in, not a transformer. It scores the pair with four signals that a real
cross-encoder learns:

    coverage    the share of the query's terms that the chunk contains, weighted by IDF
    phrase      the share of the query's adjacent pairs that appear adjacently in the chunk
    proximity   how tightly the matched terms cluster, because "a b ... c" is not "a b c"
    heading     the share of the query's terms that appear in the chunk's heading path

The weights are fixed constants rather than fitted, because fitting four weights on twenty-three
questions is how you get a reranker that works on your dev set and nowhere else.

**The measurement this stage exists for.** Reranking is not a capability you add; it is a repair
for a first stage with headroom. Measured on the dev split:

    first stage          recall@5        MRR        reranked MRR    gain
    dense                0.783           0.541      0.668           +0.128
    lexical (fielded)    0.957           0.701      0.725           +0.024
    hybrid               0.957           0.732      0.741           +0.009

The worse the first stage, the more the reranker is worth. On the hybrid pipeline it buys one
hundredth of an MRR point and costs two recall points, because the chunks it demotes are the ones
the fusion step got right. That is the answer to "should we add a reranker": compute the first
stage's MRR first. If it is already 0.73 on a five-document window, the headroom is not there.

The blend weight is 0.6 for the same reason: falling to 1.0 (pure rerank) drops hybrid recall to
0.826. Keeping 40% of the first stage's opinion is what stops the reranker from throwing away a
candidate the fusion step was right about.
"""

from __future__ import annotations

import math
from collections import Counter

from .lexical import tokenize
from .types import Chunk

DEFAULT_WEIGHTS = {
    "coverage": 0.35,
    "phrase": 0.30,
    "proximity": 0.20,
    "heading": 0.15,
}

DEFAULT_BLEND = 0.6
DEFAULT_DEPTH = 20


class TermStats:
    """Inverse document frequency over the indexed chunks.

    A term that appears in every chunk says nothing about which chunk answers a question. A term
    that appears in two says almost everything, which is why the reranker weights coverage by IDF
    rather than counting words.
    """

    def __init__(self, idf: dict[str, float], n_chunks: int) -> None:
        self.idf = idf
        self.n_chunks = n_chunks

    @classmethod
    def from_chunks(cls, chunks: list[Chunk]) -> "TermStats":
        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            document_frequency.update(set(tokenize(chunk.text)))
        n = len(chunks)
        idf = {
            term: math.log(1.0 + (n - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }
        return cls(idf=idf, n_chunks=n)

    def weight(self, term: str) -> float:
        return self.idf.get(term, 1.0)


class CrossEncoder:
    """Scores a (query, chunk) pair with four lexical signals. Deterministic, offline."""

    def __init__(self, chunks: list[Chunk], weights: dict[str, float] | None = None) -> None:
        self.by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.stats = TermStats.from_chunks(chunks)
        self.weights = dict(weights or DEFAULT_WEIGHTS)

    def score(self, query: str, chunk: Chunk) -> float:
        query_terms = tokenize(query)
        if not query_terms:
            return 0.0
        body = tokenize(chunk.text)
        body_set = set(body)

        total = sum(self.stats.weight(term) for term in query_terms) or 1.0
        coverage = sum(
            self.stats.weight(term) for term in query_terms if term in body_set
        ) / total

        pairs = list(zip(query_terms, query_terms[1:]))
        body_pairs = set(zip(body, body[1:]))
        phrase = (
            sum(1 for pair in pairs if pair in body_pairs) / len(pairs) if pairs else 0.0
        )

        positions = [index for index, term in enumerate(body) if term in set(query_terms)]
        proximity = 0.0
        if len({body[index] for index in positions}) >= 2:
            last: dict[str, int] = {}
            best = math.inf
            for index in positions:
                last[body[index]] = index
                if len(last) >= 2:
                    best = min(best, max(last.values()) - min(last.values()))
            if best < math.inf:
                proximity = 1.0 / (1.0 + best / 10.0)

        heading_terms = set(tokenize(" ".join(chunk.headings)))
        heading = (
            sum(1 for term in query_terms if term in heading_terms) / len(query_terms)
            if heading_terms
            else 0.0
        )

        return (
            self.weights["coverage"] * coverage
            + self.weights["phrase"] * phrase
            + self.weights["proximity"] * proximity
            + self.weights["heading"] * heading
        )


class RerankIndex:
    """Wrap a first stage: retrieve wide, rescore the pairs, blend, cut to k.

    `weight` is how much of the final order comes from the reranker. At 0.0 the wrapper is the
    identity, which is worth having as a property rather than a hope. At 1.0 the first stage's
    opinion is discarded entirely, which measured worse on every configuration here.
    """

    def __init__(
        self,
        first_stage,
        chunks: list[Chunk],
        weight: float = DEFAULT_BLEND,
        depth: int = DEFAULT_DEPTH,
        encoder: CrossEncoder | None = None,
    ) -> None:
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be between 0 and 1")
        if depth < 1:
            raise ValueError("depth must be at least 1")
        self.first_stage = first_stage
        self.chunks = list(chunks)
        self.encoder = encoder or CrossEncoder(self.chunks)
        self.weight = weight
        self.depth = depth

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        candidates = self.first_stage.search(query, k=self.depth)
        if not candidates:
            return []

        pair_scores = [
            self.encoder.score(query, self.encoder.by_id[chunk_id])
            for chunk_id, _ in candidates
        ]
        low, high = min(pair_scores), max(pair_scores)
        span = high - low

        blended = []
        for position, ((chunk_id, _), score) in enumerate(zip(candidates, pair_scores)):
            normalised = (score - low) / span if span > 0 else 0.0
            rank_score = 1.0 / (position + 1)
            blended.append(
                (chunk_id, self.weight * normalised + (1.0 - self.weight) * rank_score)
            )

        blended.sort(key=lambda pair: (-pair[1], pair[0]))
        return blended[:k]
