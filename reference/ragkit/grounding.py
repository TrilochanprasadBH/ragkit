"""Stage 06 reference: the sufficiency gate.

A grounding check answers one question: does the material in front of me actually
support an answer, or does it merely talk about the topic?

The distinction matters because a retriever always returns something. Ask about Kafka
and the top five chunks will be about middleware and background tasks, because those
are the closest things in this corpus. A reader without a gate will answer a Kafka
question using FastAPI documentation, confidently and wrongly.

**How the gate works.** Each question term is weighted by how rare it is in the corpus.
A word that appears nowhere is maximally informative: if the question says "terraform"
and the corpus never does, no amount of topical similarity makes the pack sufficient.

    sufficiency = sum(idf(t) for t in question terms present in pack)
                / sum(idf(t) for t in question terms)

The score is between 0 and 1. A question whose rare terms are all present scores near
1. A question about an absent technology scores near 0 even though half its words are
common English.

**What the gate cannot do.** It cannot tell "this pack answers the question" from
"this pack is about the same subject". Both produce moderate coverage. That is why the
threshold is a trade-off rather than a solution, and why the lesson reports a curve
instead of a single number.

**Where the threshold comes from.** Not from accuracy. From which error costs more. A
false answer misleads a user who will act on it; a false refusal annoys them and they
rephrase. For documentation, the asymmetry favours refusing.
"""

from __future__ import annotations

import math
from collections import Counter

from . import analysis
from .answer import Answer, answer as read, refuse
from .types import Chunk


class Grounder:
    def __init__(self, chunks: list[Chunk], min_sufficiency: float = 0.5) -> None:
        self.min_sufficiency = min_sufficiency

        df: Counter = Counter()
        for chunk in chunks:
            for term in set(analysis.tokenize(" ".join([*chunk.headings, chunk.text]))):
                df[term] += 1

        self.n = len(chunks)
        self.df = df

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def sufficiency(self, question: str, pack_text: str) -> float:
        terms = set(analysis.tokenize(question))
        if not terms:
            return 0.0
        pack_terms = set(analysis.tokenize(pack_text))
        total = sum(self.idf(term) for term in terms)
        if total <= 0:
            return 0.0
        present = sum(self.idf(term) for term in terms if term in pack_terms)
        return present / total

    def answer(self, question: str, pack_text: str) -> Answer:
        """Answer only if the pack is sufficient; otherwise refuse."""
        if self.sufficiency(question, pack_text) < self.min_sufficiency:
            return refuse()
        return read(question, pack_text)


def sufficiency(question: str, pack_text: str, chunks: list[Chunk]) -> float:
    """Convenience wrapper for one-off checks."""
    return Grounder(chunks).sufficiency(question, pack_text)
