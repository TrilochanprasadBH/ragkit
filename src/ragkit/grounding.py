"""Stage 06 — the sufficiency gate.

Read `stages/06_grounding/README.md` first.

You implement two methods. The idea is in the lesson; the contract is here.

A retriever always returns something, so "I found passages" is not the same as "the
passages answer the question". This module decides the difference, and refusing is a
correct outcome.
"""

from __future__ import annotations

from .answer import Answer
from .types import Chunk


class Grounder:
    def __init__(self, chunks: list[Chunk], min_sufficiency: float = 0.5) -> None:
        """Record document frequencies so that terms can be weighted by rarity.

        Contract:
          - count, for every chunk, the distinct terms in the heading path and body
          - store the number of chunks, so idf can be computed
        """
        raise NotImplementedError("stage 06: implement Grounder.__init__")

    def idf(self, term: str) -> float:
        """log(1 + (N - df + 0.5) / (df + 0.5)), as in every previous stage."""
        raise NotImplementedError("stage 06: implement Grounder.idf")

    def sufficiency(self, question: str, pack_text: str) -> float:
        """How much of the question, weighted by rarity, is present in the pack.

        Contract:
          - take the distinct terms of the question
          - total = sum of idf over those terms
          - present = sum of idf over the terms that also appear in the pack
          - return present / total, or 0.0 when there are no terms

        A term that appears nowhere in the corpus has the highest idf, so a question
        about an absent technology scores near zero even if most of its words are
        common English.
        """
        raise NotImplementedError("stage 06: implement Grounder.sufficiency")

    def answer(self, question: str, pack_text: str) -> Answer:
        """Refuse when the pack is not sufficient; otherwise defer to the reader.

        Contract:
          - compute sufficiency
          - if it is below `min_sufficiency`, return `refuse()`
          - otherwise return `read(question, pack_text)`
        """
        raise NotImplementedError("stage 06: implement Grounder.answer")
