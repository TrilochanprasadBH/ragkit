"""Stage 05: the reader.

This module is **given**. You do not implement it and you should not change it. It
exists so that the packing stage has something downstream that behaves like a model
under a constraint.

The reader can only use what is in the pack:

  - it splits the pack back into numbered passages
  - it scores each sentence by overlap with the question
  - it returns the best sentence, plus the number of the passage it came from
  - if nothing overlaps at all, it refuses

That last rule is the important one. A reader that always answers will always appear
to have an answer, which hides a broken retriever. A reader that refuses when the
context is empty makes retrieval and packing failures visible.

The scoring is deliberately crude. A real model would understand the sentence; this
one only counts words, which means any improvement you measure in this stage comes
from *what you put in the pack*, not from the reader's cleverness. That is the point.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import analysis

PASSAGE = re.compile(r"^\[(\d+)\]\s", re.MULTILINE)
SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Answer:
    text: str
    citation: int | None
    grounded: bool


def refuse() -> Answer:
    return Answer(
        text="I could not find that in the provided documentation.",
        citation=None,
        grounded=False,
    )


def split_passages(pack_text: str) -> list[tuple[int, str]]:
    """Split a pack into (number, body) pairs, preserving order."""
    matches = list(PASSAGE.finditer(pack_text))
    passages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(pack_text)
        passages.append((int(match.group(1)), pack_text[start:end].strip()))
    return passages


def answer(question: str, pack_text: str) -> Answer:
    if not pack_text.strip():
        return refuse()

    query_terms = set(analysis.tokenize(question))
    best: tuple[float, int, str] | None = None

    for number, body in split_passages(pack_text):
        for sentence in SENTENCE.split(body):
            sentence = sentence.strip()
            if not sentence:
                continue
            terms = analysis.tokenize(sentence)
            if not terms:
                continue
            overlap = sum(1 for term in terms if term in query_terms)
            if overlap == 0:
                continue
            score = overlap / (len(terms) ** 0.5)
            if best is None or score > best[0]:
                best = (score, number, sentence)

    if best is None:
        return refuse()

    _score, number, sentence = best
    return Answer(text=sentence, citation=number, grounded=True)
