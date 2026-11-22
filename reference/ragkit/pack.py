"""Stage 05 reference: packing retrieved chunks into a citable context window.

A context window is a budget, and a citation is a promise. This module has to satisfy
both at once.

The naive approach is to concatenate whole chunks in rank order until the budget runs
out. It is easy, it is what most implementations do, and it wastes most of the budget
on sentences that have nothing to do with the question.

The approach here scores individual sentences against the question, then buys the
highest-value sentences that fit, keeping each one attached to the chunk it came from
so that it can be cited.

Three rules make the packing honest:

  - a sentence is never cut in half; the budget buys sentences or nothing
  - every passage carries a number and a source label, so a reader can check it
  - if nothing scores above zero, the pack is empty and the system should refuse
    rather than answer from nothing
"""

from __future__ import annotations

import re
from . import analysis
from .baselines import label_for
from .types import Chunk, Citation, Pack

SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE.split(text) if part.strip()]


def _score(sentence: str, query_terms: set[str]) -> float:
    """Overlap with the query, normalised by length.

    Division by length matters: a long paragraph contains more query terms by
    accident, and without normalisation the packer would buy one enormous vague
    passage instead of three precise ones.
    """
    terms = analysis.tokenize(sentence)
    if not terms:
        return 0.0
    overlap = sum(1 for term in terms if term in query_terms)
    if overlap == 0:
        return 0.0
    return overlap / (len(terms) ** 0.5)


def pack(question: str, chunks: list[Chunk], budget: int = 120) -> Pack:
    """Select the sentences that best answer `question`, within `budget` words."""
    query_terms = set(analysis.tokenize(question))
    if not query_terms or not chunks:
        return Pack("", (), 0)

    # Rank candidate sentences, keeping their source chunk and rank.
    candidates: list[tuple[float, int, int, str, Chunk]] = []
    for chunk_rank, chunk in enumerate(chunks):
        for position, sentence in enumerate(sentences(chunk.text)):
            score = _score(sentence, query_terms)
            if score > 0:
                candidates.append((score, chunk_rank, position, sentence, chunk))

    if not candidates:
        return Pack("", (), 0)

    # Highest value per token first; ties prefer the better-ranked chunk.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used = 0
    chosen: list[tuple[int, int, str, Chunk]] = []
    numbers: dict[str, int] = {}
    for score, chunk_rank, position, sentence, chunk in candidates:
        cost = len(sentence.split())
        label_cost = 0
        if chunk.chunk_id not in numbers:
            label_cost = len(label_for(chunk).split()) + 1
        if used + cost + label_cost > budget:
            continue
        if chunk.chunk_id not in numbers:
            numbers[chunk.chunk_id] = len(numbers) + 1
        chosen.append((chunk_rank, position, sentence, chunk))
        used += cost + label_cost

    if not chosen:
        return Pack("", (), 0)

    # Present the selected material in document order, not score order, with the
    # label on its own line so that a reader can check the source at a glance.
    chosen.sort(key=lambda item: (numbers[item[3].chunk_id], item[1]))

    by_id = {chunk.chunk_id: chunk for _rank, _pos, _sentence, chunk in chosen}
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for _rank, _position, sentence, chunk in chosen:
        if chunk.chunk_id not in grouped:
            grouped[chunk.chunk_id] = []
            order.append(chunk.chunk_id)
        grouped[chunk.chunk_id].append(sentence)

    blocks = [
        f"[{numbers[chunk_id]}] {label_for(by_id[chunk_id])}\n"
        + " ".join(grouped[chunk_id])
        for chunk_id in order
    ]

    text = "\n\n".join(blocks)
    citations = tuple(
        Citation(number=numbers[chunk_id], chunk_id=chunk_id, label=label_for(by_id[chunk_id]))
        for chunk_id in order
    )
    return Pack(text=text, citations=citations, tokens=len(text.split()))


def pack_whole_chunks(question: str, chunks: list[Chunk], budget: int = 120) -> Pack:
    """The naive baseline: whole chunks in rank order until the budget fills.

    Kept in the project so the lesson can compare against it rather than assert that
    it is bad.
    """
    blocks: list[str] = []
    citations: list[Citation] = []
    used = 0

    for chunk in chunks:
        label = label_for(chunk)
        cost = len((chunk.text + " " + label).split())
        if used + cost > budget:
            continue
        number = len(citations) + 1
        citations.append(Citation(number=number, chunk_id=chunk.chunk_id, label=label))
        blocks.append(f"[{number}] {label}\n{chunk.text}")
        used += cost

    text = "\n\n".join(blocks)
    return Pack(text=text, citations=tuple(citations), tokens=len(text.split()))
