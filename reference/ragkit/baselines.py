"""Baseline packers, given so that the lesson can compare against them.

`pack_whole_chunks` is the naive approach: concatenate whole chunks in rank order
until the budget is full. It is what most implementations do before anyone measures
anything, and it is here so that you can measure it rather than assert that it is bad.
"""

from __future__ import annotations

from .types import Chunk, Citation, Pack


def label_for(chunk: Chunk) -> str:
    if chunk.headings:
        return f"{chunk.source} > {' > '.join(chunk.headings)}"
    return chunk.source


def pack_whole_chunks(question: str, chunks: list[Chunk], budget: int = 180) -> Pack:
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
