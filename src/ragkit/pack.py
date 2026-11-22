"""Stage 05 — packing retrieved chunks into a citable context window.

Read `stages/05_packing/README.md` first.

You implement one function. The naive baseline is provided in `baselines.py` so that
the lesson can compare against it, and the reader is provided in `answer.py` so that
the packing stage has a fixed downstream that can only use what you give it.
"""

from __future__ import annotations

from .types import Chunk, Pack


def pack(question: str, chunks: list[Chunk], budget: int = 180) -> Pack:
    """Select sentences that answer `question`, within `budget` words.

    Contract:
      - split each chunk's body into whole sentences; never cut a sentence in half
      - score each sentence against the question; sentences with no overlap are not
        candidates
      - prefer sentences that score highly per word, so that a long vague paragraph
        loses to a short precise one
      - stop when adding the next sentence would exceed the budget; skip it and try
        the next candidate rather than stopping early
      - present the chosen material in the order it appeared in the sources, grouped
        by chunk, each group introduced by `[n] <label>` on its own line
      - number the passages 1..n, and return a `Citation` for each
      - `tokens` is the word count of the final text
      - if no sentence overlaps, or no chunk fits, return an empty `Pack`

    The first citation number must be 1, and the numbers must be contiguous.

    Use `label_for` from `baselines.py` for the source label, and `analysis.tokenize`
    for terms. A sentence is worth what it adds to the pack, not how long it is.
    """
    raise NotImplementedError("stage 05: implement pack")
