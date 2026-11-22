"""Stage 02 — lexical search with BM25.

Read `stages/02_lexical/README.md` first.

The index is built once and queried many times. Everything here operates on
`Chunk` objects produced in stage 01.

The tests in `stages/02_lexical/test_stage02.py` are the specification.
"""

from __future__ import annotations

from .types import Chunk


def tokenize(text: str) -> list[str]:
    """Split text into lowercased terms and drop stop words.

    Contract:
      - lowercase everything
      - keep only letters and digits, splitting on anything else
      - remove the stop words listed in STOP_WORDS
      - preserve duplicates, because term frequency matters to BM25
    """
    raise NotImplementedError("stage 02: implement tokenize")


class BM25Index:
    """A BM25 index over a list of chunks.

    BM25 scores a chunk for a query by summing, over query terms:

        idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))

    where `tf` is how often the term occurs in the chunk, `dl` is the chunk length
    in terms, and `avgdl` is the average across the corpus.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> None:
        raise NotImplementedError("stage 02: build the index")

    def idf(self, term: str) -> float:
        """Inverse document frequency for one term.

        Use the BM25 form: log(1 + (N - df + 0.5) / (df + 0.5)).
        Terms that appear in no document return the value for df = 0.
        """
        raise NotImplementedError("stage 02: implement idf")

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Return up to `k` (chunk_id, score) pairs, highest score first.

        Contract:
          - chunks that contain none of the query terms are excluded
          - ties are broken by chunk_id, so results are deterministic
          - an empty query, or one containing only stop words, returns []
        """
        raise NotImplementedError("stage 02: implement search")


def build_index(chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> BM25Index:
    """Convenience constructor used by the eval runner and later stages."""
    raise NotImplementedError("stage 02: implement build_index")


STOP_WORDS = frozenset(
    """
    a an and are as at be but by for if in into is it its of on or such that the
    their then there these they this to was will with you your we our
    """.split()
)
