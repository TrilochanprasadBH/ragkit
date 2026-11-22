"""Stage 04 — embeddings.

Read `stages/04_dense/README.md` first.

An embedding maps text to a vector so that similarity becomes arithmetic. Vectors here
are sparse: `dict[int, float]` from dimension to weight, L2-normalised, so cosine
similarity is a dot product.
"""

from __future__ import annotations


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Dot product of two L2-normalised sparse vectors.

    Iterate over the smaller vector. Vectors that share no dimension score 0.
    """
    raise NotImplementedError("stage 04: implement cosine")


class CharNgramEmbedder:
    """Feature hashing over character n-grams.

    Contract:
      - lowercase the text and replace every non-alphanumeric run with a space
      - for each word, wrap it as `_word_` and take every substring of the configured
        sizes (default 3 and 4)
      - hash each n-gram to a dimension and a sign, and add the sign to that bucket
      - L2-normalise the result

    Expected behaviour to aim for: "deploy" and "deployment" must be closer than
    "deploy" and "banana". That property comes from shared substrings, not from any
    understanding of meaning.
    """

    name = "char-ngram"

    def __init__(self, dim: int = 512, sizes: tuple[int, ...] = (3, 4)) -> None:
        self.dim = dim
        self.sizes = sizes

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        raise NotImplementedError("stage 04: implement CharNgramEmbedder.embed")


class HashingEmbedder:
    """Whole-word feature hashing. The weakest embedding that still counts as one.

    It exists so you can see what a bad embedding does to a hybrid pipeline: it has
    no morphology and no semantics, so "deploy" and "deployment" land in unrelated
    buckets.
    """

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        raise NotImplementedError("stage 04: implement HashingEmbedder.embed")


def build_embedder(kind: str = "char-ngram", **kwargs):
    """Return an embedder by name: 'char-ngram', 'hashing', or 'remote'."""
    raise NotImplementedError("stage 04: implement build_embedder")
