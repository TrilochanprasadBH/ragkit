"""Stage 04 reference: embeddings.

An embedding turns text into a vector so that similarity becomes arithmetic. Two
questions matter and they are independent: what does the vector represent, and how
good is it.

Three embedders ship with this project.

**CharNgramEmbedder** (offline, deterministic). Hashes character n-grams into a fixed
number of dimensions. It captures morphology and spelling: "deploy", "deployed" and
"deployment" share most of their 4-grams, and so do identifiers with typos. It does
*not* capture meaning: "flag" and "enabled" share nothing.

**HashingEmbedder** (offline, deterministic). Hashes whole words. This is the weakest
possible embedding and it exists to be beaten. If your pipeline cannot distinguish it
from a good embedding, your eval is not measuring what you think.

**RemoteEmbedder** (needs an API key). Sends text to an OpenAI-compatible
`/v1/embeddings` endpoint. This is the only one that captures meaning, and it is the
one to use when you have credentials.

The interface is deliberately tiny: `embed(texts) -> list[dict[int, float]]`, where
each vector is sparse (index -> weight) and L2-normalised, so cosine similarity is a
dot product.

Sparse vectors are a practical choice, not a purity statement: for 362 chunks at 512
dimensions, a dict beats a dense list on both memory and clarity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request

NORMALISE = re.compile(r"[^a-z0-9]+")


def _unit_hash(value: str, salt: str = "") -> tuple[int, float]:
    """Stable hash of a string, returning (index, sign)."""
    digest = hashlib.blake2b((salt + value).encode(), digest_size=8).digest()
    number = int.from_bytes(digest, "big")
    return number % (2**32), 1.0 if (number >> 63) & 1 else -1.0


class CharNgramEmbedder:
    """Feature hashing over character n-grams.

    Why character n-grams:

      - morphology is free. "deploy" and "deployment" share 4-grams, so they land
        close together without any stemming rules.
      - typos and identifiers degrade gracefully instead of failing outright.
      - no training, no vocabulary, nothing to download.

    Why not:

      - no synonymy. "flag" and "enabled" share no characters and stay far apart.
      - larger n makes it stricter; smaller n makes it noisier. 3 and 4 together are
        the usual compromise.
    """

    name = "char-ngram"

    def __init__(self, dim: int = 512, sizes: tuple[int, ...] = (3, 4)) -> None:
        self.dim = dim
        self.sizes = sizes

    def _features(self, text: str) -> list[str]:
        features: list[str] = []
        for word in NORMALISE.sub(" ", text.casefold()).split():
            padded = f"_{word}_"
            for size in self.sizes:
                for start in range(len(padded) - size + 1):
                    features.append(padded[start : start + size])
        return features

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        vectors: list[dict[int, float]] = []
        for text in texts:
            vector: dict[int, float] = {}
            for feature in self._features(text):
                index, sign = _unit_hash(feature)
                bucket = index % self.dim
                vector[bucket] = vector.get(bucket, 0.0) + sign
            vectors.append(_l2_normalise(vector))
        return vectors


class HashingEmbedder:
    """Whole-word feature hashing. Deliberately weak, deliberately honest."""

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        vectors: list[dict[int, float]] = []
        for text in texts:
            vector: dict[int, float] = {}
            for word in NORMALISE.sub(" ", text.casefold()).split():
                index, sign = _unit_hash(word, salt="word:")
                bucket = index % self.dim
                vector[bucket] = vector.get(bucket, 0.0) + sign
            vectors.append(_l2_normalise(vector))
        return vectors


class RemoteEmbedder:
    """An OpenAI-compatible embeddings endpoint.

    Set EMBEDDING_API_KEY and optionally EMBEDDING_BASE_URL and EMBEDDING_MODEL.
    Works with OpenRouter, OpenAI, or any compatible gateway.
    """

    name = "remote"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
        self.api_key = api_key or os.environ.get("EMBEDDING_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("EMBEDDING_BASE_URL")
            or "https://api.openai.com/v1/embeddings"
        )
        self.timeout = timeout

        if not self.api_key:
            raise RuntimeError(
                "no EMBEDDING_API_KEY set; use CharNgramEmbedder, or export a key"
            )

    def embed(self, texts: list[str]) -> list[dict[int, float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode()
        request = urllib.request.Request(
            self.base_url,
            data=payload,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = json.loads(response.read().decode())

        vectors = []
        for item in sorted(body["data"], key=lambda entry: entry["index"]):
            dense = item["embedding"]
            vectors.append(_l2_normalise({i: value for i, value in enumerate(dense)}))
        return vectors


def _l2_normalise(vector: dict[int, float]) -> dict[int, float]:
    norm = math.sqrt(sum(value * value for value in vector.values()))
    if norm == 0:
        return {}
    return {index: value / norm for index, value in vector.items()}


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    """Dot product of two L2-normalised sparse vectors."""
    if len(a) > len(b):
        a, b = b, a
    return sum(weight * b.get(index, 0.0) for index, weight in a.items())


def build_embedder(kind: str = "char-ngram", **kwargs):
    if kind == "char-ngram":
        return CharNgramEmbedder(**kwargs)
    if kind == "hashing":
        return HashingEmbedder(**kwargs)
    if kind == "remote":
        return RemoteEmbedder(**kwargs)
    raise ValueError(f"unknown embedder {kind!r}")
