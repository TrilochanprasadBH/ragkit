"""Stage 04 tests: embeddings, dense retrieval, and rank fusion.

    python stages/04_dense/test_stage04.py
    IMPL=reference python stages/04_dense/test_stage04.py
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.dense import DenseIndex, HybridIndex, reciprocal_rank_fusion  # noqa: E402
from ragkit.embeddings import (  # noqa: E402
    CharNgramEmbedder,
    HashingEmbedder,
    build_embedder,
    cosine,
)
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


class TestCosine(unittest.TestCase):
    def test_identical_vectors(self):
        a = {0: 0.6, 3: 0.8}
        self.assertAlmostEqual(cosine(a, a), 1.0, places=6)

    def test_disjoint_vectors(self):
        self.assertEqual(cosine({0: 1.0}, {1: 1.0}), 0.0)

    def test_order_does_not_matter(self):
        a = {0: 0.6, 3: 0.8}
        b = {3: 0.8, 0: 0.6}
        self.assertAlmostEqual(cosine(a, b), cosine(b, a), places=9)


class TestCharNgramEmbedder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.embedder = CharNgramEmbedder()

    def test_is_deterministic(self):
        first = self.embedder.embed(["deployment strategy"])
        second = self.embedder.embed(["deployment strategy"])
        self.assertEqual(first, second)

    def test_vectors_are_normalised(self):
        vector = self.embedder.embed(["deployment strategy"])[0]
        norm = math.sqrt(sum(value * value for value in vector.values()))
        self.assertAlmostEqual(norm, 1.0, places=6)

    def test_morphology_is_captured(self):
        deploy, deployment, banana = self.embedder.embed(
            ["deploy", "deployment", "banana"]
        )
        self.assertGreater(cosine(deploy, deployment), cosine(deploy, banana))

    def test_empty_text_gives_empty_vector(self):
        self.assertEqual(self.embedder.embed([""])[0], {})


class TestHashingEmbedder(unittest.TestCase):
    def test_has_no_morphology(self):
        embedder = HashingEmbedder()
        deploy, deployment, banana = embedder.embed(["deploy", "deployment", "banana"])
        # Whole-word hashing cannot relate a word to its own inflection.
        self.assertAlmostEqual(cosine(deploy, deployment), cosine(deploy, banana), places=6)


class TestBuildEmbedder(unittest.TestCase):
    def test_known_kinds(self):
        self.assertIsInstance(build_embedder("char-ngram"), CharNgramEmbedder)
        self.assertIsInstance(build_embedder("hashing"), HashingEmbedder)

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            build_embedder("telepathy")


class TestDenseIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.index = DenseIndex(cls.chunks, CharNgramEmbedder())
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}

    def test_returns_at_most_k(self):
        results = self.index.search("deployment", k=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)

    def test_scores_are_descending(self):
        scores = [score for _, score in self.index.search("request body", k=10)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_is_deterministic(self):
        self.assertEqual(
            self.index.search("query parameters", k=5),
            self.index.search("query parameters", k=5),
        )

    def test_finds_the_right_document(self):
        results = self.index.search("deployment containers docker image", k=5)
        sources = [self.by_id[chunk_id].source for chunk_id, _ in results]
        self.assertIn("deployment/docker.md", sources)


class TestReciprocalRankFusion(unittest.TestCase):
    def test_hand_computed(self):
        first = [("a", 9.0), ("b", 8.0)]
        second = [("b", 0.9), ("c", 0.8)]
        fused = reciprocal_rank_fusion([first, second], k=60)
        # b: 1/62 + 1/61, a: 1/61, c: 1/62
        self.assertEqual([chunk_id for chunk_id, _ in fused], ["b", "a", "c"])

    def test_scores_are_ignored(self):
        small = reciprocal_rank_fusion([[("a", 0.001), ("b", 0.0009)]], k=60)
        large = reciprocal_rank_fusion([[("a", 999.0), ("b", 1.0)]], k=60)
        self.assertEqual(small, large)

    def test_single_list_preserves_order(self):
        only = [("x", 5.0), ("y", 4.0), ("z", 3.0)]
        self.assertEqual(
            [chunk_id for chunk_id, _ in reciprocal_rank_fusion([only])],
            ["x", "y", "z"],
        )

    def test_empty_input(self):
        self.assertEqual(reciprocal_rank_fusion([]), [])

    def test_k_controls_how_much_rank_one_matters(self):
        # Smaller k makes the gap between adjacent ranks larger, so a first-place
        # finish dominates. Larger k flattens the list and rewards consistency
        # across lists instead.
        ranking = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        sharp = dict(reciprocal_rank_fusion([ranking], k=1))
        gentle = dict(reciprocal_rank_fusion([ranking], k=60))
        self.assertGreater(sharp["a"] / sharp["b"], gentle["a"] / gentle["b"])


class TestHybrid(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.lexical = build_fielded_index(cls.chunks)
        cls.dense = DenseIndex(cls.chunks, CharNgramEmbedder())
        cls.hybrid = HybridIndex(cls.lexical, cls.dense)
        cls.questions = [
            json.loads(line)
            for line in QUESTIONS.read_text().splitlines()
            if line.strip()
        ]
        cls.gold = {
            q["id"]: {
                c.chunk_id
                for c in cls.chunks
                if q["gold_phrase"].casefold() in c.text.casefold()
            }
            for q in cls.questions
        }

    def test_returns_k_results(self):
        self.assertLessEqual(len(self.hybrid.search("request body", k=5)), 5)

    def test_beats_lexical_on_recall(self):
        def recall(index):
            hits = 0
            for question in self.questions:
                results = index.search(question["question"], k=5)
                if any(chunk_id in self.gold[question["id"]] for chunk_id, _ in results):
                    hits += 1
            return hits / len(self.questions)

        self.assertGreaterEqual(recall(self.hybrid), 0.85)
        self.assertGreaterEqual(recall(self.hybrid), recall(self.lexical))

    def test_mrr_floor(self):
        total = 0.0
        for question in self.questions:
            for position, (chunk_id, _) in enumerate(
                self.hybrid.search(question["question"], k=5), start=1
            ):
                if chunk_id in self.gold[question["id"]]:
                    total += 1.0 / position
                    break
        self.assertGreaterEqual(total / len(self.questions), 0.65)


if __name__ == "__main__":
    unittest.main(verbosity=2)
