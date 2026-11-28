"""Stage 09 tests: a cross-encoder stand-in, and the measurement that decides whether it pays.

    python stages/09_rerank/test_stage09.py
    IMPL=reference python stages/09_rerank/test_stage09.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import build_embedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.rerank import CrossEncoder, RerankIndex, TermStats  # noqa: E402
from ragkit.types import Chunk  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


def chunk(chunk_id: str, text: str, headings: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="d1",
        source="test.md",
        text=text,
        position=0,
        headings=headings,
    )


class FixedIndex:
    """A first stage with a fixed order, for testing the wrapper in isolation."""

    def __init__(self, ids: list[str]) -> None:
        self.ids = ids
        self.calls: list[int] = []

    def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        self.calls.append(k)
        return [(chunk_id, 1.0 / (position + 1)) for position, chunk_id in enumerate(self.ids[:k])]


class TestTermStats(unittest.TestCase):
    def test_rarer_terms_weigh_more(self):
        stats = TermStats.from_chunks(
            [chunk("a", "deploy app"), chunk("b", "deploy app"), chunk("c", "certificate")]
        )
        self.assertGreater(stats.weight("certificate"), stats.weight("deploy"))

    def test_unseen_term_has_weight_one(self):
        stats = TermStats.from_chunks([chunk("a", "deploy app")])
        self.assertEqual(stats.weight("kubernetes"), 1.0)

    def test_weights_are_positive(self):
        stats = TermStats.from_chunks([chunk("a", "deploy app"), chunk("b", "certificate")])
        self.assertTrue(all(value > 0 for value in stats.idf.values()))


class TestCrossEncoder(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("tight", "deploy app with one command"),
            chunk(
                "loose",
                "deploy something entirely different then later on eventually app appears here",
            ),
            chunk("other", "certificate renewal happens automatically"),
        ]
        self.encoder = CrossEncoder(self.chunks)

    def test_deterministic(self):
        first = self.encoder.score("deploy app", self.chunks[0])
        second = self.encoder.score("deploy app", self.chunks[0])
        self.assertEqual(first, second)

    def test_empty_query_scores_zero(self):
        self.assertEqual(self.encoder.score("", self.chunks[0]), 0.0)

    def test_score_is_a_fraction(self):
        for text in ("deploy app", "certificate", "nothing here", "deploy command"):
            score = self.encoder.score(text, self.chunks[2])
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_adjacent_terms_beat_scattered_terms(self):
        tight = self.encoder.score("deploy app", self.chunks[0])
        loose = self.encoder.score("deploy app", self.chunks[1])
        self.assertGreater(tight, loose)

    def test_proximity_rewards_a_tighter_window(self):
        near = chunk("near", "rate limiting starts when requests exceed the quota")
        far = chunk(
            "far",
            "rate configuration is separate from quota and the limiting factor appears later",
        )
        encoder = CrossEncoder([near, far])
        self.assertGreater(
            encoder.score("rate limiting", near), encoder.score("rate limiting", far)
        )

    def test_heading_terms_count(self):
        body = "the value is read once at startup"
        with_heading = chunk("a", body, headings=("Query Parameters", "Defaults"))
        without = chunk("b", body)
        encoder = CrossEncoder([with_heading, without])
        self.assertGreater(
            encoder.score("query parameters", with_heading),
            encoder.score("query parameters", without),
        )


class TestRerankIndex(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            chunk("a", "deploy app with one command"),
            chunk("b", "certificate renewal happens automatically"),
            chunk("c", "deploy something entirely different then app"),
            chunk("d", "unrelated"),
        ]

    def test_cannot_invent_chunks(self):
        first = FixedIndex(["a", "b", "c"])
        index = RerankIndex(first, self.chunks, depth=3)
        returned = {chunk_id for chunk_id, _ in index.search("deploy app", k=5)}
        self.assertLessEqual(returned, {"a", "b", "c"})

    def test_depth_limits_what_can_be_returned(self):
        first = FixedIndex(["a", "b", "c", "d"])
        index = RerankIndex(first, self.chunks, depth=2)
        self.assertLessEqual(len(index.search("deploy", k=5)), 2)

    def test_weight_zero_is_the_identity(self):
        first = FixedIndex(["c", "b", "a"])
        index = RerankIndex(first, self.chunks, weight=0.0)
        self.assertEqual(
            [chunk_id for chunk_id, _ in index.search("deploy app", k=3)], ["c", "b", "a"]
        )

    def test_weight_one_orders_by_the_pair_score(self):
        first = FixedIndex(["b", "c", "a"])
        index = RerankIndex(first, self.chunks, weight=1.0)
        self.assertEqual([chunk_id for chunk_id, _ in index.search("deploy app", k=3)][0], "a")

    def test_empty_first_stage_returns_nothing(self):
        index = RerankIndex(FixedIndex([]), self.chunks)
        self.assertEqual(index.search("deploy", k=3), [])

    def test_invalid_weight_raises(self):
        with self.assertRaises(ValueError):
            RerankIndex(FixedIndex(["a"]), self.chunks, weight=1.5)

    def test_invalid_depth_raises(self):
        with self.assertRaises(ValueError):
            RerankIndex(FixedIndex(["a"]), self.chunks, depth=0)


class TestMeasured(unittest.TestCase):
    """The numbers that decide whether a reranker belongs in the pipeline.

    Tuned and reported on the development split. The held-out split has nine questions, which is
    too few to resolve a change of one hundredth of an MRR point — and the last two tests are
    there to make that failure visible rather than to hide it.
    """

    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.questions = [
            json.loads(line)
            for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        cls.gold = {
            question["id"]: {
                candidate.chunk_id
                for candidate in cls.chunks
                if question["gold_phrase"].casefold() in candidate.text.casefold()
                or question["gold_phrase"].casefold()
                in " ".join(candidate.headings).casefold()
            }
            for question in cls.questions
        }
        cls.dev = [q for q in cls.questions if q["split"] == "dev"]
        cls.test = [q for q in cls.questions if q["split"] == "test"]

        fielded = build_fielded_index(cls.chunks)
        dense = DenseIndex(cls.chunks, build_embedder("char-ngram"))
        cls.dense = dense
        cls.hybrid = HybridIndex(fielded, dense)
        cls.dense_reranked = RerankIndex(dense, cls.chunks)
        cls.hybrid_reranked = RerankIndex(cls.hybrid, cls.chunks)

    def measure(self, index, subset):
        hits = 0
        total = 0.0
        for question in subset:
            results = index.search(question["question"], k=5)
            rank = next(
                (
                    position + 1
                    for position, (chunk_id, _) in enumerate(results)
                    if chunk_id in self.gold[question["id"]]
                ),
                None,
            )
            if rank is not None and rank <= 5:
                hits += 1
                total += 1.0 / rank
        return hits / len(subset), total / len(subset)

    def test_reranking_a_weak_first_stage_earns_its_cost(self):
        base = self.measure(self.dense, self.dev)
        reranked = self.measure(self.dense_reranked, self.dev)
        self.assertGreaterEqual(reranked[1] - base[1], 0.05)

    def test_reranking_a_strong_first_stage_does_not(self):
        base = self.measure(self.hybrid, self.dev)
        reranked = self.measure(self.hybrid_reranked, self.dev)
        self.assertLess(reranked[1] - base[1], 0.02)
        self.assertLess(reranked[0], base[0], "the reranker demotes chunks fusion got right")

    def test_the_held_out_split_disagrees_with_dev(self):
        # Dense gains 0.13 on dev and loses ground on the nine held-out questions. That is not a
        # property of the reranker; it is the interval on nine questions being wider than the
        # effect. The lesson reports both numbers and claims neither.
        dev_gain = self.measure(self.dense_reranked, self.dev)[1] - self.measure(self.dense, self.dev)[1]
        test_gain = self.measure(self.dense_reranked, self.test)[1] - self.measure(self.dense, self.test)[1]
        self.assertGreater(dev_gain, 0.05)
        self.assertLess(test_gain, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
