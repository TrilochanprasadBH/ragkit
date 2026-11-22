"""Stage 06 tests: the sufficiency gate.

    python stages/06_grounding/test_stage06.py
    IMPL=reference python stages/06_grounding/test_stage06.py
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

from ragkit.answer import refuse  # noqa: E402
from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import CharNgramEmbedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.grounding import Grounder  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.pack import pack  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
ANSWERS = ROOT / "evals" / "questions.jsonl"
UNANSWERS = ROOT / "evals" / "questions_unanswerable.jsonl"
BUDGET = 180


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Fixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}
        cls.retriever = HybridIndex(
            build_fielded_index(cls.chunks), DenseIndex(cls.chunks, CharNgramEmbedder())
        )
        cls.answerable = load(ANSWERS)
        cls.unanswerable = load(UNANSWERS)

    def packed_for(self, question: str):
        hits = self.retriever.search(question, k=5)
        docs = [self.by_id[chunk_id] for chunk_id, _ in hits]
        return pack(question, docs, budget=BUDGET)


class TestIdf(Fixtures):
    def test_rare_words_score_higher(self):
        grounder = Grounder(self.chunks)
        self.assertGreater(grounder.idf("terraform"), grounder.idf("request"))

    def test_idf_is_never_negative(self):
        grounder = Grounder(self.chunks)
        for term in ("request", "response", "app", "body"):
            self.assertGreaterEqual(grounder.idf(term), 0.0)


class TestSufficiency(Fixtures):
    def test_full_coverage_is_one(self):
        grounder = Grounder(self.chunks)
        question = "query parameters"
        self.assertAlmostEqual(
            grounder.sufficiency(question, "query parameters are described here"), 1.0
        )

    def test_absent_technology_scores_low(self):
        grounder = Grounder(self.chunks)
        packed = self.packed_for("How do I configure Terraform state locking with S3?")
        self.assertLess(grounder.sufficiency("How do I configure Terraform state locking with S3?", packed.text), 0.75)

    def test_empty_question_is_zero(self):
        grounder = Grounder(self.chunks)
        self.assertEqual(grounder.sufficiency("the of and", "anything"), 0.0)

    def test_empty_pack_is_zero(self):
        grounder = Grounder(self.chunks)
        self.assertEqual(grounder.sufficiency("query parameters", ""), 0.0)


class TestGate(Fixtures):
    def test_refuses_below_threshold(self):
        grounder = Grounder(self.chunks, min_sufficiency=0.99)
        result = grounder.answer("query parameters", "query parameters")
        self.assertFalse(result.grounded)
        self.assertEqual(result.text, refuse().text)

    def test_defers_above_threshold(self):
        grounder = Grounder(self.chunks, min_sufficiency=0.0)
        result = grounder.answer(
            "query parameters default",
            "[1] tutorial/query-params.md\nQuery parameters have a default value.",
        )
        self.assertTrue(result.grounded)
        self.assertEqual(result.citation, 1)


class TestSeparation(Fixtures):
    def test_answerable_questions_score_higher_on_average(self):
        grounder = Grounder(self.chunks)
        good = [
            grounder.sufficiency(q["question"], self.packed_for(q["question"]).text)
            for q in self.answerable
        ]
        bad = [
            grounder.sufficiency(q["question"], self.packed_for(q["question"]).text)
            for q in self.unanswerable
        ]
        self.assertGreater(sum(good) / len(good), sum(bad) / len(bad) + 0.05)

    def test_default_threshold_keeps_both_errors_bounded(self):
        grounder = Grounder(self.chunks, min_sufficiency=0.5)

        false_refusals = sum(
            1
            for q in self.answerable
            if not grounder.answer(q["question"], self.packed_for(q["question"]).text).grounded
        )
        false_answers = sum(
            1
            for q in self.unanswerable
            if grounder.answer(q["question"], self.packed_for(q["question"]).text).grounded
        )

        self.assertLessEqual(false_refusals / len(self.answerable), 0.30)
        self.assertLessEqual(false_answers / len(self.unanswerable), 0.30)

    def test_no_gate_answers_everything(self):
        grounder = Grounder(self.chunks, min_sufficiency=0.0)
        answered = sum(
            1
            for q in self.unanswerable
            if grounder.answer(q["question"], self.packed_for(q["question"]).text).grounded
        )
        self.assertGreater(answered / len(self.unanswerable), 0.9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
