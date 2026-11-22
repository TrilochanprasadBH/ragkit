"""Stage 03 tests: text analysis and the fielded index.

    python stages/03_analysis/test_stage03.py
    IMPL=reference python stages/03_analysis/test_stage03.py
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

from ragkit import analysis  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


class TestSplitIdentifier(unittest.TestCase):
    def test_pascal_case(self):
        self.assertEqual(analysis.split_identifier("HTMLResponse"), ["HTML", "Response"])

    def test_camel_case(self):
        self.assertEqual(analysis.split_identifier("getOrder"), ["get", "Order"])

    def test_acronym_then_word(self):
        self.assertEqual(analysis.split_identifier("HTTPServer"), ["HTTP", "Server"])

    def test_digits_split_both_ways(self):
        self.assertEqual(
            analysis.split_identifier("OAuth2PasswordBearer"),
            ["OAuth", "2", "Password", "Bearer"],
        )

    def test_snake_case(self):
        self.assertEqual(analysis.split_identifier("get_order"), ["get", "order"])

    def test_plain_word_is_unchanged(self):
        self.assertEqual(analysis.split_identifier("response"), ["response"])


class TestStem(unittest.TestCase):
    def test_plural(self):
        self.assertEqual(analysis.stem("headers"), "header")

    def test_ies(self):
        self.assertEqual(analysis.stem("policies"), "policy")

    def test_ing(self):
        self.assertEqual(analysis.stem("deploying"), "deploy")

    def test_ed(self):
        self.assertEqual(analysis.stem("deployed"), "deploy")

    def test_trailing_e(self):
        self.assertEqual(analysis.stem("response"), "respons")

    def test_short_words_are_untouched(self):
        self.assertEqual(analysis.stem("api"), "api")
        self.assertEqual(analysis.stem("is"), "is")

    def test_does_not_strip_is_suffix(self):
        self.assertEqual(analysis.stem("analysis"), "analysis")


class TestTokenize(unittest.TestCase):
    def test_splits_and_keeps_whole(self):
        # The whole run is kept, and it is stemmed like any other term.
        tokens = analysis.tokenize("HTMLResponse")
        self.assertEqual(tokens[0], "htmlrespons")
        self.assertIn("html", tokens)
        self.assertIn("respons", tokens)

    def test_snake_case_does_not_keep_whole(self):
        tokens = analysis.tokenize("get_order")
        self.assertEqual(tokens, ["get", "order"])

    def test_drops_stop_words(self):
        self.assertEqual(analysis.tokenize("the response of the server"), ["respons", "server"])

    def test_lowercases(self):
        self.assertEqual(analysis.tokenize("RESPONSE"), ["respons"])


class TestFieldedIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.index = build_fielded_index(cls.chunks)
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}

    def test_indexes_every_chunk(self):
        self.assertEqual(len(self.index.chunks), len(self.chunks))

    def test_finds_camel_case_identifier(self):
        results = self.index.search("CORSMiddleware", k=5)
        self.assertTrue(results)
        self.assertEqual(self.by_id[results[0][0]].source, "advanced/middleware.md")

    def test_finds_header_plural(self):
        results = self.index.search("custom header", k=5)
        sources = [self.by_id[chunk_id].source for chunk_id, _ in results]
        self.assertIn("advanced/response-headers.md", sources)

    def test_empty_query(self):
        self.assertEqual(self.index.search("the of and"), [])

    def test_deterministic(self):
        first = self.index.search("request body", k=5)
        second = self.index.search("request body", k=5)
        self.assertEqual(first, second)


class TestGoldenSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.index = build_fielded_index(cls.chunks)
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

    def test_recall_at_5(self):
        hits = 0
        for question in self.questions:
            results = self.index.search(question["question"], k=5)
            if any(chunk_id in self.gold[question["id"]] for chunk_id, _ in results):
                hits += 1
        recall = hits / len(self.questions)
        self.assertGreaterEqual(recall, 0.75, f"recall@5 was {recall:.1%}")

    def test_mrr(self):
        total = 0.0
        for question in self.questions:
            results = self.index.search(question["question"], k=5)
            for position, (chunk_id, _) in enumerate(results, start=1):
                if chunk_id in self.gold[question["id"]]:
                    total += 1.0 / position
                    break
        mrr = total / len(self.questions)
        self.assertGreaterEqual(mrr, 0.60, f"MRR was {mrr:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
