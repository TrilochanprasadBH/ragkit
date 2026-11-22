"""Stage 02 tests: BM25 lexical search.

    python stages/02_lexical/test_stage02.py
    IMPL=reference python stages/02_lexical/test_stage02.py
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

from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.lexical import BM25Index, build_index, tokenize  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


class TestTokenize(unittest.TestCase):
    def test_lowercases_and_splits(self):
        self.assertEqual(tokenize("Query PARAMETERS!"), ["query", "parameters"])

    def test_drops_stop_words(self):
        self.assertEqual(tokenize("the value of the parameter"), ["value", "parameter"])

    def test_keeps_duplicates(self):
        self.assertEqual(tokenize("path path path"), ["path", "path", "path"])

    def test_keeps_identifiers_whole(self):
        # Camel-case splitting is a stage 03 concern. Here an identifier is one term.
        self.assertEqual(tokenize("OAuth2PasswordBearer"), ["oauth2passwordbearer"])


class TestIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.index = build_index(cls.chunks)

    def test_indexes_every_chunk(self):
        self.assertEqual(len(self.index.chunks), len(self.chunks))
        for chunk in self.chunks:
            self.assertIn(chunk.chunk_id, self.index.term_counts)

    def test_common_terms_have_lower_idf(self):
        common = self.index.idf("request")
        rare = self.index.idf("corsmiddleware")
        self.assertLess(common, rare)

    def test_unseen_term_has_the_highest_idf(self):
        unseen = self.index.idf("zzzznonexistentterm")
        self.assertGreater(unseen, self.index.idf("request"))
        # Stop words are removed at tokenisation, so the index has never seen them.
        self.assertEqual(self.index.df.get("the", 0), 0)

    def test_idf_is_never_negative(self):
        for term in ("request", "response", "def", "app"):
            self.assertGreaterEqual(self.index.idf(term), 0.0)


class TestSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.index = build_index(cls.chunks)
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}

    def test_returns_at_most_k(self):
        results = self.index.search("response", k=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)

    def test_scores_are_descending(self):
        results = self.index.search("query parameters", k=10)
        scores = [score for _, score in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_query_returns_nothing(self):
        self.assertEqual(self.index.search(""), [])

    def test_stop_words_only_returns_nothing(self):
        self.assertEqual(self.index.search("the of and"), [])

    def test_exact_identifier_is_found(self):
        results = self.index.search("CORSMiddleware", k=5)
        self.assertTrue(results)
        top_chunk = self.by_id[results[0][0]]
        self.assertEqual(top_chunk.source, "advanced/middleware.md")

    def test_headings_are_searchable(self):
        results = self.index.search("lifespan events", k=5)
        self.assertTrue(results)
        top_chunk = self.by_id[results[0][0]]
        self.assertEqual(top_chunk.source, "advanced/events.md")

    def test_results_are_deterministic(self):
        first = self.index.search("response model validation", k=5)
        second = self.index.search("response model validation", k=5)
        self.assertEqual(first, second)

    def test_missing_term_scores_nothing(self):
        self.assertEqual(self.index.search("zzzznonexistentterm"), [])


class TestGoldenSet(unittest.TestCase):
    """Quality floor. These are the numbers stage 03 has to beat."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = chunk_corpus(CORPUS)
        cls.index = build_index(cls.chunks)
        cls.questions = [
            json.loads(line)
            for line in QUESTIONS.read_text().splitlines()
            if line.strip()
        ]

    def test_labels_resolve_to_chunks(self):
        for question in self.questions:
            phrase = question["gold_phrase"].casefold()
            found = any(
                phrase in chunk.text.casefold()
                or phrase in " ".join(chunk.headings).casefold()
                for chunk in self.chunks
            )
            self.assertTrue(found, f"{question['id']}: phrase not present in corpus")

    def test_recall_at_5(self):
        hits = 0
        for question in self.questions:
            phrase = question["gold_phrase"].casefold()
            gold = {
                c.chunk_id
                for c in self.chunks
                if phrase in c.text.casefold()
                or phrase in " ".join(c.headings).casefold()
            }
            results = self.index.search(question["question"], k=5)
            if any(chunk_id in gold for chunk_id, _ in results):
                hits += 1

        recall = hits / len(self.questions)
        self.assertGreaterEqual(recall, 0.60, f"recall@5 was {recall:.1%}")

    def test_mrr(self):
        total = 0.0
        for question in self.questions:
            phrase = question["gold_phrase"].casefold()
            gold = {
                c.chunk_id
                for c in self.chunks
                if phrase in c.text.casefold()
                or phrase in " ".join(c.headings).casefold()
            }
            results = self.index.search(question["question"], k=5)
            for position, (chunk_id, _) in enumerate(results, start=1):
                if chunk_id in gold:
                    total += 1.0 / position
                    break

        mrr = total / len(self.questions)
        self.assertGreaterEqual(mrr, 0.45, f"MRR was {mrr:.3f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
