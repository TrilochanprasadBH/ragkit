"""Stage 01 tests: loading and chunking.

Run from the repository root:

    python stages/01_ingest/test_stage01.py

To check the reference implementation instead of your own:

    IMPL=reference python stages/01_ingest/test_stage01.py
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.ingest import (  # noqa: E402
    chunk_corpus,
    chunk_document,
    estimate_tokens,
    load_documents,
)
from ragkit.types import Document  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"


def body_words(text: str) -> set[str]:
    """Every word in the document that is not part of a heading line."""
    words: set[str] = set()
    for line in text.splitlines():
        if re.match(r"^#{1,6}\s", line):
            continue
        words.update(re.findall(r"\S+", line))
    return words


class TestEstimateTokens(unittest.TestCase):
    def test_counts_words(self):
        self.assertEqual(estimate_tokens("one two three"), 3)

    def test_ignores_extra_whitespace(self):
        self.assertEqual(estimate_tokens("  one\n\n two  "), 2)

    def test_empty_is_zero(self):
        self.assertEqual(estimate_tokens(""), 0)


class TestLoadDocuments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_documents(CORPUS)

    def test_loads_every_markdown_file(self):
        on_disk = sorted(p for p in CORPUS.rglob("*.md"))
        self.assertEqual(len(self.docs), len(on_disk))
        self.assertGreaterEqual(len(self.docs), 20)

    def test_documents_are_not_empty(self):
        for doc in self.docs:
            self.assertTrue(doc.text.strip(), doc.path)
            self.assertTrue(doc.title.strip(), doc.path)

    def test_ids_are_unique(self):
        ids = [doc.doc_id for doc in self.docs]
        self.assertEqual(len(ids), len(set(ids)))

    def test_paths_are_relative_and_sorted(self):
        paths = [doc.path for doc in self.docs]
        self.assertEqual(paths, sorted(paths))
        for path in paths:
            self.assertFalse(path.startswith("/"), path)
            self.assertTrue(path.endswith(".md"), path)

    def test_title_has_no_anchor_markup(self):
        for doc in self.docs:
            self.assertNotIn("{ #", doc.title, doc.path)

    def test_query_params_document_is_present(self):
        doc = next(
            (d for d in self.docs if d.path.endswith("tutorial/query-params.md")),
            None,
        )
        self.assertIsNotNone(doc)
        self.assertEqual(doc.title, "Query Parameters")
        self.assertEqual(doc.doc_id, "tutorial-query-params")


class TestChunkDocument(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = {doc.path: doc for doc in load_documents(CORPUS)}
        cls.doc = cls.docs["tutorial/query-params.md"]
        cls.chunks = chunk_document(cls.doc, max_tokens=180, overlap=40)

    def test_produces_chunks(self):
        self.assertGreater(len(self.chunks), 1)

    def test_respects_max_tokens(self):
        for chunk in self.chunks:
            self.assertLessEqual(estimate_tokens(chunk.text), 180, chunk.chunk_id)

    def test_chunk_ids_are_unique_and_ordered(self):
        ids = [c.chunk_id for c in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual([c.position for c in self.chunks], list(range(len(self.chunks))))

    def test_covers_the_document_body(self):
        joined = " ".join(c.text for c in self.chunks)
        missing = body_words(self.doc.text) - set(re.findall(r"\S+", joined))
        self.assertEqual(missing, set(), f"lost words: {sorted(missing)[:10]}")

    def test_headings_become_a_path(self):
        nested = [c for c in self.chunks if len(c.headings) >= 2]
        self.assertTrue(nested, "expected at least one chunk under a sub-heading")
        first = nested[0]
        self.assertEqual(first.headings[0], "Query Parameters")
        self.assertIn(">", first.citation)
        self.assertIn("tutorial/query-params.md", first.citation)

    def test_every_chunk_has_text(self):
        for chunk in self.chunks:
            self.assertTrue(chunk.text.strip(), chunk.chunk_id)


class TestOverlap(unittest.TestCase):
    def test_overlap_carries_the_previous_tail(self):
        text = "\n\n".join(
            f"Sentence number {i} contains several ordinary words here." for i in range(200)
        )
        doc = Document(doc_id="synthetic", title="Synthetic", text=text, path="synthetic.md")
        chunks = chunk_document(doc, max_tokens=50, overlap=20)
        self.assertGreater(len(chunks), 3)
        first_unit_of_second = chunks[1].text.split("\n")[0]
        self.assertIn(first_unit_of_second, chunks[0].text)

    def test_no_overlap_duplicates_nothing(self):
        text = "\n\n".join(f"Sentence {i} has a handful of words." for i in range(120))
        doc = Document(doc_id="synthetic", title="Synthetic", text=text, path="synthetic.md")
        chunks = chunk_document(doc, max_tokens=40, overlap=0)
        total = sum(estimate_tokens(c.text) for c in chunks)
        self.assertEqual(total, estimate_tokens(doc.text))


class TestCorpusAndDeterminism(unittest.TestCase):
    def test_chunk_corpus_is_non_empty_and_stable(self):
        first = chunk_corpus(CORPUS)
        second = chunk_corpus(CORPUS)
        self.assertGreater(len(first), 50)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        self.assertEqual([c.text for c in first], [c.text for c in second])

    def test_all_chunk_ids_unique(self):
        chunks = chunk_corpus(CORPUS)
        self.assertEqual(len(chunks), len({c.chunk_id for c in chunks}))

    def test_every_chunk_has_a_citation(self):
        for chunk in chunk_corpus(CORPUS):
            self.assertTrue(chunk.citation)
            self.assertIn(".md", chunk.citation)


if __name__ == "__main__":
    unittest.main(verbosity=2)
