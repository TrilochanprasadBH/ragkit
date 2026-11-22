"""Stage 05 tests: packing and citation.

    python stages/05_packing/test_stage05.py
    IMPL=reference python stages/05_packing/test_stage05.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.baselines import pack_whole_chunks  # noqa: E402
from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import CharNgramEmbedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.pack import pack  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"
BUDGET = 180
SENTENCE = re.compile(r"(?<=[.!?])\s+")
HEADER = re.compile(r"^\[(\d+)\]\s(.*)$", re.MULTILINE)


def build_retriever():
    chunks = chunk_corpus(CORPUS)
    lexical = build_fielded_index(chunks)
    dense = DenseIndex(chunks, CharNgramEmbedder())
    return chunks, HybridIndex(lexical, dense)


def passages(text: str) -> list[tuple[int, str, str]]:
    """Return (number, label, body) for each passage in a pack."""
    matches = list(HEADER.finditer(text))
    out = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        label, _, rest = body.partition("\n")
        out.append((int(match.group(1)), label.strip(), rest.strip()))
    return out


class TestPackStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.retriever = build_retriever()
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}
        hits = cls.retriever.search("how do I declare a required query parameter", k=5)
        cls.docs = [cls.by_id[chunk_id] for chunk_id, _ in hits]
        cls.pack = pack("how do I declare a required query parameter", cls.docs, budget=BUDGET)

    def test_returns_a_pack(self):
        self.assertTrue(self.pack.text.strip())
        self.assertGreater(len(self.pack.citations), 0)

    def test_citation_numbers_are_contiguous_from_one(self):
        numbers = [citation.number for citation in self.pack.citations]
        self.assertEqual(numbers, list(range(1, len(numbers) + 1)))

    def test_every_citation_maps_to_a_real_chunk(self):
        for citation in self.pack.citations:
            self.assertIn(citation.chunk_id, self.by_id)
            self.assertIn(".md", citation.label)

    def test_headers_appear_in_the_text(self):
        for citation in self.pack.citations:
            self.assertIn(f"[{citation.number}]", self.pack.text)

    def test_token_count_matches_the_text(self):
        self.assertEqual(self.pack.tokens, len(self.pack.text.split()))

    def test_within_budget(self):
        self.assertLessEqual(self.pack.tokens, BUDGET)


class TestSentenceIntegrity(unittest.TestCase):
    """Nothing may be invented, and no sentence may be cut in half."""

    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.retriever = build_retriever()
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}

    def test_selected_sentences_are_verbatim(self):
        questions = [
            json.loads(line)
            for line in QUESTIONS.read_text().splitlines()
            if line.strip()
        ]
        for question in questions:
            hits = self.retriever.search(question["question"], k=5)
            docs = [self.by_id[chunk_id] for chunk_id, _ in hits]
            packed = pack(question["question"], docs, budget=BUDGET)
            labels = {citation.number: citation.chunk_id for citation in packed.citations}
            for number, _label, body in passages(packed.text):
                source = self.by_id[labels[number]].text
                for sentence in SENTENCE.split(body):
                    sentence = sentence.strip()
                    if sentence:
                        self.assertIn(
                            sentence,
                            source,
                            f"{question['id']}: invented or truncated sentence in passage {number}",
                        )


class TestEmptyPacks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks, _ = build_retriever()

    def test_no_overlap_gives_an_empty_pack(self):
        packed = pack("zzzz qqqq wwww", self.chunks[:3], budget=BUDGET)
        self.assertTrue(packed.is_empty())
        self.assertEqual(packed.text, "")

    def test_no_chunks_gives_an_empty_pack(self):
        self.assertTrue(pack("anything at all", [], budget=BUDGET).is_empty())

    def test_tiny_budget_gives_an_empty_pack_rather_than_a_fragment(self):
        packed = pack("query parameters", self.chunks[:3], budget=2)
        self.assertLessEqual(packed.tokens, 2)


class TestAgainstTheBaseline(unittest.TestCase):
    """The claim of this stage, encoded as a test."""

    @classmethod
    def setUpClass(cls):
        cls.chunks, cls.retriever = build_retriever()
        cls.by_id = {chunk.chunk_id: chunk for chunk in cls.chunks}
        cls.questions = [
            json.loads(line)
            for line in QUESTIONS.read_text().splitlines()
            if line.strip()
        ]

    def _retention(self, packer) -> float:
        kept = 0
        for question in self.questions:
            hits = self.retriever.search(question["question"], k=5)
            docs = [self.by_id[chunk_id] for chunk_id, _ in hits]
            packed = packer(question["question"], docs, budget=BUDGET)
            if question["gold_phrase"].casefold() in packed.text.casefold():
                kept += 1
        return kept / len(self.questions)

    def test_beats_whole_chunk_packing_on_retention(self):
        ours = self._retention(pack)
        baseline = self._retention(pack_whole_chunks)
        self.assertGreaterEqual(
            ours, baseline, f"sentence packing {ours:.1%} vs whole chunks {baseline:.1%}"
        )

    def test_retention_floor(self):
        self.assertGreaterEqual(self._retention(pack), 0.65)


class TestDeterminism(unittest.TestCase):
    def test_same_input_same_output(self):
        chunks, retriever = build_retriever()
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        hits = retriever.search("request body", k=5)
        docs = [by_id[chunk_id] for chunk_id, _ in hits]
        first = pack("request body", docs, budget=BUDGET)
        second = pack("request body", docs, budget=BUDGET)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main(verbosity=2)
