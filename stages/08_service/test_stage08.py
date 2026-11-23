"""Stage 08 tests: the service, its instrumentation, and its HTTP surface.

    python stages/08_service/test_stage08.py
    IMPL=reference python stages/08_service/test_stage08.py
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import CharNgramEmbedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.server import serve  # noqa: E402
from ragkit.service import Service  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
GOOD_QUESTION = "what are query parameters in a URL"
BAD_QUESTION = "how do I mount a Ceph volume into a Nomad job"


def build(usage_path=None) -> Service:
    chunks = chunk_corpus(CORPUS)
    embedder = CharNgramEmbedder()
    retriever = HybridIndex(build_fielded_index(chunks), DenseIndex(chunks, embedder))
    return Service(
        chunks=chunks,
        retriever=retriever,
        embedder_name=embedder.name,
        embedder_dim=embedder.dim,
        embed_price_per_mtok=0.02,
        usage_path=usage_path,
    )


class TestAsk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = build()

    def test_response_shape(self):
        response = self.service.ask(GOOD_QUESTION)
        for key in (
            "question", "answer", "refused", "citations", "cited",
            "timings", "cost_usd", "embedded_tokens", "pack_tokens",
        ):
            self.assertIn(key, response)

    def test_answerable_question_is_answered_with_citations(self):
        response = self.service.ask(GOOD_QUESTION)
        self.assertFalse(response["refused"])
        self.assertGreater(len(response["citations"]), 0)
        self.assertIn(".md", response["citations"][0]["label"])

    def test_unanswerable_question_is_refused(self):
        response = self.service.ask(BAD_QUESTION)
        self.assertTrue(response["refused"])

    def test_adjacent_topic_is_not_refused(self):
        # The documented limitation from stage 06: a question about a technology the
        # corpus mentions only in passing scores high enough to pass the gate. The
        # system answers from related material. This is a false answer, and the test
        # records it rather than hiding it.
        response = self.service.ask("How do I use WebSockets with Spring Boot?")
        self.assertFalse(response["refused"])

    def test_timings_are_present_and_ordered(self):
        timings = self.service.ask(GOOD_QUESTION)["timings"]
        for key in ("retrieve_ms", "pack_ms", "answer_ms", "total_ms"):
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0.0)
        self.assertGreaterEqual(timings["total_ms"], timings["retrieve_ms"])

    def test_cost_is_computed_from_tokens(self):
        response = self.service.ask(GOOD_QUESTION)
        expected = len(GOOD_QUESTION.split()) / 1_000_000 * 0.02
        self.assertAlmostEqual(response["cost_usd"], expected, places=8)


class TestUsageLog(unittest.TestCase):
    def test_one_line_per_request(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "usage.jsonl"
            service = build(usage_path=path)
            service.ask(GOOD_QUESTION)
            service.ask(BAD_QUESTION)
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            entry = json.loads(lines[0])
            for key in ("at", "question_sha", "refused", "citations", "timings", "cost_usd"):
                self.assertIn(key, entry)
            self.assertEqual(len(entry["question_sha"]), 12)

    def test_logging_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "usage.jsonl"
            service = build(usage_path=path)
            service.ask(GOOD_QUESTION, log=False)
            self.assertFalse(path.exists())


class TestManifest(unittest.TestCase):
    def test_manifest_contents(self):
        service = build()
        manifest = service.manifest()
        self.assertEqual(manifest["embedder"], "char-ngram")
        self.assertEqual(manifest["dimension"], 512)
        self.assertGreater(manifest["chunks"], 300)
        self.assertEqual(manifest["documents"], 22)

    def test_manifest_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "nested" / "index.json"
            build().write_manifest(path)
            data = json.loads(path.read_text())
            self.assertEqual(data["embedder"], "char-ngram")


class TestHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = build()
        cls.server = serve(cls.service, port=0)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def _url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def test_health(self):
        with urllib.request.urlopen(self._url("/health")) as response:
            payload = json.loads(response.read())
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["embedder"], "char-ngram")

    def test_ask(self):
        body = json.dumps({"question": GOOD_QUESTION}).encode()
        request = urllib.request.Request(
            self._url("/ask"), data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read())
        self.assertIn("answer", payload)
        self.assertIn("timings", payload)

    def test_bad_request(self):
        body = json.dumps({"question": ""}).encode()
        request = urllib.request.Request(self._url("/ask"), data=body)
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertEqual(context.exception.code, 400)


class TestLatencyBudget(unittest.TestCase):
    def test_p95_is_inside_the_budget(self):
        result = subprocess.run(
            [sys.executable, "scripts/bench.py", "--budget-ms", "300"],
            cwd=ROOT,
            env={**os.environ, "IMPL": IMPL},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout[-500:] + result.stderr[-500:])


if __name__ == "__main__":
    unittest.main(verbosity=2)
