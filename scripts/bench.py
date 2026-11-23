#!/usr/bin/env python3
"""Replay the golden questions and report latency per stage.

    IMPL=reference python scripts/bench.py
    python scripts/bench.py --budget-ms 300

Percentiles, not averages. A mean hides the request that took 900ms, and that is the
one your users remember. Budgets are set on p95 because that is roughly one request in
twenty.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import CharNgramEmbedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.service import Service  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def build_service(price: float = 0.02) -> Service:
    chunks = chunk_corpus(CORPUS)
    embedder = CharNgramEmbedder()
    retriever = HybridIndex(build_fielded_index(chunks), DenseIndex(chunks, embedder))
    return Service(
        chunks=chunks,
        retriever=retriever,
        embedder_name=embedder.name,
        embedder_dim=embedder.dim,
        embed_price_per_mtok=price,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-ms", type=float, default=300.0)
    args = parser.parse_args()

    service = build_service()
    questions = [
        json.loads(line)
        for line in QUESTIONS.read_text().splitlines()
        if line.strip()
    ]

    stages = {"retrieve_ms": [], "pack_ms": [], "answer_ms": [], "total_ms": []}
    cost = 0.0

    for question in questions:
        response = service.ask(question["question"], log=False)
        for key in stages:
            stages[key].append(response["timings"][key])
        cost += response["cost_usd"]

    print()
    print(f" implementation {IMPL}")
    print(f" requests       {len(questions)}")
    print()
    print(f" {'stage':<14}{'p50':>9}{'p95':>9}{'max':>9}")
    for key, values in stages.items():
        print(
            f" {key:<14}{percentile(values, 0.5):>8.1f}ms"
            f"{percentile(values, 0.95):>8.1f}ms{max(values):>8.1f}ms"
        )
    print()
    print(f" cost/request   ${cost / len(questions):.8f}")
    print(f" cost/1M        ${cost / len(questions) * 1_000_000:.2f}")

    p95 = percentile(stages["total_ms"], 0.95)
    verdict = "PASS" if p95 <= args.budget_ms else "FAIL"
    print(f" p95 budget     {args.budget_ms:.0f}ms  [{verdict}]")
    print()
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
