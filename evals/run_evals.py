#!/usr/bin/env python3
"""Measure retrieval quality against the golden set.

    python evals/run_evals.py
    IMPL=reference python evals/run_evals.py
    python evals/run_evals.py --k 10 --show-failures

Two metrics are reported.

  recall@k   Of the questions whose gold passage exists in the corpus, the fraction
             where at least one gold passage appears in the top k results. This
             measures whether the answer was found at all.

  MRR        Mean reciprocal rank. If the gold passage is ranked 1st you score 1.0,
             2nd 0.5, 3rd 0.33, and 0 if it is outside the top k. This measures
             ranking quality, which matters because the model reads the top of the
             list first and context space is limited.

A question is scored against every chunk that contains its `gold_phrase`, so the
labels survive re-chunking. If a phrase is missing from the corpus the question is
reported as broken rather than silently scored as a miss.
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

from ragkit.ingest import chunk_corpus  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


def load_questions() -> list[dict]:
    return [
        json.loads(line)
        for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--index",
        choices=("lexical", "fielded", "dense", "hybrid"),
        default="lexical",
        help="lexical is stage 02's BM25; fielded is stage 03's, with pluggable tokenisation",
    )
    parser.add_argument("--show-failures", type=int, default=5)
    parser.add_argument(
        "--embedder",
        choices=("char-ngram", "hashing", "remote"),
        default="char-ngram",
        help="embedder used by the dense and hybrid indexes",
    )
    args = parser.parse_args()

    chunks = chunk_corpus(CORPUS)

    if args.index == "fielded":
        from ragkit.fielded import build_fielded_index

        index = build_fielded_index(chunks)
    elif args.index in ("dense", "hybrid"):
        from ragkit.dense import DenseIndex, HybridIndex
        from ragkit.embeddings import build_embedder
        from ragkit.fielded import build_fielded_index

        embedder = build_embedder(args.embedder)
        dense = DenseIndex(chunks, embedder)
        if args.index == "dense":
            index = dense
        else:
            index = HybridIndex(build_fielded_index(chunks), dense)
    else:
        from ragkit.lexical import build_index

        index = build_index(chunks)
    questions = load_questions()

    by_phrase = {q["id"]: q for q in questions}
    hits = 0
    reciprocal = 0.0
    scored = 0
    broken: list[str] = []
    failures: list[tuple[str, int | None]] = []

    for question in questions:
        phrase = question["gold_phrase"].casefold()
        gold_ids = {
            chunk.chunk_id
            for chunk in chunks
            if phrase in chunk.text.casefold() or phrase in " ".join(chunk.headings).casefold()
        }
        if not gold_ids:
            broken.append(question["id"])
            continue

        scored += 1
        results = index.search(question["question"], k=args.k)
        rank = next(
            (position for position, (chunk_id, _) in enumerate(results, start=1)
             if chunk_id in gold_ids),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal += 1.0 / rank
        else:
            failures.append((question["id"], None))
            continue

        if rank > 3:
            failures.append((question["id"], rank))

    recall = hits / scored if scored else 0.0
    mrr = reciprocal / scored if scored else 0.0

    print()
    print(f" implementation  {IMPL}")
    print(f" index           {args.index}")
    print(f" chunks          {len(chunks)}")
    print(f" questions       {scored} scored, {len(broken)} broken labels")
    print(f" recall@{args.k}       {recall:.1%}")
    print(f" MRR             {mrr:.3f}")

    if broken:
        print(f"\n broken labels (phrase not found in corpus): {', '.join(broken)}")

    if failures and args.show_failures:
        print(f"\n worst {min(args.show_failures, len(failures))} question(s)")
        for question_id, rank in failures[: args.show_failures]:
            text = by_phrase[question_id]["question"]
            where = "not in top k" if rank is None else f"ranked {rank}"
            print(f"   {question_id}  {where:<12} {text[:70]}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
