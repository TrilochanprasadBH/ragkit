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


def gold_chunks(chunks, question) -> set[str]:
    """Every chunk whose text or heading path contains the gold phrase.

    Labels are phrases rather than chunk ids so they survive re-chunking. This is the single
    definition of "the gold chunk" in the repository; `evals/check_docs.py` and the stage tests
    use it too, so a change here cannot silently move a number in one place and not another.
    """
    phrase = question["gold_phrase"].casefold()
    return {
        chunk.chunk_id
        for chunk in chunks
        if phrase in chunk.text.casefold()
        or phrase in " ".join(chunk.headings).casefold()
    }


def build_configuration(name: str, chunks, embedder: str = "char-ngram"):
    """Build one named retrieval configuration. The only place that knows how."""
    if name == "lexical":
        from ragkit.lexical import build_index

        return build_index(chunks)
    if name == "fielded":
        from ragkit.fielded import build_fielded_index

        return build_fielded_index(chunks)

    from ragkit.dense import DenseIndex, HybridIndex
    from ragkit.embeddings import build_embedder
    from ragkit.fielded import build_fielded_index

    dense = DenseIndex(chunks, build_embedder(embedder))
    if name == "dense":
        return dense
    hybrid = HybridIndex(build_fielded_index(chunks), dense)
    if name == "hybrid":
        return hybrid

    from ragkit.rerank import RerankIndex

    return RerankIndex(hybrid if name == "rerank-hybrid" else dense, chunks)


def score_configuration(index, chunks, questions, k: int = 5) -> tuple[int, float, float]:
    """(hits, recall@k, MRR) over the questions whose gold phrase exists in the corpus."""
    hits = 0
    reciprocal = 0.0
    scored = 0
    for question in questions:
        gold = gold_chunks(chunks, question)
        if not gold:
            continue
        scored += 1
        results = index.search(question["question"], k=k)
        rank = next(
            (
                position
                for position, (chunk_id, _) in enumerate(results, start=1)
                if chunk_id in gold
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal += 1.0 / rank
    return hits, hits / scored if scored else 0.0, reciprocal / scored if scored else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--index",
        choices=("lexical", "fielded", "dense", "hybrid", "rerank-dense", "rerank-hybrid"),
        default="lexical",
        help=(
            "lexical is stage 02's BM25; fielded is stage 03's, with pluggable tokenisation; "
            "rerank-dense and rerank-hybrid add stage 09's reranker to that first stage"
        ),
    )
    parser.add_argument("--show-failures", type=int, default=5)
    parser.add_argument(
        "--embedder",
        choices=("char-ngram", "hashing", "remote"),
        default="char-ngram",
        help="embedder used by the dense and hybrid indexes",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=180,
        help=(
            "maximum chunk size in words; the knob stage 01's measurement section sweeps. "
            "180 is the default the lessons were written against"
        ),
    )
    args = parser.parse_args()

    chunks = chunk_corpus(CORPUS, max_tokens=args.chunk_words)
    index = build_configuration(args.index, chunks, args.embedder)
    questions = load_questions()

    by_phrase = {q["id"]: q for q in questions}
    hits = 0
    reciprocal = 0.0
    scored = 0
    broken: list[str] = []
    failures: list[tuple[str, int | None]] = []

    for question in questions:
        gold_ids = gold_chunks(chunks, question)
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
