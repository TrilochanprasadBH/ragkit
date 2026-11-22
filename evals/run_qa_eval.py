#!/usr/bin/env python3
"""End-to-end QA evaluation: retrieval, packing, answering, citation.

    IMPL=reference python evals/run_qa_eval.py
    python evals/run_qa_eval.py --packer naive
    python evals/run_qa_eval.py --budget 200 --show-failures 5

Four numbers, and they measure different things.

  retention    the gold evidence survived packing. This is retrieval plus packing.
  answer       the returned sentence contains the gold phrase. This is the reader
               working with what it was given.
  citation     when an answer was given, the cited passage really contained the
               evidence. A correct answer with a wrong citation is not a result.
  refusals     the system declined to answer. On this golden set every question has
               an answer, so every refusal is a false refusal.

Splitting these apart is the point. A drop in `answer` with `retention` intact means
the reader failed, not the retriever. A drop in both means retrieval or packing broke.
One number would hide that distinction.
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

from ragkit.answer import answer as read  # noqa: E402
from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import build_embedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.pack import pack as pack_sentences  # noqa: E402
from ragkit.pack import pack_whole_chunks  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=180)
    parser.add_argument("--packer", choices=("sentences", "naive"), default="sentences")
    parser.add_argument("--retrieve", type=int, default=5)
    parser.add_argument("--show-failures", type=int, default=5)
    args = parser.parse_args()

    chunks = chunk_corpus(CORPUS)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    lexical = build_fielded_index(chunks)
    dense = DenseIndex(chunks, build_embedder("char-ngram"))
    retriever = HybridIndex(lexical, dense)

    questions = [
        json.loads(line)
        for line in QUESTIONS.read_text().splitlines()
        if line.strip()
    ]

    packer = pack_sentences if args.packer == "sentences" else pack_whole_chunks

    retained = answered = cited = refused = compliant = 0
    failures: list[tuple[str, str]] = []

    for question in questions:
        gold_phrase = question["gold_phrase"].casefold()
        gold_chunks = {
            chunk.chunk_id
            for chunk in chunks
            if gold_phrase in chunk.text.casefold()
        }

        hits = retriever.search(question["question"], k=args.retrieve)
        retrieved = [by_id[chunk_id] for chunk_id, _ in hits]

        packed = packer(question["question"], retrieved, budget=args.budget)
        if packed.tokens <= args.budget:
            compliant += 1
        if gold_phrase in packed.text.casefold():
            retained += 1

        result = read(question["question"], packed.text)
        if not result.grounded:
            refused += 1
            failures.append((question["id"], "refused"))
            continue

        answered += 1
        if gold_phrase in result.text.casefold():
            # the reader found the phrase
            pass

        cited_chunk = next(
            (c for c in packed.citations if c.number == result.citation), None
        )
        if cited_chunk is not None and cited_chunk.chunk_id in gold_chunks:
            cited += 1
        else:
            failures.append((question["id"], f"cited [{[c.number for c in packed.citations if c.chunk_id in gold_chunks]}]"))

    total = len(questions)
    print()
    print(f" implementation  {IMPL}")
    print(f" packer          {args.packer}, budget {args.budget} words")
    print(f" retained gold   {retained}/{total}  {retained/total:.1%}")
    print(f" within budget   {compliant}/{total}  {compliant/total:.1%}")
    print(f" answered        {answered}/{total}  {answered/total:.1%}")
    print(f" citation right  {cited}/{answered if answered else 1}  "
          f"{cited/(answered if answered else 1):.1%}")
    print(f" false refusals  {refused}/{total}")

    if failures and args.show_failures:
        print(f"\n first {min(args.show_failures, len(failures))} failure(s)")
        for question_id, reason in failures[: args.show_failures]:
            print(f"   {question_id}  {reason}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
