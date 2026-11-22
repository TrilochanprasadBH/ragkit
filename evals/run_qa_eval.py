#!/usr/bin/env python3
"""End-to-end evaluation: retrieval, packing, gating, answering, citation.

    IMPL=reference python evals/run_qa_eval.py
    IMPL=reference python evals/run_qa_eval.py --sweep
    python evals/run_qa_eval.py --min-sufficiency 0.6 --show-failures 5

Two question sets are evaluated together.

  answerable    (evals/questions.jsonl) questions the corpus can answer
  unanswerable  (evals/questions_unanswerable.jsonl) questions about technology this
                corpus never mentions

That split is what makes the gate measurable. Without unanswerable questions, a system
that answers everything looks perfect.

Four numbers:

  false refusals   declined a question that had an answer. Annoying.
  false answers    answered a question that had no answer in the corpus. Dangerous.
  answer accuracy  on answerable questions, the answer contains the gold evidence.
  citation         the cited passage is one that contains the evidence.

The two error types are reported separately on purpose: they cost different things,
and a single accuracy number would hide which one you are buying.
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

from ragkit.baselines import pack_whole_chunks  # noqa: E402
from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import build_embedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.grounding import Grounder  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.pack import pack as pack_sentences  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
ANSWERABLE = ROOT / "evals" / "questions.jsonl"
UNANSWERABLE = ROOT / "evals" / "questions_unanswerable.jsonl"


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build():
    chunks = chunk_corpus(CORPUS)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    lexical = build_fielded_index(chunks)
    dense = DenseIndex(chunks, build_embedder("char-ngram"))
    return chunks, by_id, HybridIndex(lexical, dense)


def evaluate(packer, retriever, by_id, grounder, questions, budget, k=5):
    """Return per-question outcomes."""
    outcomes = []
    for question in questions:
        hits = retriever.search(question["question"], k=k)
        docs = [by_id[chunk_id] for chunk_id, _ in hits]
        packed = packer(question["question"], docs, budget=budget)
        result = grounder.answer(question["question"], packed.text)

        cited_chunk = next(
            (c for c in packed.citations if c.number == result.citation), None
        )
        outcomes.append(
            {
                "id": question["id"],
                "question": question["question"],
                "grounded": result.grounded,
                "text": result.text,
                "packed": packed,
                "cited": cited_chunk.chunk_id if cited_chunk else None,
            }
        )
    return outcomes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=180)
    parser.add_argument("--packer", choices=("sentences", "naive"), default="sentences")
    parser.add_argument("--min-sufficiency", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--show-failures", type=int, default=4)
    args = parser.parse_args()

    chunks, by_id, retriever = build()
    packer = pack_sentences if args.packer == "sentences" else pack_whole_chunks
    answerable = load(ANSWERABLE)
    unanswerable = load(UNANSWERABLE)

    gold = {
        q["id"]: {
            c.chunk_id for c in chunks if q["gold_phrase"].casefold() in c.text.casefold()
        }
        for q in answerable
    }

    def run(threshold: float):
        grounder = Grounder(chunks, min_sufficiency=threshold)
        good = evaluate(packer, retriever, by_id, grounder, answerable, args.budget)
        bad = evaluate(packer, retriever, by_id, grounder, unanswerable, args.budget)

        correct = sum(
            1
            for outcome in good
            if outcome["grounded"]
            and outcome["id"] in gold
            and any(
                q["gold_phrase"].casefold() in outcome["text"].casefold()
                for q in answerable
                if q["id"] == outcome["id"]
            )
        )
        answered = sum(1 for outcome in good if outcome["grounded"])
        cited = sum(
            1 for outcome in good if outcome["grounded"] and outcome["cited"] in gold[outcome["id"]]
        )
        false_answers = sum(1 for outcome in bad if outcome["grounded"])
        return {
            "correct": correct,
            "answered": answered,
            "cited": cited,
            "false_refusals": len(good) - answered,
            "false_answers": false_answers,
            "good": good,
            "bad": bad,
        }

    if args.sweep:
        print()
        print(f" {'threshold':>9} {'false refusals':>15} {'false answers':>14} {'answer accuracy':>16}")
        for threshold in (0.0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8):
            result = run(threshold)
            print(
                f" {threshold:>9.2f} "
                f"{result['false_refusals'] / len(answerable):>14.0%} "
                f"{result['false_answers'] / len(unanswerable):>13.0%} "
                f"{result['correct'] / len(answerable):>15.0%}"
            )
        print()
        return 0

    result = run(args.min_sufficiency)
    total_good, total_bad = len(answerable), len(unanswerable)

    print()
    print(f" implementation    {IMPL}")
    print(f" packer            {args.packer}, budget {args.budget} words")
    print(f" min sufficiency   {args.min_sufficiency}")
    print()
    print(f" answerable        {total_good} questions")
    print(f"   answered        {result['answered']}/{total_good}  {result['answered']/total_good:.1%}")
    print(f"   correct         {result['correct']}/{total_good}  {result['correct']/total_good:.1%}")
    print(f"   citation right  {result['cited']}/{result['answered'] or 1}")
    print(f"   false refusals  {result['false_refusals']}/{total_good}  "
          f"{result['false_refusals']/total_good:.1%}")
    print()
    print(f" unanswerable      {total_bad} questions")
    print(f"   refused         {total_bad - result['false_answers']}/{total_bad}  "
          f"{(total_bad - result['false_answers'])/total_bad:.1%}")
    print(f"   false answers   {result['false_answers']}/{total_bad}  "
          f"{result['false_answers']/total_bad:.1%}")

    if args.show_failures:
        missed = [o for o in result["good"] if not o["grounded"]][: args.show_failures]
        if missed:
            print("\n refused but answerable")
            for outcome in missed:
                print(f"   {outcome['id']}  {outcome['question'][:66]}")
        invented = [o for o in result["bad"] if o["grounded"]][: args.show_failures]
        if invented:
            print("\n answered but unanswerable")
            for outcome in invented:
                print(f"   {outcome['id']}  {outcome['text'][:66]}")

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
