#!/usr/bin/env python3
"""Generate the evaluation report.

    IMPL=reference python evals/report.py
    IMPL=reference python evals/report.py --output reports/latest.md

The report answers five questions, in this order:

  1. How good is retrieval, on questions the tuning never saw?
  2. How often does the pipeline answer at all, and does the answer cite evidence?
  3. How often does it answer a question the corpus cannot answer?
  4. Where does it fail: by category, so the next hour of work is obvious.
  5. Is the difference between two configurations real, or inside the noise?

Every proportion carries a 95% Wilson interval. A number without one is a rumour.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.dense import DenseIndex, HybridIndex  # noqa: E402
from ragkit.embeddings import build_embedder  # noqa: E402
from ragkit.fielded import build_fielded_index  # noqa: E402
from ragkit.grounding import Grounder  # noqa: E402
from ragkit.ingest import chunk_corpus  # noqa: E402
from ragkit.lexical import build_index  # noqa: E402
from ragkit.pack import pack  # noqa: E402
from ragkit.stats import mcnemar_exact_p, required_sample_size, wilson_interval  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
QUESTIONS = ROOT / "evals" / "questions.jsonl"
UNANSWERABLE = ROOT / "evals" / "questions_unanswerable.jsonl"
BUDGET = 180
THRESHOLD = 0.5


def load(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def interval(successes: int, n: int) -> str:
    low, high = wilson_interval(successes, n)
    return f"{pct(low)}–{pct(high)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/latest.md")
    args = parser.parse_args()

    chunks = chunk_corpus(CORPUS)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    questions = load(QUESTIONS)
    unanswerable = load(UNANSWERABLE)

    lexical = build_index(chunks)
    fielded = build_fielded_index(chunks)
    dense = DenseIndex(chunks, build_embedder("char-ngram"))
    configs = {
        "lexical": lexical,
        "fielded": fielded,
        "dense": dense,
        "hybrid": HybridIndex(fielded, dense),
    }
    grounder = Grounder(chunks, min_sufficiency=THRESHOLD)

    gold = {
        q["id"]: {c.chunk_id for c in chunks if q["gold_phrase"].casefold() in c.text.casefold()}
        for q in questions
    }

    def retrieval_hits(index, subset):
        hits = {}
        for question in subset:
            results = index.search(question["question"], k=5)
            hits[question["id"]] = any(cid in gold[question["id"]] for cid, _ in results)
        return hits

    def end_to_end(index, subset):
        out = {}
        for question in subset:
            results = index.search(question["question"], k=5)
            docs = [by_id[cid] for cid, _ in results]
            packed = pack(question["question"], docs, budget=BUDGET)
            answer = grounder.answer(question["question"], packed.text)
            cited = next((c.chunk_id for c in packed.citations if c.number == answer.citation), None)
            out[question["id"]] = {
                "grounded": answer.grounded,
                "text": answer.text,
                "correct": answer.grounded
                and question["gold_phrase"].casefold() in answer.text.casefold(),
                "cited_right": answer.grounded and cited in gold[question["id"]],
                "citations": len(packed.citations),
            }
        return out

    dev = [q for q in questions if q["split"] == "dev"]
    test = [q for q in questions if q["split"] == "test"]

    retrieval = {
        name: {
            "dev": retrieval_hits(index, dev),
            "test": retrieval_hits(index, test),
        }
        for name, index in configs.items()
    }

    e2e = {name: end_to_end(index, test) for name, index in configs.items()}
    unanswerable_out = {}
    for name, index in configs.items():
        answered = 0
        for question in unanswerable:
            results = index.search(question["question"], k=5)
            docs = [by_id[cid] for cid, _ in results]
            packed = pack(question["question"], docs, budget=BUDGET)
            if grounder.answer(question["question"], packed.text).grounded:
                answered += 1
        unanswerable_out[name] = answered

    lines: list[str] = []
    add = lines.append

    add("# Evaluation report")
    add("")
    add(f"Generated {datetime.date.today().isoformat()} by `evals/report.py`.")
    add("")
    add(f"- Corpus: {len(chunks)} chunks from {len({c.doc_id for c in chunks})} documents")
    add(f"- Questions: {len(dev)} development, {len(test)} held out, "
        f"{len(unanswerable)} unanswerable")
    add(f"- Packing: {BUDGET} words, sufficiency threshold {THRESHOLD}")
    add("")
    add("Every proportion carries a 95% Wilson interval. Intervals this wide are a "
        "finding, not a nuisance: they say the eval set is too small to support a "
        "precision claim.")
    add("")

    add("## 1. Retrieval")
    add("")
    add("| configuration | dev recall@5 | held-out recall@5 |")
    add("| --- | --- | --- |")
    for name, splits in retrieval.items():
        dev_hits = sum(splits["dev"].values())
        test_hits = sum(splits["test"].values())
        add(
            f"| {name} | {dev_hits}/{len(dev)} ({pct(dev_hits / len(dev))}) "
            f"| {test_hits}/{len(test)} ({pct(test_hits / len(test))}, "
            f"95% CI {interval(test_hits, len(test))}) |"
        )
    add("")

    add("## 2. End to end, held out")
    add("")
    add("| configuration | answered | answer contains evidence | citation correct |")
    add("| --- | --- | --- | --- |")
    for name, outcomes in e2e.items():
        grounded = sum(1 for o in outcomes.values() if o["grounded"])
        correct = sum(1 for o in outcomes.values() if o["correct"])
        cited = sum(1 for o in outcomes.values() if o["cited_right"])
        denominator = grounded if grounded else 1
        add(
            f"| {name} | {grounded}/{len(test)} | {correct}/{len(test)} "
            f"| {cited}/{denominator} |"
        )
    add("")

    add("## 3. Unanswerable questions")
    add("")
    add("| configuration | false answers |")
    add("| --- | --- |")
    for name, answered in unanswerable_out.items():
        add(
            f"| {name} | {answered}/{len(unanswerable)} "
            f"({pct(answered / len(unanswerable))}, 95% CI "
            f"{interval(answered, len(unanswerable))}) |"
        )
    add("")

    add("## 4. Where hybrid fails, by category (held out)")
    add("")
    add("| category | questions | retrieval hit | correct answer |")
    add("| --- | --- | --- | --- |")
    categories = sorted({q["category"] for q in test})
    for category in categories:
        subset = [q for q in test if q["category"] == category]
        hits = sum(1 for q in subset if retrieval["hybrid"]["test"][q["id"]])
        correct = sum(1 for q in subset if e2e["hybrid"][q["id"]]["correct"])
        add(f"| {category} | {len(subset)} | {hits}/{len(subset)} | {correct}/{len(subset)} |")
    add("")

    add("## 5. Is the difference real? (hybrid vs fielded, held out)")
    add("")
    n01 = sum(
        1
        for q in test
        if retrieval["fielded"]["test"][q["id"]] and not retrieval["hybrid"]["test"][q["id"]]
    )
    n10 = sum(
        1
        for q in test
        if retrieval["hybrid"]["test"][q["id"]] and not retrieval["fielded"]["test"][q["id"]]
    )
    p_value = mcnemar_exact_p(n01, n10)
    add(f"- fielded only: {n01} questions")
    add(f"- hybrid only: {n10} questions")
    add(f"- exact McNemar p = {p_value:.3f}")
    verdict = "the difference is not distinguishable from noise" if p_value >= 0.05 else "the difference is unlikely to be noise"
    add(f"- verdict: {verdict}")
    add("")

    needed = required_sample_size(0.75, 0.10)
    add("## 6. Power")
    add("")
    add(f"To detect a ten-point improvement over a 75% baseline at 95% confidence and "
        f"80% power you would need roughly **{needed} questions** per configuration. "
        f"This report has {len(test)} held out.")
    add("")
    add("That is why the intervals are wide and why the significance test above can "
        "only rule out large differences. Grow the set before drawing fine conclusions.")

    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    print()
    print(f" report written to {path}")
    print(f" {len(dev)} dev, {len(test)} held out, {len(unanswerable)} unanswerable")
    print(f" hybrid held-out recall {sum(retrieval['hybrid']['test'].values())}/{len(test)}")
    print(f" McNemar p {p_value:.3f} ({n01} vs {n10} discordant)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
