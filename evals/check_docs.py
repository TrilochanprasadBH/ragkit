#!/usr/bin/env python3
"""Fail if a number in README.md no longer matches a run.

    python evals/check_docs.py
    IMPL=src python evals/check_docs.py        # against your workspace

The README quotes retrieval numbers in two places: the per-stage results table, and a
"Current numbers" table that is recomputed on the golden set as it exists today. The second one
is the one this script checks in full, because it is the one a reader will reproduce by running
`evals/run_evals.py` after cloning.

The stage-by-stage table is a historical record: each row is the number that stage achieved when
it was written, against the question set that existed then. Rewriting those would erase the
comparison the lessons are built on, so they are not checked against today's run. The lesson they
carry is checked by the stage tests.

What is checked:

1. every row of the "Current numbers" table: hits, recall@5 and MRR, for five configurations
2. stage 09's row in the results table: the two dev-split MRR gains it quotes
3. that no configuration name in the table is unknown — a silent miss is worse than a failure
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
IMPL = os.environ.get("IMPL", "reference")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.ingest import chunk_corpus  # noqa: E402

sys.path.insert(0, str(ROOT / "evals"))
from run_evals import build_configuration, load_questions, score_configuration  # noqa: E402

CORPUS = ROOT / "corpus" / "fastapi"
README = ROOT / "README.md"

# The first cell of each row in "Current numbers", and the configuration it names.
ROWS = {
    "lexical (stage 02)": "lexical",
    "fielded (stage 03)": "fielded",
    "hybrid (stage 04)": "hybrid",
    "dense + rerank": "rerank-dense",
    "hybrid + rerank": "rerank-hybrid",
}

ROW = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)/(\d+)\s*\|\s*(\d+\.\d)%\s*\|\s*(\d+\.\d{3})\s*\|")


def current_numbers_block(text: str) -> list[str]:
    start = text.index("### Current numbers")
    end = text.index("\n## ", start)
    return text[start:end].splitlines()


def main() -> int:
    chunks = chunk_corpus(CORPUS)
    questions = load_questions()
    text = README.read_text(encoding="utf-8")
    lines = current_numbers_block(text)

    problems: list[str] = []
    seen: set[str] = set()
    checked = 0

    for line in lines:
        match = ROW.match(line)
        if not match:
            continue
        name, hits, total, recall, mrr = match.groups()
        key = ROWS.get(name)
        if key is None:
            problems.append(f"unknown configuration in the table: {name!r}")
            continue
        seen.add(key)

        index = build_configuration(key, chunks)
        actual_hits, actual_recall, actual_mrr = score_configuration(index, chunks, questions)
        checked += 1

        if int(hits) != actual_hits:
            problems.append(f"{name}: table says {hits}/{total}, measured {actual_hits}/{len(questions)}")
        if abs(float(recall) - round(actual_recall * 100, 1)) > 0.05:
            problems.append(
                f"{name}: table says recall {recall}%, measured {actual_recall * 100:.1f}%"
            )
        if abs(float(mrr) - round(actual_mrr, 3)) > 0.0005:
            problems.append(f"{name}: table says MRR {mrr}, measured {actual_mrr:.3f}")

    missing = set(ROWS) - {name for name in ROWS if ROWS[name] in seen}
    if missing:
        problems.append(f"rows missing from the table: {sorted(missing)}")

    # Stage 09's row quotes two dev-split gains.
    dev = [question for question in questions if question["split"] == "dev"]
    gains = {}
    for label, base_key, reranked_key in (
        ("dense", "dense", "rerank-dense"),
        ("hybrid", "hybrid", "rerank-hybrid"),
    ):
        base = score_configuration(build_configuration(base_key, chunks), chunks, dev)[2]
        reranked = score_configuration(build_configuration(reranked_key, chunks), chunks, dev)[2]
        gains[label] = reranked - base

    row09 = next(
        (line for line in text.splitlines() if line.startswith("| 09 | reranking")), None
    )
    if row09 is None:
        problems.append("stage 09 has no row in the results table")
    else:
        quoted = re.findall(r"\+(\d+\.\d\d)", row09)
        if len(quoted) != 2:
            problems.append(f"stage 09 row should quote two gains, found {quoted}")
        else:
            for label, value in zip(("dense", "hybrid"), quoted):
                if abs(float(value) - round(gains[label], 2)) > 0.005:
                    problems.append(
                        f"stage 09 row says {value} on {label}, measured {gains[label]:+.3f}"
                    )

    if problems:
        print(f" {len(problems)} problem(s)")
        for problem in problems:
            print(f" - {problem}")
        return 1

    print(f" ok: {checked} configurations match the README tables")
    print(
        f" stage 09 gains: dense {gains['dense']:+.3f}, hybrid {gains['hybrid']:+.3f} "
        "(dev split)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
