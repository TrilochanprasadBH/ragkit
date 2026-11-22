# ragkit

A documentation assistant that answers questions about a real documentation set and
cites the exact sections it used.

This repository is a **course and a product at the same time**. You build it in eight
stages. At the end you have a working service, an eval report with real numbers, and
a deployable container.

The corpus is the FastAPI documentation (MIT licensed), vendored in `corpus/fastapi/`
so everything runs offline.

---

## What it does

```
question  ->  retrieve candidate passages  ->  pack them into a budget
          ->  generate an answer  ->  verify every claim against the sources
          ->  answer with citations, or refuse
```

## Why this project exists

Retrieval-augmented generation is the most common shape of AI product work. It is
also where most implementations are quietly wrong: the retrieval is untuned, nobody
measured it, and the model fills gaps with plausible text.

This project forces the measurement. Every stage adds one capability and one number.

## Requirements

- Python 3.10 or newer
- No third-party packages for stages 1 to 6. Stage 7 adds the eval runner, stage 8
  adds the web service. Both are listed with setup instructions when you reach them.

## Quick start

```sh
# 1. see where you are
python scripts/status.py

# 2. read the course
open GUIDE.md

# 3. start stage 1
python stages/01_ingest/test_stage01.py
```

The tests fail at first. That is the starting state.

## How to work in this repo

- **You write code in `src/ragkit/`.** Every function there starts with
  `raise NotImplementedError` and a comment describing the contract.
- **`reference/ragkit/` contains a complete implementation.** Read it only after you
  have tried, or after you have been stuck for 30 minutes on one stage.
- **`stages/NN_name/` contains the lesson and the tests for that stage.** The tests
  are the specification.
- Run `IMPL=reference python stages/01_ingest/test_stage01.py` to confirm the tests
  themselves are correct.

## Results

Filled in as each stage completes. Numbers are produced by `evals/run_evals.py`.

| stage | capability | measure | result |
| --- | --- | --- | --- |
| 01 | ingest and chunk | 22 documents to 362 citable chunks | done |
| 02 | lexical retrieval | recall@5, MRR | **66.7%**, 0.536 |
| 03 | text analysis | recall@5, MRR | **83.3%**, 0.668 |
| 04 | dense retrieval and fusion | recall@5, MRR | **87.5%**, 0.688 |
| 05 | packing and citation | evidence retained, budget compliance | **70.8%**, 100% |
| 06 | grounding and refusal | false answers 25% at 21% false refusals | **curve measured** |
| 07 | eval suite | full report with confidence intervals | not yet measured |
| 08 | service | p95 latency, cost per query | not yet measured |

## Repository layout

```
corpus/fastapi/        22 real documentation pages, vendored
src/ragkit/            you implement here
reference/ragkit/      a complete implementation, for comparison
stages/NN_name/        one lesson and one test suite per stage
evals/                 golden questions and the eval runner
scripts/               helper commands
GUIDE.md               the course
FIELD-NOTES.md         what production teams did, and what we did not build
```

## Licence

Code in this repository is yours to use. The corpus in `corpus/fastapi/` is from the
FastAPI project, MIT licensed, and the licence file is included at
`corpus/fastapi/LICENSE`.
