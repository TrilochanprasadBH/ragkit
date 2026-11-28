# Stage 01 — Document loading and heading-aware chunking

```sh
python stages/01_ingest/test_stage01.py
```

Twenty tests, all failing at first.

---

## What you will build

A loader that reads every Markdown file in the corpus, and a chunker that splits each
document into citable passages of at most 180 words.

## Why this step is next

Retrieval cannot be better than the pieces it searches. If a chunk contains three
unrelated topics, every query about one of them retrieves noise. If a chunk is a
single sentence, it retrieves precisely and answers nothing.

Chunking is the decision that sets the ceiling for everything after it.

## The idea

**A document is not a bag of words.** It has structure: headings, sections,
paragraphs. That structure is the author telling you what belongs together. Ignoring
it and splitting by character count throws away free information.

**A heading is a path, not a label.** The heading `Defaults` means nothing on its
own. Under `Query Parameters` it means something specific. So each chunk carries the
full chain of headings above it, from the document title down. That chain is what
makes a citation useful: `tutorial/query-params.md > Query Parameters > Defaults`
tells a reader exactly where to look.

**Two sizes, two failure modes.**

- Chunks too small lose the surrounding context. A sentence like "It defaults to
  `None`." is unanswerable without the sentence before it.
- Chunks too large dilute the match. If a 900-word chunk is retrieved because one
  sentence matched, the model receives 890 irrelevant words and pays for them.

180 words is a starting point, not a law. The measurement below compares chunk sizes
empirically. It can only be run once a retriever exists, because the question "is this
chunk the right size" is really "does the retriever find it and rank it well" — so run
it after stage 04 and treat every earlier choice as provisional.

**Overlap exists because meaning crosses boundaries.** A sentence that defines a term
often begins in one paragraph and ends in the next. Carrying the last few sentences
of a chunk into the next one means a chunk whose start is missing the setup still
contains it.

Overlap is not free: it duplicates text, which inflates the index and the cost. Use
enough to cover typical sentence length, not more.

**Stable identifiers matter more than they look.** A citation is a promise. If chunk
ids change between runs, then a stored answer can no longer be traced to its source.
So ids are derived from the document path and the position, and documents are loaded
in sorted order.

## The measurement

Chunking can only be scored through a retriever, so this experiment runs after stage
04, over the 32 golden questions, with the flag the eval runner now carries:

```sh
for w in 90 120 180 240 300 360; do
  printf "%-3s words  " $w
  IMPL=reference python evals/run_evals.py --index hybrid --chunk-words $w \
    | grep -E "recall|MRR" | tr '\n' ' '
  echo
done
```

The run on 2026-11-28 produced this:

| chunk words | lexical recall@5 | lexical MRR | hybrid recall@5 | hybrid MRR |
| --- | --- | --- | --- | --- |
| 90 | 78.1% | 0.575 | 87.5% | **0.707** |
| 120 | 75.0% | 0.545 | 87.5% | 0.658 |
| **180 (default)** | 75.0% | 0.577 | 87.5% | 0.672 |
| 240 | 75.0% | 0.556 | 87.5% | 0.639 |
| 300 | 75.0% | 0.571 | 84.4% | 0.615 |
| 360 | 75.0% | 0.556 | 87.5% | 0.618 |

Read it honestly, in three lines.

**Recall barely moves.** Hybrid holds 87.5% at every size except 300 words, where it
loses one question net: bigger chunks *fix* the HTTPS question q15 (more surrounding
context for the paraphrase to match) and *break* q11 and q29, whose gold passages are
now diluted among more vocabulary. That is the two failure modes above fighting each
other, and the net is a loss. Every gold phrase still resolves to a chunk at every
size — nothing was split in half — so the entire movement is ranking, not truncation.

**Small chunks buy MRR.** At 90 words, hybrid MRR is 0.707 against 0.672 at 180: a
precise chunk outranks a dilute one, the same dilution argument stage 02 makes about
BM25 length normalisation, arriving from the other end. The difference is 0.035, about
one question's worth on 32 questions (1/32 = 0.031), and this measurement cannot tell
you it would survive a different corpus.

**The default stays 180, with the reason stated rather than hidden.** The packing stage
is designed around a 180-word budget, and a 90-word chunk answers nothing alone — the
"too small" failure mode is real and MRR does not measure it. If you keep 90 after
reading this, write down why in `NOTES.md`; the sweep is one command and every corpus
you are handed deserves its own run.

One detail makes the sweep possible at all: golden labels are phrases, not chunk ids
(stage 02). Re-chunking changes every id, and a phrase label survives it.

## Field notes

From `FIELD-NOTES.md`:

- **Anthropic's contextual retrieval** had a model write 50–100 tokens of context into
  every chunk before indexing. The heading breadcrumb this stage captures is the free
  version of the same idea: context the document already carried, with no model call.
  The paid version is in the "deliberately did not build" table, with the corpus size
  that would justify it.
- **AWS Well-Architected** recommends chunking that follows document structure rather
  than fixed character counts. That is the design here: sections first, sentences
  second, and the word limit only as a cap.
- **The 2026 replication** of contextual retrieval found that cutting documents at
  their headings matched generated context headers for retrieval quality. Structure
  you can parse for free is hard to beat, and it is what this stage relies on.

## Do this

1. Read `src/ragkit/ingest.py`. The docstrings are the contract.
2. Implement `estimate_tokens`. One line is enough.
3. Implement `load_documents`. Load files, compute ids, extract the title.
4. Implement `chunk_document`. Split at headings, then pack sentences with overlap.
5. Implement `chunk_corpus`.
6. Run the tests.

Two details that will bite you:

- Headings in this corpus look like `## Defaults { #defaults }`. Strip the
  `{ #anchor }` part or your breadcrumbs will contain it.
- When a section is short, the whole section is one chunk. Do not force a minimum
  size; a two-sentence section is a valid chunk.

## Check yourself

```sh
python stages/01_ingest/test_stage01.py
```

All tests pass. Then:

```sh
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from ragkit.ingest import chunk_corpus
chunks = chunk_corpus(Path('corpus/fastapi'))
print(len(chunks), 'chunks')
print(chunks[10].citation)
print(chunks[10].text[:200])
"
```

Read that output. The citation should look like a path a human could follow, and the
text should read as a coherent passage.

## If you are stuck

**Hint 1.** Split the text into lines. When a line starts with `#`, you have found a
heading: close the current section and start a new one with the updated breadcrumb.

**Hint 2.** To pack sentences, keep a running word count. Before adding a sentence,
check whether it would exceed the limit. If it would, close the chunk and start a
new one seeded with the tail of the previous chunk.

**Hint 3.** The breadcrumb behaves like a stack. A level-2 heading replaces everything
from level 2 down, but keeps level 1. `breadcrumb[: level - 1] + [title]` does it.

## Reference

`reference/ragkit/ingest.py` is the complete version. Look at it after your tests
pass, and compare decisions rather than copying code. If yours differs and both pass
the tests, yours is fine; note the difference in `NOTES.md`.

## What a senior would notice

Three things separate a chunker that works from one that is merely correct.

**Whether the ids survive re-ingestion.** Re-run the loader twice and diff the output.
If chunk ids move, citations in stored answers are already broken. Sorting the file
list before loading is what prevents this, and it is one line that people forget.

**Whether overlap is measured or guessed.** Forty words is a guess. The measured
answer is "the 95th percentile of sentence length in this corpus". Run that number
once and your overlap stops being arbitrary.

**Whether chunk size is a decision or a default.** The measurement above compares six
sizes on the same questions. The default stayed 180 and the reason is written down; the
sweep is one command, and every corpus you are handed deserves its own run before you
inherit either the number or the reason.
