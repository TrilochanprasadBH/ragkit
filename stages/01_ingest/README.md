"""Stage 01 — loading documents and splitting them into chunks.

Run everything from the repository root:

    python stages/01_ingest/test_stage01.py

Every test fails at first. That is the starting state.

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

180 words is a starting point, not a law. Stage 04 will let you compare chunk sizes
empirically, which is the only correct way to choose one.

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

**Whether chunk size is a decision or a default.** In stage 04 you will compare 120,
180 and 300 words on the same questions. Until then, write down why you chose 180,
so that the later measurement has something to contradict.
