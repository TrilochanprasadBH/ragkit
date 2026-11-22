# Stage 02 — Lexical search with BM25

```sh
python stages/02_lexical/test_stage02.py
```

Nineteen tests, all failing at first.

---

## What you will build

A search index over the 362 chunks from stage 01, and a scoring function that ranks
chunks for a question.

## Why this step is next

You now have chunks. You need a way to ask "which of these is likely to answer this
question". That is retrieval, and it is the part of RAG that decides whether the rest
of the system has any chance.

You are building the lexical version first, before any embeddings, for one reason:
**you cannot tell whether a vector database helped unless you have a number to beat.**

Most projects skip this step, add embeddings, and then have no idea whether the
quality comes from the embeddings, the chunking, or luck. This baseline takes an hour
and makes every later decision measurable.

## The idea

**Start with a simple model.** A document is a bag of words. A question is a bag of
words. Score a document by how many of the question's words it contains.

This fails immediately, for three reasons.

**Reason 1: some words are more informative than others.** "the" appears in every
chunk. "corsmiddleware" appears in two. Matching "corsmiddleware" tells you almost
everything; matching "the" tells you nothing.

The fix is inverse document frequency:

```
idf(term) = log(1 + (N - df + 0.5) / (df + 0.5))
```

`N` is the number of chunks, `df` is how many contain the term. A term in every chunk
gets a score near zero. A term in one chunk gets a large score.

The `+1` and the `0.5` are not decoration. They keep the value positive for very
common terms, so that a long question cannot accumulate negative points and rank a
correct chunk below a wrong one.

**Reason 2: a term appearing four times is not four times as important.** Term
frequency saturates. The tenth occurrence of "request" adds much less evidence than
the first.

BM25 handles this with:

```
tf_component = (tf * (k1 + 1)) / (tf + k1 * (...))
```

with `k1 = 1.5`. As `tf` grows, the component approaches `k1 + 1` rather than growing
without bound.

**Reason 3: long chunks win by accident.** A chunk with 400 words contains more of
everything, including your query terms, so it outranks a precise 40-word chunk.

The `b` term normalises by length:

```
length_norm = 1 - b + b * (dl / avgdl)
```

with `b = 0.75`. A chunk of average length is unaffected. A long chunk is penalised.

Put together:

```
score = sum over query terms of
        idf(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
```

That is BM25. It has been the default for keyword search for thirty years because it
is cheap, it needs no training, and it is hard to beat on exact-match queries.

**One decision specific to this project: index the heading path too.** A chunk under
`Query Parameters > Defaults` says nothing about query parameters in its body. The
breadcrumb supplies that vocabulary, so the searchable text is the heading path plus
the chunk body. This is the cheapest version of the "contextual chunk" idea covered
in the field notes below, and it costs nothing because stage 01 already captured the
headings.

## Field notes: what production teams do

Read these before you tune anything. They are the difference between "I implemented
BM25" and "I know why this is the default in production".

**Anthropic, Contextual Retrieval (September 2024).** They measured retrieval failure
across codebases, fiction, and papers. Plain embeddings failed to place the right
chunk in the top 20 about 5.7% of the time. Adding a short generated context to each
chunk before embedding cut failures by 35%. Adding **BM25 over the same enriched
chunks** cut failures by 49%. Adding a reranker cut them by 67%. Their own Pass@10
numbers moved from 87.15% (baseline) to 93.21% (hybrid) to 95.26% (with reranking).

*What it means for you:* lexical search is not the legacy option you replace with
vectors. It is a permanent half of a production retrieval system, because it finds
exact terms that embeddings smooth away. You are building that half first, on purpose.

**Oracle, production RAG evaluation (July 2026).** Their guidance is methodological:
evaluate retrieval and generation separately, because a system can retrieve the right
evidence and still produce an unsupported answer; compare keyword, vector, and hybrid
against the same ground-truth questions before choosing a default; test exact
identifiers, citations, abstention, freshness, and tenant isolation alongside average
recall. They also warn against declaring a universal winner.

*What it means for you:* the golden set you just loaded is the instrument Oracle is
describing. Stage 07 turns it into a report. Do not tune anything before that.

**AWS Well-Architected, Agentic AI Lens.** Their guidance on retrieval: track latency
per stage (embedding, search, rerank) with explicit budgets; use hybrid retrieval and
reranking where the corpus and query mix justify them, "rather than stacked by
default"; reformulate queries before retrieval so recall survives vocabulary
mismatch.

*What it means for you:* do not add a reranker because a blog post said so. Add it
because your failure list says ranking is the problem. You will make that decision in
stage 04 with evidence.

**A replication worth knowing (October 2026).** A team testing contextual retrieval
on Japanese corpora found that generating context headers did **not** beat simply
cutting documents at their headings for dense retrieval, while reranking again gave
the largest single improvement. Their fusion results also depended on how questions
were phrased: BM25-only won on lexically close questions, dense-only on paraphrased
ones, and equal-weight fusion followed by reranking was best on both.

*What it means for you:* published numbers are arguments to test, not conclusions to
copy. Two lessons survive replication: **chunk at structure**, and **rerank last**.

**A scale check.** Anthropic's own guidance is that if the entire knowledge base fits
in the context window (roughly 200,000 tokens, about 500 pages), retrieval may be
unnecessary. This corpus is about 45,000 words, well under that. We build retrieval
anyway, because the eval is the point and because the same code runs unchanged at a
hundred times the size. Write down that you knew this, so that in an interview you
can say when you would *not* build RAG.

## Do this

1. Read `src/ragkit/lexical.py`. The docstrings are the contract.
2. Implement `tokenize`: lowercase, keep letters and digits, drop stop words.
3. Implement `BM25Index.__init__`: store term counts, lengths, document frequencies.
4. Implement `idf`.
5. Implement `search`: score every chunk, sort, return the top k.
6. Run the tests.
7. Run the eval and write the number down.

```sh
IMPL=reference python evals/run_evals.py    # the number you are aiming at
python evals/run_evals.py                   # yours
```

## Check yourself

```sh
python stages/02_lexical/test_stage02.py
python evals/run_evals.py --show-failures 6
```

Expected: all tests pass, recall@5 around 66%, MRR around 0.53.

Then run the reference and compare. If your number is much lower, the bug is almost
always in one of three places: stop words not removed, headings not indexed, or
sorting ascending instead of descending.

## If you are stuck

**Hint 1.** Build the index in one pass: for each chunk, tokenise the heading path and
the body together, store the counts, and increment a document-frequency counter for
each distinct term.

**Hint 2.** `search` is a loop over distinct query terms, then over chunks that contain
them. Do not score chunks that contain no query term.

**Hint 3.** Sort with a tuple key, `(-score, chunk_id)`, so ties break deterministically.
Tests compare exact output on repeated runs.

## The failures you should expect

Six of the twenty-four questions fail. Read them; they are the syllabus for the next
two stages.

- **"How do I return HTML instead of JSON?"** The chunk says `HTMLResponse`. The query
  says `HTML`. They do not match as tokens. → tokenisation and expansion, stage 03.
- **"custom header"** against a chunk that says `headers`. → stemming, stage 03.
- **"benchmarks misleading"** against a chunk that says "compared as equivalent". The
  words share no tokens at all. → synonyms, then embeddings, stages 03 and 04.
- **"deployment strategy"** against "the best way to deploy". → same problem.

A lexical system fails on vocabulary mismatch. That is not a bug in your code; it is
the reason the next two stages exist. Write the failures down before you fix them.

## Reference

`reference/ragkit/lexical.py`. Compare it after your tests pass, and check your
parameter choices against theirs: `k1 = 1.5`, `b = 0.75`. Then change `b` to 0 and
re-run the eval. Seeing the number drop is worth more than reading about length
normalisation.

## What a senior would notice

**Whether the parameters were tuned or inherited.** `k1 = 1.5` and `b = 0.75` are the
standard defaults. On this corpus, does `b = 0.5` do better? If you never tried,
you do not know whether the default is right for your data, and the honest answer is
"I used the default and did not tune it", not "I know it is optimal".

**Whether the index is rebuilt or incrementally updated.** This implementation
rebuilds on every run, which is fine for 362 chunks and wrong for a live corpus. Know
where the boundary is.

**Whether the headings decision was deliberate.** Indexing the breadcrumb is a real
choice with a real cost: it can create false matches when a heading word appears in
many documents. Try turning it off and re-running the eval. If the number barely
moves, the decision was cheap; if it drops hard, you have learned why contextual
chunks matter, in your own corpus, with your own number.
