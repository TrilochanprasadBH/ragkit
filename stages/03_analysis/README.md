# Stage 03 — Text analysis

```sh
python stages/03_analysis/test_stage03.py
```

---

## What you will build

Three functions that turn raw text into index terms: identifier splitting, stemming,
and tokenisation. Plus an index that keeps headings separate from body text.

## Why this step is next

Stage 02 ended at **recall@5 of 66.7%**. The failures had one thing in common: the
question and the document used different strings for the same idea.

| question says | document says |
| --- | --- |
| HTML | `HTMLResponse` |
| header | `headers` |
| deployment | `Deployments` |

None of those are different concepts. They are different spellings. A retriever that
cannot see that will keep failing on questions that a human finds trivial.

## The idea

**Identifiers hide their words.** `HTMLResponse` is one token to a whitespace
tokeniser, and the word "HTML" is invisible inside it. Code corpora are full of these.

The fix is to split on case and digit boundaries before lowercasing:

```
HTMLResponse         -> HTML | Response
OAuth2PasswordBearer -> OAuth | 2 | Password | Bearer
```

One rule needs care. `OAuth` looks like an acronym followed by a word, so a naive
splitter produces `O` | `Auth`. Requiring **two or more** leading capitals before a
capitalised word fixes it, and keeps `HTTPServer` splitting correctly.

**Plurals and tenses fragment the vocabulary.** `headers` and `header` are the same
word to a reader. Stripping a trailing `s` collapses them. A handful of suffix rules
covers most English:

```
-ies -> -y      policies  -> policy
-sses -> -ss    classes   -> class
-s              headers   -> header   (but not "analysis", "status", "class")
-ing            deploying -> deploy
-ed             deployed  -> deploy
```

This stemmer is deliberately small. It does not handle `deployment` → `deploy`,
because the rules that would (`-ment`) also turn `document` into `docu`. Stage 04
solves that class of problem properly, with meaning rather than spelling.

**A split identifier should contribute both halves and the whole.** This one is not
obvious and it cost a measurable amount in development.

Splitting `HTMLResponse` is a win: a question about "HTML" now matches. Splitting
`FastAPI` is a loss: it produces `fast` and `api`, and `api` appears in almost every
chunk in this corpus. A query mentioning FastAPI then matches everything and ranks
nothing.

The fix is to index the whole term *as well as* its parts. `fastapi` carries the
strong signal; `fast` and `api` carry the weak one. Precision comes from the whole,
recall from the parts.

## The experiment that did not work

Stage 03 was planned to include **field weighting**: score terms in the heading path
more heavily than terms in the body, because a heading says what a chunk is *about*.

I implemented it (`heading_weight`, applied to both term frequency and length) and
measured it. The result:

| heading_weight | recall@5 | MRR |
| --- | --- | --- |
| 1.0 (no boost) | 83.3% | 0.668 |
| 1.25 | 83.3% | 0.665 |
| 1.50 | 83.3% | 0.658 |
| 1.75 | 83.3% | 0.678 |
| 2.00 | 83.3% | 0.678 |
| 2.50 | 83.3% | 0.642 |

**Recall does not move at all, and the MRR differences are noise.** Boosting headings
did not help on this corpus.

There is a reason, and it is visible in the failure list. Anthropic's documentation
headings are short and topical (`Query Parameters`, `Defaults`). Boosting them is
harmless. But this corpus also has headings like `Examples of Replication Tools and
Strategies`, which match a question about strategy without containing the answer. The
boost promotes a wrong chunk, and one wrong chunk in the top five is exactly the kind
of error that shows up as a lower MRR rather than lower recall.

So the default is `heading_weight = 1.0`: headings are indexed, but not boosted. The
parameter stays in the code because you should be able to reproduce the table above
and disagree with me.

**What actually produced the 16-point gain was tokenisation, not field weighting.** If
you take one thing from this stage, take the habit: build the thing, measure it,
report the null result, and keep looking.

## Field notes

`FIELD-NOTES.md` at the repository root collects what production teams published, what
we took from it, and what we deliberately did not build. Two entries are relevant
here:

- **Anthropic's Contextual Retrieval** measured that adding a short generated context
  to each chunk, indexed by *both* embeddings and BM25, cut retrieval failures by 49%.
  Our heading-path indexing is a free approximation of the same idea: it is the
  context you already have, without a model call. If you want the real thing, the
  next step would be one small model call per chunk at ingest time.
- **A 2026 replication** found the generated context did not beat plain
  heading-aligned chunking for dense retrieval on their data. Which is why the
  heading experiment above is worth running yourself rather than assuming.

## Do this

1. Implement `split_identifier` in `src/ragkit/analysis.py`.
2. Implement `stem`.
3. Implement `tokenize`, including the whole-run rule.
4. Implement `fielded.py`: two fields, weighted term frequency, weighted length.
5. Run the tests.
6. Run the eval against both indexes and compare.

```sh
IMPL=reference python evals/run_evals.py --index lexical
IMPL=reference python evals/run_evals.py --index fielded
python evals/run_evals.py --index fielded
```

## Check yourself

```sh
python stages/03_analysis/test_stage03.py
python evals/run_evals.py --index fielded --show-failures 4
```

Expected: all tests pass, recall@5 around 83%, MRR around 0.67.

Then try the experiment yourself:

```sh
python -c "
import sys; sys.path.insert(0, 'reference')
from pathlib import Path
from ragkit.ingest import chunk_corpus
from ragkit.fielded import build_fielded_index
chunks = chunk_corpus(Path('corpus/fastapi'))
for w in (1.0, 2.0, 3.0):
    print(w, build_fielded_index(chunks, heading_weight=w).search('deployment strategy', k=3))
"
```

## If you are stuck

**Hint 1.** Do the split before lowercasing. The case boundary is the signal, and
`casefold()` destroys it.

**Hint 2.** Apply the four regex substitutions in sequence, inserting a space at each
boundary, then split on whitespace.

**Hint 3.** For `tokenize`, build the parts first, then decide whether to prepend the
whole run. The condition is: more than one part, and no underscore in the original.

## What is still failing, and why

Four questions survive:

- **"When should I use a normal def instead of async def?"** The answer chunk is in
  `async.md`, but so is half the vocabulary. Lexical matching cannot tell which
  passage answers the question.
- **"How do I think about choosing a deployment strategy?"** Other chunks in the same
  document mention "strategies" more often.
- **"Is HTTPS just a flag I turn on?"** The document says "enabled", not "flag".
- **"What does the simplest possible FastAPI file look like?"** "Simplest" and
  "possible" are not in the document.

Every one of these is a **vocabulary gap**, not a tokenisation gap. No amount of
suffix stripping fixes "flag" versus "enabled". That is the boundary of lexical
retrieval, and it is where stage 04 begins.

## Reference

`reference/ragkit/analysis.py` and `reference/ragkit/fielded.py`.

## What a senior would notice

**Whether you reported the null result.** The field-weighting experiment did not work.
The professional move is to write that down and keep the parameter, not to quietly
raise the weight until the number moves on one seed and call it an improvement.

**Whether your stemmer risks merging unrelated words.** `-ment` would fix
"deployment" and break "document". Every stemming rule trades recall against merging
distinct concepts; the conservative choice is to under-stem and let stage 04 handle
meaning.

**Whether tokenisation is a per-corpus decision.** Identifier splitting is right for
a code documentation corpus and wrong for, say, a corpus of legal prose where
`Section 4.2(a)` is a meaningful identifier. There is no universal tokeniser, only a
tokeniser matched to the corpus.
