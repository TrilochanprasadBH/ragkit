# Notes

Your working log. Three lines per stage. This is the file that turns nine exercises
into nine lessons.

Do not tidy it up. Write what actually happened, including the wrong turns.

---

## Template

```md
### Stage NN — name

What I changed

What moved the number

What surprised me

What I would do differently
```

---

## The stages

### Stage 01 — ingest and chunk

What I changed

I vendored 22 FastAPI documentation pages (MIT) as the offline corpus and wrote a
heading-aware chunker: split at headings, pack sentences to at most 180 words with
overlap, carry the full heading path on every chunk. The 22 documents became 362
citable chunks.

What moved the number

The number this stage owns is 362 chunks from 22 documents, each citable as a path a
person can follow, like `tutorial/query-params.md > Query Parameters > Defaults`.
Chunk ids are derived from the document path and position, and files load in sorted
order, so a second run reproduces every id.

What surprised me

How much of the contract lived in one-line details. Sorting the file list before
loading is what keeps a stored citation traceable on the next run, and it is the line
I would have forgotten. Also: a two-sentence section is a valid chunk, and forcing a
minimum size would have broken it.

What I would do differently

Measure the overlap instead of guessing it. Forty words is a guess; the measured
answer is the 95th percentile of sentence length in this corpus, and I never ran it.
Write down why 180 words now, so the later measurement has something to contradict.

### Stage 02 — lexical search

What I changed

A BM25 index over the 362 chunks, with k1 = 1.5 and b = 0.75, and the heading path
indexed alongside the body. I also wrote the golden set — 24 questions, each with a
verifiable gold phrase — and the eval runner that reports recall@k and MRR.

What moved the number

Baseline recall@5 66.7% (16 of 24), MRR 0.536. A quality floor test verifies that
every gold phrase resolves to a chunk, so a failure is a retrieval failure and not a
labelling failure. The runner names the misses, and the named ones are vocabulary
mismatch, not code bugs.

What surprised me

The failures taught more than the passes. The question says HTML and the chunk says
`HTMLResponse`; the question says header and the chunk says `headers`. The code was
correct the whole time, and reading the failure list told me what stages 03 and 04
had to fix before I wrote either of them.

What I would do differently

Say plainly that k1 and b are inherited defaults, not tuned ones. I never tried
b = 0.5 on this corpus, so I can report the standard value and cannot claim it is
right for this data.

### Stage 03 — text analysis

What I changed

Identifier splitting (camel case, digits, snake case), a deliberately conservative
stemmer, and a tokeniser that indexes the whole run alongside its parts. Plus a
fielded index with a heading weight — the feature I planned this stage around.

What moved the number

recall@5 went from 66.7% to 83.3% on the 24 questions (16 of 24 to 20 of 24), MRR
0.536 to 0.668, and all of the gain came from tokenisation. Heading weighting did
nothing: recall sat at 83.3% at every weight from 1.25 to 2.50, and MRR wandered
between 0.642 and 0.678 with no trend. The default stayed 1.0 and the null result
went into the lesson.

What surprised me

The headline idea was the dud. Boosting headings promotes chunks like `Examples of
Replication Tools and Strategies`, which match a question about strategy without
containing the answer. And splitting identifiers cuts both ways: `FastAPI` becomes
`fast` and `api`, and `api` is in nearly every chunk. Hence the whole-run rule —
precision from the whole term, recall from the parts.

What I would do differently

Keep the null result and keep the parameter, so anyone can reproduce the table and
disagree with me. And under-stem on purpose: `-ment` would fix "deployment" and
break "document".

### Stage 04 — dense retrieval and rank fusion

What I changed

Three embedders behind one interface — char-ngram (offline), hashing (offline, the
negative control), remote (needs a key) — a dense index scoring by cosine, and
reciprocal rank fusion. The sweep chose candidates = 10, k = 10.

What moved the number

Hybrid recall@5 87.5% (21 of 24), MRR 0.688, against 83.3% (20 of 24) for lexical.
Dense alone was 79.2% (19 of 24), worse than lexical. The negative control: fusing
the weak hashing embedder scored 75.0% (18 of 24), worse than not fusing at all. The
conventional k = 60 lost at depth — at 30 candidates it gave 83.3% recall and MRR
0.646, where k = 10 gave 91.7% and 0.665.

What surprised me

Adding a bad retriever made the system worse than lexical alone. Rank fusion treats
noise as evidence, so "we added vector search" is a claim about architecture, not
about quality. And the quoted k = 60 comes from web search at a different scale; it
is not a constant.

What I would do differently

Run the negative control before the first claim about embeddings, not after it. And
re-sweep the fusion constant on every new corpus, because the published default lost
here.

### Stage 05 — packing

What I changed

A sentence-level packer: score each sentence by query overlap normalised by length,
buy the best value per word until the budget is full, never cut a sentence, present
in source order. Citations are assigned by the packer, not generated, so they cannot
be hallucinated. The stand-in reader in answer.py picks the sentence with the most
query overlap, and that is what citation precision measures.

What moved the number

Retention 70.8% (17 of 24) inside a 180-word budget against 58.3% (14 of 24) for
whole chunks, and budget compliance 100% against 95.8% (23 of 24). Citation precision
fell from 54.2% (13 of 24) to 45.8% (11 of 24).

What surprised me

Retention rose and citations fell, and the fix I tried did nothing: scoring the
reader semantically with the stage 04 embedding gave 46%, 54%, 50% at three budgets.
In a pack assembled around a question every sentence is topically related, so
similarity cannot separate "the sentence that answers it" from "a sentence about it".
That located the fault in generation, not packing, and made it stage 06's problem.

What I would do differently

Split retention from citation precision from the first measurement; one blended
number would have hidden the trade. And treat the budget as a variable, not a
constant — at 350 words the gap between the packers nearly closes.

### Stage 06 — grounding

What I changed

An IDF-weighted sufficiency gate over the pack, which refuses below a threshold. I
also added 12 unanswerable questions about technologies the corpus never mentions
(Terraform, Kafka, Helm, PostgreSQL, and eight more), because a system that answers
everything scores perfectly on answerable questions alone.

What moved the number

The trade-off curve over 32 answerable and 12 unanswerable questions. At the default
threshold 0.50: false refusals 22% (7 of 32), false answers 17% (2 of 12) — 25 of 32
answered, 10 of 12 refused. No point on the curve makes both errors small. These are
post-fix numbers; stage 08 found a bug in this gate and I re-measured.

What surprised me

The two false answers that survive are the same case: the corpus mentions the topic
briefly inside another page, and a lexical gate cannot tell "mentions WebSockets"
from "explains WebSockets with Spring Boot". No threshold fixes that. Only
understanding the question would.

What I would do differently

Write the unanswerable set before building the gate, not alongside it. Without those
12 questions, every threshold looks free.

### Stage 07 — eval suite

What I changed

No new retrieval. Three statistical functions — the Wilson interval, exact McNemar,
required sample size — and a report generator that separates retrieval from
end-to-end. Questions got a category and a dev/test split: 32 total, 23 development,
9 held out. (The Corrections entry below re-splits this by document.)

What moved the number

The first report run: hybrid 96% dev against 67% held out (Wilson interval 35–88% on
the nine), and the ranking flipped — dense beat both hybrid and fielded on the
held-out split. Hybrid against fielded there was one discordant question, exact
McNemar p = 1.0. The power calculation sized the whole problem: detecting a ten-point
improvement over a 75% baseline needs roughly 250 questions per configuration, and
the set has 32.

What surprised me

Every parameter chosen in stages 03 and 04 — heading weight, candidate depth, fusion
constant — was chosen while looking at questions that are now the dev split. The gap
was not a bug; it was what tuning on a set does. And separating retrieval from
generation showed two failures at once: retrieval found evidence for 6 of the 9
held-out questions, and only 2 of those answers contained it.

What I would do differently

Grow the question set before touching the retriever again. The eval is the bottleneck,
not the code, and the next sprint is writing questions, which is unglamorous.

### Stage 08 — service

What I changed

Service.ask composing retrieval, packing, the gate and the answer, with per-stage
timings, cost per request, an append-only usage log (a hash of the question, the
refusal, the citations, the timings), and an index manifest that records the
embedder. A stdlib HTTP server with /health and /ask, a Dockerfile, and a benchmark
script in front of it.

What moved the number

p95 3.3ms end to end, of which retrieval is 2.4ms, and $0.00000019 per request —
$0.19 per million. Both numbers are stated with what they exclude: the generation
call, which the stand-in reader replaces. A real model call costs hundreds of
milliseconds and a fraction of a cent; what transfers is the instrumentation, not the
3.3ms.

What surprised me

The first realistic request found a bug that 44 questions had missed. "How do I set
a response header" was refused while "response header" was answered, because `how`,
`do` and `i` counted toward sufficiency and supplied over half of the denominator.
Fixed with the GRAMMATICAL stop list, and the stage 06 curve re-measured — the
numbers above are post-fix.

What I would do differently

Put the pipeline behind HTTP sooner. The bug was invisible in the question set and
visible in the first curl, because the questions I had written did not sound like the
ones a user asks.

### Stage 09 — reranking

What I changed

A cross-encoder stand-in reading four signals (coverage, phrase, proximity, heading),
and a RerankIndex that rescores a first stage's top 20 and blends 60/40 with it.
Measured, and not shipped.

What moved the number

Development split, 22 questions, first stage to depth 20: dense +rerank MRR gain
+0.133, lexical +0.025, hybrid +0.009. None of it is significant — dense improved 6
questions, worsened 2, left 14 unchanged, sign test p = 0.29; hybrid went 3/3/16,
p = 1.00. The held-out split flips the sign on dense: +0.133 dev against −0.016 held
out.

What surprised me

Even the big gain does not survive a test. The gain is the first stage's headroom:
the weak stage leaves room to repair and the reranker repairs it, the strong one
leaves nothing, and the chunks it demotes are the ones fusion got right. The decision
was available before writing any code — look at the first stage's MRR.

What I would do differently

Ask the headroom question first. Above roughly 0.7 MRR on a five-chunk window there
is nothing for a second stage to win, and 20 forward passes per query is a high price
for a hundredth of a point. The next improvement is more labelled questions; stage 07
said the same thing from the other direction.

---

## Corrections

### 2026-11-28 — the golden set was re-split by document

What I changed

The dev/test split in evals/questions.jsonl was reassigned so no document appears in
both splits. The original split was by question: 23 development, 9 held out, and 5 of
the 9 held-out questions targeted documents that also had development questions. One
pair (q09 test, q27 dev) were paraphrases with the identical gold phrase. q27 was
rewritten to target a different passage in the same document ("without having to read
it all into memory at once"), and the splits became 22 development, 10 held out, with
no document spanning.

What moved the number

Every split-dependent number was re-measured. Fielded: 91% dev, 70% held out.
Hybrid: 91% dev, 80% held out. Dense: 73% dev, 90% held out — dense still leads on
held out, now on documents the tuning never saw. Whole-set numbers did not move: the
rewritten question is retrieved at the same rank by every configuration, and
evals/check_docs.py still passes unchanged. The stage 06 trade-off curve moved at two
thresholds only (0.65: 47%→44% false refusals; 0.80: 59%→56%); the default operating
point is unchanged at 22% false refusals, 17% false answers.

What surprised me

The contaminated split and the clean split tell the same story: dev numbers
optimistic, dense leading held out. The finding survived the fix, which is the only
reason it is safe to report.

What I would do differently

Split by document the first time. A held-out question about a document you tuned on
is not held out, and no interval fixes that.
