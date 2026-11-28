# Stage 09 — Reranking, and when it earns its cost

```sh
python stages/09_rerank/test_stage09.py
IMPL=reference python stages/09_rerank/test_stage09.py
```

---

## What you will build

A cross-encoder stand-in that scores a (query, chunk) pair jointly, a wrapper that re-orders a
first stage's candidates, and the measurement that decides whether the wrapper belongs in the
pipeline. The answer on this corpus is **no**, and the stage is mostly about why.

## Why this step is next

Stage 08 ended with a service at p95 3.3ms. That number is the reason to ask this question: a
reranker is the one addition that can multiply it, because it scores every candidate separately
rather than comparing two precomputed vectors.

The naive version of this stage adds a reranker and reports the improvement. There are two
problems with that. The improvement on a strong first stage is within noise, and the second-stage
cost is not. So the stage is framed as a question — *does this first stage leave anything for a
reranker to fix?* — and the answer is measured before anything is shipped.

## The idea

**A bi-encoder embeds the query and the document separately.** Stage 04's dense index does this:
two independent vectors, one dot product. It is fast enough to score the whole corpus, and it
cannot notice that "the flag I turn on" and "enabled or not" are the same idea, because neither
vector knows about the other.

**A cross-encoder reads the pair together** and emits one score. Every token can attend to every
token on the other side, which is why it is stronger and why it costs 10 to 100 times more: every
candidate is its own forward pass. That asymmetry is the whole design constraint. You cannot
cross-encode 362 chunks per query; you cross-encode the 20 that the first stage already found.

So the pipeline is:

```
retrieve wide  ->  rescore the pairs  ->  blend with the first stage  ->  cut to k
```

**This file is not a transformer.** It is a stand-in that reads the pair with four signals a real
cross-encoder learns:

| signal | what it measures | why |
| --- | --- | --- |
| coverage | share of the query's terms in the chunk, weighted by IDF | a rare term is the answer's fingerprint |
| phrase | share of the query's adjacent pairs appearing adjacently | "rate limit" is not "rate" and "limit" |
| proximity | how tightly the matched terms cluster | "a … b" scores below "a b" |
| heading | share of the query's terms in the heading path | the heading says what the chunk is about |

Each is in [0, 1] and the weights are fixed constants rather than fitted, because fitting four
weights on twenty-three questions is how you build a reranker that works on your dev set and
nowhere else.

**The blend matters.** `weight = 1.0` discards the first stage's opinion and drops hybrid recall
from 90.9% to 77.3%. `weight = 0.6` keeps enough of it that a chunk fusion was confident about
survives a mediocre pair score. At `weight = 0.0` the wrapper is the identity, which is a property
worth testing rather than hoping for.

## The measurement

Development split, 22 questions, first stage retrieved to depth 20:

| first stage | recall@5 | MRR | reranked recall@5 | reranked MRR | MRR gain |
| --- | --- | --- | --- | --- | --- |
| dense alone | 72.7% | 0.505 | 77.3% | 0.638 | **+0.133** |
| lexical alone | 81.8% | 0.623 | 81.8% | 0.648 | **+0.025** |
| hybrid | 90.9% | 0.674 | 81.8% | 0.683 | **+0.009** |

**The gain is a function of the first stage, not of the reranker.** A weak first stage leaves a
lot to repair, and the reranker repairs it: +0.133 MRR on dense. A strong first stage leaves
almost nothing, and the reranker costs recall while buying one hundredth of an MRR point, because
the chunks it demotes are the ones fusion got right.

That is the answer to "should we add a reranker", and you can have it before you write any
reranking code: **look at the first stage's MRR.** At 0.67 on a five-document window, there is
0.33 of theoretical headroom and almost none of it is reachable by re-ordering.

### And even the big gain does not survive a test

The +0.133 on dense is a mean over 22 questions. Per question:

| first stage | questions improved | worsened | unchanged | sign test |
| --- | --- | --- | --- | --- |
| dense | 6 | 2 | 14 | p = 0.29 |
| hybrid | 3 | 3 | 16 | p = 1.00 |

Six wins against two losses is not evidence, it is a coin landing six times out of eight. And
the held-out split, ten questions, does not confirm either dev number: dense's gain flips sign,
hybrid's grows sixfold:

| configuration | dev MRR gain | held-out MRR gain |
| --- | --- | --- |
| dense + rerank | +0.133 | **−0.016** |
| hybrid + rerank | +0.009 | **+0.053** |

One question on ten is ten points of recall and up to 0.1 of MRR, so neither number means
anything. The honest conclusion is the one the stage ships with: **the mechanism is real and the
evidence is not sufficient.** Do not put the reranker in the pipeline on this corpus; do keep the
measurement, because the next corpus will have different headroom.

## Field notes

**Nogueira and Cho, "Passage Re-ranking with BERT" (2019).** They re-implemented BERT as a
query-passage cross-encoder and re-ranked a first-stage list with it. The abstract reports the
system as state of the art on TREC-CAR and the top entry on the MS MARCO passage leaderboard,
"outperforming the previous state of the art by 27% (relative) in MRR@10".

**What it means.** The gain came from reading the pair with a pretrained language model, not from
re-ordering heuristics. That is the part this stage cannot reproduce offline: our stand-in reads
the pair with four lexical counts, so it can only notice what a term-frequency model can notice.
It is enough to reorder a weak first stage and not enough to beat a strong one.

**In this project.** The stand-in gains 0.133 MRR on the weakest first stage and 0.009 on the
strongest, and neither gain is significant at n = 22. A real cross-encoder would be measured the
same way and the eval set would still be the bottleneck. We did not build the transformer, and
this stage does not claim its numbers transfer.

## Do this

1. Implement `TermStats.from_chunks` and `weight`. Check that a term in one chunk outweighs a term
   in every chunk.
2. Implement `CrossEncoder.score` with the four signals. Run the encoder tests: adjacency beats
   scattering, proximity beats distance, headings count.
3. Implement `RerankIndex`. Get `weight = 0.0` to be the identity before you try anything else;
   it is the only configuration whose output you can predict by hand.
4. Run the measured tests. `test_reranking_a_weak_first_stage_earns_its_cost` should pass and
   `test_reranking_a_strong_first_stage_does_not` should pass with it — the point is that both
   are true at once.
5. Change `weight` to 1.0 and `depth` to 5, and record what happens to recall. Put it back.
6. Read `test_the_held_out_split_disagrees_with_dev` and decide what you would need to conclude
   anything. Write the number down.

## Check yourself

```sh
IMPL=reference python stages/09_rerank/test_stage09.py
# Ran 19 tests ... OK

python evals/check_docs.py
# ok: 5 configurations match the README tables
```

You should be able to say:

- "A cross-encoder reads the query and the chunk together, which is why it is stronger and 10 to
  100 times more expensive, which is why it only ever sees the first stage's top 20."
- "Reranking dense gained 0.133 MRR and reranking hybrid gained 0.009, because the gain is the
  first stage's headroom and hybrid had already used it up."
- "Six wins and two losses on 22 questions is p = 0.29, and the held-out split flips the sign,
  so the correct decision here is not to ship it."

## If you are stuck

1. `weight = 0.0` must reproduce the first stage exactly. If it does not, your rank score is not
   strictly decreasing in position — check that you use `1 / (position + 1)` and start at 1.
2. If `test_adjacent_terms_beat_scattered_terms` fails, your bigram list is probably built from
   the chunk instead of from the query. Pairs come from the query; the chunk only supplies the
   lookup set.
3. If the measured tests fail by a large margin, you are probably scoring chunks with an encoder
   built over a different chunk list. The encoder's term statistics and the wrapper's candidates
   must come from the same corpus.

## What a senior would notice

**The right time to ask is before you build it.** The headroom rule means the decision costs
nothing: run the first stage, look at MRR, and if it is already in the high 0.6s or better on a
five-chunk window (hybrid here: 0.674), the reranker has nothing to win — the measured gain was
nine thousandths of a point. Teams that add one anyway usually measure a small gain on
dev, ship it, and pay 20 forward passes per query for it.

**A negative result needs the same rigour as a positive one.** This stage reports a gain, then
tests it, then reports that the test does not support the gain. That is the difference between
"we measured it and it did not help" and "we did not measure it and it felt better".

**The bottleneck was the eval set, not the reranker.** Twenty-two dev questions cannot resolve
a hundredth of an MRR point, and ten held-out questions cannot resolve anything. The next
improvement to this pipeline is more labelled questions, not a better second stage — and that is
the same conclusion stage 07 reached from a different direction.
