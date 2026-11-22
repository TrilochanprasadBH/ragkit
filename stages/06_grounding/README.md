# Stage 06 — The sufficiency gate

```sh
python stages/06_grounding/test_stage06.py
```

---

## What you will build

A check that decides whether the packed material actually supports an answer, and
refuses when it does not.

## Why this step is next

Two problems meet here.

**A retriever always returns something.** Ask this corpus about Kafka and you will get
five chunks about middleware and background tasks, because those are the nearest things
in the vocabulary. The retriever is not malfunctioning; it has no way to return nothing.

**A reader always finds a sentence.** Stage 05 showed this precisely: evidence was
retained 70% of the time, but the citation was only right about half the time, because
any sentence in a topical pack overlaps the question somewhat. A system that answers
every question looks confident and is wrong a third of the time.

So the pipeline needs a gate: a check that runs before answering and is allowed to say
no.

## The idea

**The question you are answering is not "did I find text". It is "does this text
contain the answer".** Those are different questions and they need different evidence.

**Weight the question's terms by rarity.** Most of a question is filler: "how", "do",
"configure", "with". If you measure coverage by counting words, a question about
Terraform scores 82% against a pack about FastAPI, because "configure" and "with" are
present.

The signal is in the rare terms. If the question says "terraform" and the corpus never
does, that single absence is decisive. So weight each term by inverse document
frequency:

```
sufficiency = sum of idf over question terms present in the pack
            / sum of idf over all question terms
```

A term absent from the corpus gets the maximum idf, so it dominates the denominator. A
question about an absent technology scores near zero even when most of its words are
common English.

**The threshold is a product decision, not an accuracy setting.** Two errors are
possible and they are not symmetric:

- **False refusal.** You decline a question you could have answered. The user
  rephrases, or leaves. Cost: annoyance.
- **False answer.** You answer from material that does not support the answer. The user
  believes it and acts on it. Cost: trust, and whatever the wrong action cost.

No threshold removes both. You choose which one you are willing to pay for.

## The measurement

24 answerable questions and 12 unanswerable ones (about Terraform, Kafka, Helm,
PostgreSQL, Nginx, Elasticsearch, Prometheus, Rails, Kubernetes, Spring Boot, Ceph,
Nomad). Same retriever, same packer, budget 180 words.

| threshold | false refusals | false answers |
| --- | --- | --- |
| 0.00 (no gate) | 0% | 100% |
| 0.30 | 4% | 92% |
| 0.40 | 4% | 50% |
| **0.50 (default)** | **21%** | **25%** |
| 0.55 | 33% | 17% |
| 0.60 | 38% | 8% |
| 0.65 | 50% | 0% |
| 0.80 | 75% | 0% |

**There is no point on this curve where both errors are small.** That is the result,
and it is more useful than a single accuracy number would have been.

At the default (0.50): 19 of 24 answerable questions answered, 9 of 12 unanswerable
ones refused, and 3 questions answered that should not have been. Read the three:
OAuth2 scopes, WebSockets, and a deployment question. The corpus mentions all three
topics, briefly, inside other pages. A lexical gate cannot tell "this page mentions
OAuth2" from "this page explains the OAuth2 device flow".

**What would do better.** A gate that understands the question, not just its
vocabulary. That is a model call, and it is exactly the decision-model pattern from
PostHog: a cheap classifier in front of an expensive generator. We are offline, so the
gate is lexical, and the lesson is the curve rather than a number.

## Field notes

From `FIELD-NOTES.md`:

- **PostHog, decision models.** They set the PII gate at 0.4 and the Slack-answer gate
  at 0.85, and state the rule plainly: set the threshold by which mistake costs you
  more. A false negative stored personal data; a false positive annoyed a user. Our
  curve is the same argument with different numbers.
- **Oracle.** Evaluate retrieval and generation separately. This stage adds the third
  thing to separate: the *decision to answer*. A system can retrieve well, generate
  plausibly, and still be wrong because it answered when it should have declined.
- **AWS Well-Architected.** Explicit budgets per stage. Refusal is a stage, and it has
  a latency budget too: a gate that takes 800ms to save a $0.004 answer is a bad trade.

## Do this

1. Implement `Grounder.__init__`, recording document frequencies over heading paths and
   bodies.
2. Implement `idf`.
3. Implement `sufficiency` — weighted coverage of the question.
4. Implement `answer` — refuse below the threshold, otherwise defer to the reader.
5. Run the tests.
6. Run the sweep and look at the curve.

```sh
python stages/06_grounding/test_stage06.py
IMPL=reference python evals/run_qa_eval.py --sweep
IMPL=reference python evals/run_qa_eval.py --min-sufficiency 0.6
```

## Check yourself

Expected: all tests pass; the sweep shows false answers falling and false refusals
rising as the threshold grows. If your curve is flat, your `sufficiency` is probably
ignoring idf and counting words.

```sh
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from ragkit.ingest import chunk_corpus
from ragkit.grounding import Grounder
g = Grounder(chunk_corpus(Path('corpus/fastapi')))
print('terraform idf', round(g.idf('terraform'), 2))
print('request idf  ', round(g.idf('request'), 2))
"
```

`terraform` should be much higher than `request`. If it is 0.0, your document frequency
table is empty.

## If you are stuck

**Hint 1.** Build the frequency table once, over distinct terms per chunk. Counting
repeats inflates df and flattens the weights.

**Hint 2.** Include the heading path when counting. A technology named only in a heading
is still part of the page's vocabulary.

**Hint 3.** `sufficiency` compares *sets*: a term present ten times is not "more
present" than one present once. The question is coverage, not frequency.

## What a senior would notice

**Whether the unanswerable set is treated as an asset.** Twelve questions about absent
technology are the most valuable part of this eval, because they are the only ones that
can catch a system that answers everything. Teams forget to write them, and then their
eval reports 95% accuracy forever.

**Whether the threshold is documented with its cost.** "0.5" is a number. "0.5 costs us
one in five answerable questions to catch three in four unanswerable ones" is a
decision. Write the trade-off next to the constant, or the next person will change it
without knowing what they are buying.

**Whether refusal is user-visible.** A refusal is a product moment. The best systems
say what they looked for and what was missing ("the documentation covers deployment but
not Terraform"), which turns a dead end into a next step. Our refusal text is a single
sentence; that is a limitation worth naming.
