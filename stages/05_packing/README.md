# Stage 05 — Packing and citation

```sh
python stages/05_packing/test_stage05.py
```

---

## What you will build

One function. Given a question, a ranked list of chunks, and a word budget, it returns
a context window in which every passage is numbered and attributed to a source.

## Why this step is next

Retrieval gives you chunks. A model needs a prompt. The gap between those two things is
where most documentation assistants live or die.

Two questions have to be answered at once:

**What fits?** A context window is a budget. You cannot paste everything.

**What can be checked?** An answer with no source is an assertion. An answer with a
wrong source is worse than no source, because it looks verified.

## The idea

**A chunk is the wrong unit to spend money on.** Chunks in this corpus average about
ninety words. A question usually needs one sentence from one chunk. If you pay ninety
words for one useful sentence, you are spending your budget at roughly 5% efficiency.

That is what whole-chunk packing does, and it is the baseline in `baselines.py`:

```
for chunk in ranked_chunks:
    if it fits: add the whole thing
```

It is simple, it is honest, and it wastes most of the budget.

**The unit should be the sentence.** Split each chunk into whole sentences, score each
sentence against the question, and buy the highest-value sentences that fit. Value per
word, not total relevance, because the budget is in words.

**Normalise by length.** A sixty-word paragraph will contain more query terms than a
fifteen-word sentence purely by accident. Dividing the overlap by the square root of the
sentence length stops long vague passages from outbidding short precise ones.

**Never cut a sentence in half.** A truncated sentence usually loses the clause that
answered the question, and it makes the citation unverifiable. If a sentence does not
fit, try the next candidate; do not truncate.

**Show the material in source order.** The selected sentences could be presented in
score order. Source order is better for two reasons: the reader can follow the
argument, and citation numbers stay stable because they follow document structure
rather than a score that changes when you tune a parameter.

**An empty pack is a valid result.** If nothing overlaps the question, return nothing.
The reader in `answer.py` will refuse, and a refusal is the correct behaviour. A packer
that always returns something hides a broken retriever behind a confident answer.

## The measurements

Same 24 questions, same retriever (the stage 04 hybrid), budget 180 words:

| packer | evidence retained | within budget | citation right |
| --- | --- | --- | --- |
| whole chunks | 58.3% | 95.8% | 54.2% |
| **sentences** | **70.8%** | **100.0%** | 45.8% |

**Retention rose twelve points and budget compliance became perfect.** That is the
packer's contribution, and it is the number this stage owns.

**Citation precision went down.** Read that column carefully; it is the most
instructive result in the project so far.

The reader in `answer.py` is a stand-in: it picks the sentence with the most query
overlap and cites the passage that sentence came from. Sentence packing puts *more*
chunks into a smaller space, which means the pack now contains more plausible-looking
sentences that are not the answer. The stand-in picks one of them.

I tried to fix this by giving the reader a semantic score, using the stage 04
embedding. It made no difference: 46%, 54%, 50% at three budgets. The reason is
simple. In a pack assembled around a question, *every* sentence is topically related,
so every sentence overlaps the question somewhat. Distinguishing "the sentence that
answers it" from "a sentence about it" requires understanding the sentence, not a
better similarity score.

Which means: **this is not a packing failure.** It is a generation failure, and it is
stage 06's job. Reporting one number would have hidden that. Splitting retention from
citation precision is what made it visible.

## Field notes

From `FIELD-NOTES.md`:

- **Oracle** argued that retrieval and generation must be evaluated separately,
  because a system can retrieve the right evidence and still produce an unsupported
  answer. The two tables above are that argument, measured: retention (retrieval plus
  packing) and citation (selection) moved in opposite directions.
- **Anthropic's contextual retrieval** is about getting the right chunk in the window.
  This stage is about spending the window well once it is there.
- One effect worth knowing, from the literature rather than a vendor blog: models use
  information at the start and end of a context window more reliably than information
  buried in the middle ("lost in the middle", Liu et al., 2023). Packing order is not
  cosmetic. Our pack presents source order for citation stability; a system optimising
  answer quality alone might put the highest-value passage first.

## Do this

1. Read `src/ragkit/pack.py` and the contract in the docstring.
2. Split chunk bodies into sentences.
3. Score sentences by query overlap, normalised by length.
4. Select greedily by value per word until the budget is full.
5. Group by chunk, number the passages, and render `[n] label` on its own line.
6. Return a `Pack` with the text, the citations, and the word count.
7. Run the tests.
8. Run the end-to-end eval and compare against the baseline.

```sh
IMPL=reference python evals/run_qa_eval.py --packer sentences --budget 180
IMPL=reference python evals/run_qa_eval.py --packer naive --budget 180
python evals/run_qa_eval.py --packer sentences --budget 180
```

## Check yourself

```sh
python stages/05_packing/test_stage05.py
python evals/run_qa_eval.py --packer sentences --budget 180 --show-failures 5
```

Expected: all tests pass, retention above 65%, every pack inside the budget, and every
sentence in the pack found verbatim in its source.

The `test_selected_sentences_are_verbatim` test is the one to read if a pack looks
wrong. It catches invention and truncation, which are the two ways a citable window
becomes uncitable.

## If you are stuck

**Hint 1.** Keep the chunk id with each sentence as you split. You cannot reconstruct
the citation later without it.

**Hint 2.** Score first, sort, then select. Selecting while scoring makes the budget
accounting hard to reason about.

**Hint 3.** When a sentence does not fit, use `continue`, not `break`. A long sentence
near the top of the ranking should not close the pack.

## What a senior would notice

**Whether pack size was chosen or inherited.** 180 words is a choice made so that the
budget binds. In production the budget comes from the model's window minus the answer
length, minus the system prompt, and it is usually hundreds to thousands of tokens.
Changing the budget changes which packer wins: at 350 words the gap between sentence
packing and whole-chunk packing nearly closes. Try it.

**Whether citations are structural or generated.** Our citation numbers come from the
pack, not from the model. That means a citation cannot be hallucinated: the model can
only point at passages that exist. A system that asks the model to write its own
sources has no such guarantee.

**Whether refusal was tested.** The golden set has no unanswerable questions, so
refusal is never exercised. That is a gap in the eval, and a senior would notice it.
Stage 07 adds questions whose evidence is genuinely absent.
