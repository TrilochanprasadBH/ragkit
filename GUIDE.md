# Guide

This is the course. Eight stages, in order. Each stage has a lesson in
`stages/NN_name/README.md` and a test suite that defines what "done" means.

Read this page once. Then work one stage at a time.

---

## 1. How to work

For each stage:

1. **Read the lesson** in `stages/NN_name/README.md`. It is short. Read all of it
   before writing code.
2. **Run the tests.** They will fail. Read the failure messages; they describe the
   contract.
3. **Write the code** in `src/ragkit/`. The stubs contain the signatures and the
   docstrings.
4. **Run the tests again.** Repeat until they pass.
5. **Measure.** Every stage adds a number to the results table in `README.md`.
6. **Write three lines in `NOTES.md`.** What you changed, what moved, what surprised
   you.

If you are stuck for more than 30 minutes on one stage, read
`reference/ragkit/`. Diff it against yours. Then close it and write your own version
from memory. Copying the reference teaches you nothing; reconstructing it teaches you
the thing.

## 2. The stages

| stage | you build | the idea it teaches | ends with |
| --- | --- | --- | --- |
| 01 | document loader and chunker | text has structure, and structure is what makes retrieval work | 22 documents into stable, citable chunks |
| 02 | lexical index and BM25 | a word in two documents is worth more than a word in thirty | recall@5 on 40 golden questions |
| 03 | query expansion and ranking | query language is not document language | better first-hit, less ranking drift |
| 04 | dense retrieval and fusion | lexical match fails on paraphrase; vectors fail on exact terms | hybrid recall on paraphrased questions |
| 05 | context packing with citations | a context window is a budget | answers that cite section, not just document |
| 06 | grounding checks and refusal | an answer without support is a guess that reads like a fact | measured false-answer rate |
| 07 | the eval suite | a number without an interval is a rumour | a report you can defend |
| 08 | the service | latency and cost are part of correctness | a running endpoint with usage logging |

## 3. What you should be able to say after each stage

This is the bar. If you cannot say the sentence, the stage is not done.

**After 01:** "Chunks keep their heading path, so a citation can say
`tutorial/query-params.md > Query Parameter Defaults` rather than a filename."

**After 02:** "BM25 beats raw term counting because it weights terms by rarity and
normalises for document length. Recall@5 is the fraction of questions whose correct
section appears in the top five."

**After 03:** "Query expansion closes the vocabulary gap. I measured it, and it
bought 9 points of recall, which I can attribute to aliases rather than to luck."

**After 04:** "Hybrid retrieval exists because lexical search misses paraphrase and
dense search misses exact identifiers. Reciprocal rank fusion combines them without
tuning a weight."

**After 05:** "Packing is a covering problem. I select documents to cover the
question inside a token budget, not to maximise individual relevance."

**After 06:** "Grounding is a check, not a prompt. Each sentence in the answer must
map to a retrieved passage, or the system refuses."

**After 07:** "I report recall with a confidence interval and a per-category
breakdown, so a two-point change is not mistaken for progress."

**After 08:** "p95 latency and cost per query are measured and capped, because a
correct answer that takes nine seconds and four cents is not shippable."

## 4. The rules

**Do not skip stage 02 to get to the vector database.** The lexical baseline is the
thing you will compare the dense version against. Without it you cannot say whether
embeddings helped, and you will keep them because they feel modern.

**Do not tune against the golden set.** The golden set in `evals/questions.jsonl`
is your test set. If you adjust until the number moves, you have learned nothing and
measured your own memory. Stage 07 introduces a held-out split for exactly this
reason.

**Measure before and after, every time.** One number, taken twice, is a claim. One
number, taken once, is a feeling.

**Never delete a failing test.** If a test is wrong, fix the test and say why in
`NOTES.md`.

## 5. What this repository is not

It is not a framework. There is no plugin system and no configuration layer. Every
component is a function you can read in one sitting.

That is deliberate. Frameworks teach you their abstractions. This teaches you the
problem.

## 6. When you finish

You will have:

- a working RAG service over a real corpus
- an eval report with numbers you produced and can defend
- a container you can deploy
- eight stages of notes explaining your own decisions

That is a portfolio piece and an interview answer. The numbers are the part that
matters: anyone can say they built a RAG system, and very few can say what its
recall@5 is and why.
