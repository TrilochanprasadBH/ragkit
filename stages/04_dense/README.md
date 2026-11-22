# Stage 04 — Dense retrieval and rank fusion

```sh
python stages/04_dense/test_stage04.py
```

---

## What you will build

An embedding function, a dense index that scores by cosine similarity, and a fusion
step that combines lexical and dense rankings into one.

## Why this step is next

Stage 03 ended at **recall@5 of 83.3%**. Four questions still fail, and all four fail
the same way:

| question says | document says |
| --- | --- |
| "Is HTTPS just a flag I turn on?" | "HTTPS is something that is just *enabled* or not" |
| "the simplest possible file" | "the simplest FastAPI file could look like this" |
| "normal def instead of async def" | the answer exists, but the matching vocabulary is spread across the page |
| "a deployment strategy" | "the best way to deploy" |

No stemmer or identifier splitter fixes "flag" versus "enabled". They share no
characters and no structure. They are different words for the same idea, and only a
representation of *meaning* can connect them.

That is what an embedding is for. This stage builds one, measures it, and then
measures whether combining it with lexical search is worth the complexity.

## The idea

**An embedding is a function from text to a vector.** If the function is good, texts
with similar meaning land near each other. Similarity becomes arithmetic: cosine of
the angle between two vectors.

Two properties matter for the rest of the pipeline:

- **Normalise the vectors.** If every vector has length 1, cosine similarity is just a
  dot product, which is faster and simpler. (`L2_normalise` in the reference.)
- **The function must be the same at index time and query time.** A vector space built
  by model A cannot be searched with a query embedded by model B. They are different
  spaces.

**Three embeddings ship here, and they are deliberately different in quality.**

| embedder | offline | captures | measured dense recall@5 |
| --- | --- | --- | --- |
| `hashing` | yes | exact words only | 58.3% |
| `char-ngram` | yes | morphology, spelling, partial words | 79.2% |
| `remote` | no, needs a key | meaning | run it yourself |

The `hashing` embedder hashes whole words. "deploy" and "deployment" hash to unrelated
buckets, so their similarity is zero. It is the weakest possible embedding and it
exists to be beaten, because a negative control is how you know your measurement is
measuring something.

The `char-ngram` embedder hashes **character n-grams**. "deploy" and "deployment"
share most of their 4-grams, so they land close together with no stemming rules at
all. That is a real technique used for fuzzy matching and entity resolution. What it
cannot do is connect words that share no characters.

**Why fusion, and why by rank.** You now have two rankings of the same chunks, from
two retrievers that fail in different ways. Lexical search is precise on identifiers
and blind to paraphrase. Dense search is the opposite.

You cannot average their scores. A BM25 score of 12.4 and a cosine of 0.71 are not on
the same scale; adding them is a category error dressed up as arithmetic.

So you combine **ranks**:

```
score(document) = sum over each list of 1 / (k + rank_in_that_list)
```

with the rank starting at 1. A document that appears in both lists accumulates score
from both. A document that appears only once gets one contribution. That is reciprocal
rank fusion, and `k` controls how much a top position dominates.

**What `k` actually does.** Smaller `k` makes adjacent ranks far apart (1/2 versus
1/3), so a first-place finish dominates. Larger `k` flattens the list (1/61 versus
1/62), so consistency across lists matters more than being first anywhere. There is no
universal value. The number you have seen quoted, 60, comes from web search at a
different scale, and on this corpus it performs worse than 10.

## The measurements

Same 24 questions, same chunks, four configurations:

| index | recall@5 | MRR |
| --- | --- | --- |
| lexical (stage 02) | 70.8% | 0.557 |
| fielded (stage 03) | 83.3% | 0.668 |
| dense only, char-ngram | 79.2% | 0.608 |
| **hybrid, char-ngram** | **87.5%** | **0.688** |

Read that carefully. **Dense retrieval on its own was worse than lexical.** Embeddings
did not replace BM25; they added information that BM25 did not have, and the gain came
from the combination.

Now the negative control, which is the most useful table in this stage:

| fusion | recall@5 | MRR |
| --- | --- | --- |
| lexical alone | 83.3% | 0.668 |
| + `hashing` dense | 75.0% | 0.594 |
| + `char-ngram` dense | 87.5% | 0.688 |

**Adding a bad retriever made the system worse than not adding it.** The hashing
embedder contributes noise, and rank fusion treats noise as evidence. This is why "we
added vector search" is not a claim about quality; it is a claim about architecture.
Run the control.

And the parameter sweep, because the defaults are not sacred:

| candidates | k | recall@5 | MRR |
| --- | --- | --- | --- |
| 10 | 10 | 87.5% | 0.688 |
| 10 | 60 | 87.5% | 0.688 |
| 20 | 60 | 87.5% | 0.660 |
| 30 | 10 | 91.7% | 0.665 |
| 30 | 60 | 83.3% | 0.646 |

Deeper candidate lists with a flat `k` can be *worse* than shallow ones, because a long
tail of mediocre documents accumulates fused score. The defaults in the code are
`candidates = 10, k = 10` because they are the best pair on both metrics here, not
because they are conventional.

## Field notes

From `FIELD-NOTES.md`:

- **Anthropic** measured embeddings plus BM25 at 49% fewer retrieval failures, and
  embeddings alone at 35%. Our numbers show the same shape: fusion beats either half.
- **Oracle** insists retrieval be evaluated separately from generation and warns
  against declaring a universal winner. This stage is that argument in miniature: the
  winner depended on whether you scored recall or MRR, and on depth parameters.
- **The 2026 replication** found fusion weights depend on how questions are phrased.
  Our sweep agrees: the best `k` was not the published one.

## Do this

1. Implement `cosine` and the two embedders in `src/ragkit/embeddings.py`.
2. Implement `DenseIndex`, `reciprocal_rank_fusion`, and `HybridIndex`.
3. Run the tests.
4. Run the eval for all four indexes and write down the numbers.

```sh
python -c "
import sys; sys.path.insert(0, 'src')
from pathlib import Path
from ragkit.ingest import chunk_corpus
from ragkit.embeddings import CharNgramEmbedder, cosine
d, dep, ban = CharNgramEmbedder().embed(['deploy', 'deployment', 'banana'])
print('deploy~deployment', round(cosine(d, dep), 3))
print('deploy~banana    ', round(cosine(d, ban), 3))
"
```

The first number should be much larger than the second. If it is not, your n-gram
extraction is wrong.

## Check yourself

```sh
python stages/04_dense/test_stage04.py
for idx in lexical fielded dense hybrid; do
  printf "%-9s" $idx
  IMPL=reference python evals/run_evals.py --index $idx | grep -E "recall|MRR" | tr '\n' ' '
  echo
done
```

Expected: all tests pass; hybrid above the fielded baseline on both metrics.

## If you are stuck

**Hint 1.** Use a stable hash, not Python's `hash()`, which is randomised per process.
`hashlib.blake2b` with a fixed digest size gives you reproducibility.

**Hint 2.** For signed hashing, take one bit of the digest as the sign. This halves
collision bias, which matters at 512 dimensions.

**Hint 3.** RRF ignores scores entirely. If your fused output changes when you scale
the input scores, you are still using them somewhere.

## Trying the real thing

The `remote` embedder needs a key:

```sh
export EMBEDDING_API_KEY=sk-...
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_BASE_URL=https://api.openai.com/v1/embeddings
IMPL=reference python evals/run_evals.py --index hybrid --embedder remote
```

Then re-run the sweep. The four failures above are exactly the ones that a semantic
embedding should fix, and this time you will be able to say by how much on your own
corpus. That number is worth more than any blog post, including this one.

## What a senior would notice

**Whether the embedding version is pinned.** A vector index is only valid for the model
that built it. Change the model, and every vector must be recomputed. Production
systems store the model name and version alongside the vectors and treat a model
change as a full reindex. Write the version into your index metadata now, while it is
cheap.

**Whether fusion was chosen or assumed.** The sweep shows `k = 60` is worse here.
Someone who copies the default and never measures it will ship a worse system and
believe it is state of the art.

**Whether the control was run.** The `hashing` row is not a strawman for the lesson; it
is the discipline. Every time you add a retriever, measure the system without it.
Adding components is easy; removing them requires evidence.
