# Field notes

What production teams published about building AI systems, what we took from it, and
what we deliberately did not build here.

Each note has four lines: **Source**, **What they did**, **What it means**, **In this
project**. Read the ones relevant to the stage you are on. This file grows as the
stages do.

Two rules for using it:

1. A published number is an argument to test on your corpus, not a conclusion to copy.
2. Some notes end with "not built here". That is deliberate. Knowing a technique
   exists, and why you are not using it yet, is worth more than half-building it.

---

## Retrieval

### 1. Contextual Retrieval — Anthropic (September 2024)

**Source.** anthropic.com/research/contextual-retrieval

**What they did.** Before indexing a chunk, had a cheap model write 50–100 tokens
situating it in its document ("this chunk is from the refund policy for international
orders"), then indexed that enriched text with *both* embeddings and BM25. Measured
across codebases, fiction, ArXiv and papers: contextual embeddings cut top-20 failure
by 35%, adding contextual BM25 cut it by 49%, adding a reranker cut it by 67%.
Pass@10 went from 87.15% to 93.21% (hybrid) to 95.26% (hybrid + rerank). Indexing cost
around $1.02 per million document tokens with prompt caching.

**What it means.** Lexical search is not the legacy half of retrieval that vectors
replace. It is a permanent half, because it catches exact terms that embeddings smooth
away. And enrichment at ingest time is cheap relative to the query path.

**In this project.** Our heading-path indexing (stage 01) is a free approximation of
the same idea: context we already had, with no model call. Stage 04 adds the dense
half. We do not generate per-chunk context summaries — that needs a model call per
chunk, which this offline project does not make.

### 2. Production RAG evaluation — Oracle developers blog (July 2026)

**Source.** blogs.oracle.com/developers/production-rag-evaluation-keyword-vector-sql-or-hybrid-search

**What they did.** Argued that retrieval and generation must be evaluated separately,
that keyword, vector and hybrid should be compared against the same ground-truth
questions before choosing a default, and that a production eval must also test exact
identifiers, citations, abstention, freshness and tenant isolation. Warned against
declaring a universal winner.

**What it means.** Most "which retrieval is best" arguments are unanswerable without a
golden set. Build the instrument before having the argument.

**In this project.** `evals/questions.jsonl` is the instrument, and stage 07 turns it
into a report with per-category breakdowns. Tenant isolation and freshness are out of
scope for a single-corpus project.

### 3. Well-Architected Agentic AI Lens — AWS

**Source.** docs.aws.amazon.com/wellarchitected/latest/agentic-ai-lens

**What they did.** Recommended chunking matched to structure, latency tracked per
stage (embedding, search, rerank) with explicit budgets, query reformulation ahead of
retrieval, and hybrid plus reranking "used where the corpus and query mix justify
them, rather than stacked by default". Listed the anti-pattern of embedding a raw
query when the corpus uses different terminology.

**What it means.** Every added stage is a latency and cost liability. Add it because a
failure list demands it.

**In this project.** Stage 08 measures p95 latency per stage. Stage 04 adds dense
retrieval only after the lexical failure list shows vocabulary gaps, which is the
evidence AWS is asking for.

### 4. Reciprocal Rank Fusion — Azure AI Search, OpenSearch

**What they did.** Fuse ranked lists by rank rather than raw score:
`score(doc) = sum(1 / (k + rank))` with `k = 60` in the common configuration. Rank
fusion avoids pretending that a BM25 score and a cosine distance are on the same
scale.

**What it means.** When you combine two retrievers, never average their scores. Fuse
ranks.

**In this project.** Stage 04 implements RRF. No library required.

### 5. The replication that disagreed (October 2026)

**Source.** AQUA LLC technical report on contextual retrieval in Japanese corpora.

**What they did.** Re-ran the contextual-retrieval experiment with controls: context
headers versus plain heading-aligned chunking, placebo contexts, and multiple fusion
weights. Found that generated context did **not** beat heading-aligned chunking for
dense retrieval, while reranking gave the largest single improvement. Also found the
best fusion weight depended on how questions were phrased: BM25 won on lexically close
questions, dense on paraphrased ones, and equal-weight fusion plus reranking was best
overall.

**What it means.** Vendor numbers replicate unevenly. The two claims that survived:
chunk at structure, and rerank last.

**In this project.** Directly responsible for the heading experiment in stage 03, which
found no effect on this corpus. Reported as a null result instead of quietly dropped.

### 5b. Pin the embedding model — practitioner guide (2026)

**Source.** learnaiengineering.dev, "Production RAG architecture: a practical guide"

**What they did.** Recorded the embedding model version with every vector, on the
grounds that queries and documents must be embedded by the same model and that
swapping models means re-embedding the whole corpus.

**What it means.** A vector index has a schema, and the model is part of it. Treating
it as an implementation detail is how teams end up with a quietly broken index after an
upgrade.

**In this project.** The dense index is rebuilt on every run, so the problem cannot
bite yet. Stage 08 writes the embedder name into the index metadata so that it could.

---

## Agents

### 6. Building effective agents — Anthropic (December 2024)

**Source.** anthropic.com/engineering/building-effective-agents

**What they did.** Distinguished workflows (fixed code paths that call models) from
agents (models that direct their own process). Found the most successful
implementations used simple composable patterns, not frameworks. Described routing,
parallelisation, orchestrator-worker and evaluator-optimiser as the small set of
patterns that cover most needs. Emphasised tool documentation and testing as the
"agent-computer interface".

**What it means.** Start with the simplest thing that works. A workflow with two model
calls is usually better than an agent framework.

**In this project.** Stages 01–08 are a workflow, not an agent: the code owns the
control flow. `triage` (project 2) is where a real loop appears, and only because
ticket handling genuinely needs one.

### 7. Don't build multi-agents, and what changed — Cognition (2025, 2026)

**Source.** cognition.com/blog/dont-build-multi-agents and /multi-agents-working

**What they did.** In 2025 argued that parallel agents make conflicting implicit
decisions and that context should stay single-threaded. In 2026, reported that a
narrower class works: multiple agents contribute *intelligence* while writes stay
single-threaded — a clean-context reviewer, a specialist advisor, a manager that
coordinates scope.

**What it means.** The failure mode is parallel writers, not parallelism itself.
Read-only helpers are cheap; shared write access is where coherence dies.

**In this project.** `scout` (project 3) uses read-only sub-agents for search and a
single writer for synthesis, following this exactly.

### 8. When to use multi-agent systems — Anthropic (January 2026)

**Source.** claude.com/blog/building-multi-agent-systems-when-and-how-to-use-them

**What they did.** Named three situations where multiple agents beat one: context
pollution, genuinely parallel tasks, and specialisation (including too many tools
confusing a single agent at roughly 20+ tools). Described the verification subagent
and its main failure, the "early victory" problem: a verifier runs one test, sees it
pass, declares success.

**What it means.** Multi-agent is a remedy for specific constraints, not a maturity
signal. If you cannot name which constraint you are relieving, use one agent.

**In this project.** Not built. The capstone stage of `triage` includes a verification
step for exactly this reason, but as a function, not an agent.

### 9. Eight learnings from a year of agents — PostHog (November 2025)

**Source.** posthog.com/blog/8-learnings-from-1-year-of-agents-posthog-ai

**What they did.** Reported that graph-based orchestration was a bad fit for
free-form work because the model cannot self-correct and context is easily lost; moved
to a single loop with one message history. Added a `todo_write` tool whose only job is
to restate the remaining plan, which kept the agent on track over dozens of steps.
Also: "frameworks considered harmful" — they migrated to LangChain/LangGraph and then
migrated off, on the grounds that framework abstractions lag model capabilities.

**What it means.** The loop is the architecture. Structure that helps the model track
its own state (a to-do list) outperforms structure that tries to control its path.

**In this project.** The single-loop shape appears in `triage`. This project uses no
agent framework at all, which is the same conclusion arrived at independently.

### 10. Cloud agents runtime — PostHog (2026)

**Source.** posthog.com/blog/cloud-agents-runtime

**What they did.** Ran every agent task through a durable workflow that provisions a
sandbox, clones repositories, relays events, and refreshes credentials. Notably,
network access is restricted **at the network level**, so "a blocked domain stays
blocked however nicely the prompt asks".

**What it means.** The strongest guardrail is the one the model cannot negotiate with.
Prompt-level restrictions are requests; infrastructure-level restrictions are
guarantees.

**In this project.** Stage 08 of `triage` puts the approval gate in the tool registry
rather than the prompt, for the same reason.

---

## Cost, routing and gating

### 11. Decision models — PostHog (November 2026)

**Source.** posthog.com/newsletter/decision-models

**What they did.** Used small calibrated decision models (typed answers with a
probability) instead of LLMs for narrow questions. Reported them up to 444.6x cheaper
and 193.6x faster than frontier models. Three production uses: **gating** expensive
calls (in PII scrubbing, 98.7% of items skip the LLM entirely), **routing** to a model
(because "a failed run costs more than a smarter option"), and **choosing a
visualisation**. Thresholds are set by which error is more expensive: 0.4 when a false
negative stores personal data, 0.85 when a false positive annoys a user. They also
reported a failure: using decision models to rank PR-review findings gave AUC 0.52,
i.e. no better than chance, so open-ended ranking stayed with an LLM.

**What it means.** Not everything needs a generative model. A classifier with a
calibrated probability is cheaper, faster, and honest about its uncertainty — but only
for narrow, well-posed questions. Their own AUC result is the boundary marker.

**In this project.** Not built: it needs a hosted model. It directly shapes stage 08:
the router in that stage is a hand-written classifier standing in for a decision
model, and the lesson states the thresholds should be chosen by error cost, not by
accuracy.

**Other ways to do this.** A fine-tuned small classifier, a rules table, or an
embedding nearest-neighbour lookup can all play the same role. The choice depends on
whether you have labels and how often the question shape changes.

---

## Evaluation

### 12. Evaluating multi-agent systems — Anthropic (June 2025)

**Source.** anthropic.com/research/multi-agent-research-system

**What they did.** Observed that agent evaluations cannot assume a fixed path, because
valid runs take different routes. Argued for judging the outcome *and* whether the
process was reasonable, rather than comparing against a prescribed step sequence.

**What it means.** Trajectory evaluation is not "did it follow my steps"; it is "did it
reach a defensible result by a defensible route".

**In this project.** Stage 07 adds a per-question breakdown. The `triage` project
implements trajectory scoring with partial credit, safety overrides and process
penalties.

### 13. Benchmarking agents against real data — PostHog (October 2026)

**Source.** posthog.com/blog/how-to-benchmark-ai-agents

**What they did.** Benchmarked their code-review agent by treating model, effort level
and harness as three variables, against tasks with known answers, so that changing any
one could be evaluated rather than guessed.

**What it means.** Separate the variables you are comparing. "We upgraded the model and
it feels better" is not a benchmark.

**In this project.** Stage 07 separates retrieval configuration from generation, and
stage 03's heading experiment is a one-variable comparison.

### 14. Start small with evals — LangChain (June 2025)

**Source.** blog.langchain.com/how-and-when-to-build-multi-agent-systems

**What they did.** Noted that roughly twenty curated datapoints are enough to start,
that LLM-as-judge can automate scoring of experiments, and that human review remains
necessary.

**What it means.** Twenty-four questions is a legitimate eval set. Waiting for a
thousand is a way of never starting.

**In this project.** The golden set is 32 questions, split 23 development and 9 held out.
Stage 07 is where the split and the intervals came from.

### 15. Passage re-ranking with BERT — Nogueira and Cho (2019)

**Source.** arxiv.org/abs/1901.04085

**What they did.** Re-implemented BERT as a query-passage cross-encoder and re-ranked a
first-stage list with it. The abstract reports the system as state of the art on TREC-CAR
and the top entry on the MS MARCO passage retrieval leaderboard, "outperforming the
previous state of the art by 27% (relative) in MRR@10".

**What it means.** The gain comes from reading the query and the passage together with a
pretrained language model. That is the part a lexical stand-in cannot reproduce, and it is
why the size of a reranking gain depends on how much the first stage left on the table.

**In this project.** Stage 09 builds the stand-in and measures it before shipping it:
+0.128 MRR when the first stage is dense, +0.009 when it is hybrid, and neither gain
survives a sign test at n = 23. The reranker is not in the pipeline. What shipped is the
measurement and the rule: check the first stage's MRR first.

---

## What we deliberately did not build

| technique | why not here | when you would |
| --- | --- | --- |
| Generated per-chunk context | needs a model call per chunk at ingest | corpus over ~100k chunks, failures are vocabulary gaps |
| Vector database | 362 chunks fit in memory | millions of chunks, or multi-tenant filtering |
| Agent framework (LangGraph and similar) | the control flow is known and small | many tools, dynamic planning, human-in-the-loop |
| Multi-agent orchestration | one loop is enough for this task | context pollution, parallel independent work, or 20+ tools |
| Fine-tuning | no labels for behaviour, and retrieval covers knowledge | consistent format or tone is the goal, not knowledge |
| A real cross-encoder reranker | needs a hosted model per candidate; the stand-in is what stage 09 measures | first-stage MRR below roughly 0.7 and an eval set big enough to resolve the change |
| Decision-model gate | needs a hosted model | a high-volume, narrow, expensive call you want to skip |

Each row is a deliberate choice you should be able to defend, not an oversight.
