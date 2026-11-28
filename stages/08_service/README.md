# Stage 08 — The service

```sh
python stages/08_service/test_stage08.py
```

---

## What you will build

One class that composes everything, plus the instrumentation that makes it operable:
per-stage timings, cost per request, a usage log, and an index manifest. Then a small
HTTP server in front of it.

## Why this step is next

Every stage so far was a script that ran and printed a number. A product answers
requests, at a latency users tolerate, for a price the business can afford, and leaves
enough of a trail that you can debug it at 2am.

That is a different set of problems, and none of them appear until you put a clock and a
budget around the pipeline.

## The idea

**Latency is part of correctness.** A perfect answer in four seconds is a worse product
than a good answer in two hundred milliseconds. The only way to know where the time
goes is to time each stage separately: retrieve, pack, gate, answer. "The request took
120ms" tells you to optimise everything; "retrieval was 90ms of it" tells you what to
fix.

**Use percentiles, not averages.** A mean hides the slow request, and the slow request
is the one users remember. Report p50 and p95, and set the budget on p95, because that
is roughly one request in twenty.

**Cost per request is a product number.** Our pipeline embeds the query and reads from a
local index, so a request costs a fraction of a cent. At a million requests that is
still a real line item, and it is the kind of number that is invisible until someone
asks. Measure it while it is small.

**The index has a schema, and the embedder is part of it.** A vector index built by one
model cannot be queried by another. Recording the embedder name and dimension in a
manifest next to the index turns a silent quality collapse into a startup check.

**Instrument the request, not the function.** The usage log records one line per
request: timestamp, a hash of the question, whether it was refused, how many citations
it had, the timings, and the cost. A hash rather than the question text is deliberate:
logs get shared, questions contain customer data, and you rarely need the text to
answer "how often do we refuse".

## The measurement

`scripts/bench.py` replays the 32 answerable questions:

| stage | p50 | p95 | max |
| --- | --- | --- | --- |
| retrieve | 1.8ms | 2.4ms | 2.5ms |
| pack | 0.6ms | 0.7ms | 0.8ms |
| answer | 0.4ms | 0.5ms | 0.5ms |
| **total** | **2.7ms** | **3.3ms** | **3.6ms** |

Cost: **$0.00000019** per request, which is **$0.19 per million**.

Those numbers look implausibly good, and they are — for this pipeline. Retrieval is an
in-memory index over 362 chunks and generation is a local extractive reader. A real
deployment replaces the reader with a model call, which costs hundreds of milliseconds
and a fraction of a cent per request. What this stage teaches is the *instrumentation*,
not the numbers: when you swap in a real model, you will see exactly where the time and
money went because the hooks are already there.

That is worth stating plainly in an interview. "p95 is 3ms" invites the question "what
are you not measuring?", and the honest answer is "the generation call, which is what
stage 06's stand-in reader replaces".

## Deploying it

A `Dockerfile` is included:

```sh
docker build -t ragkit .
docker run -p 8000:8000 ragkit
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "what are query parameters in a URL"}'
```

Three production notes:

- **The health check reports the manifest**, so a deploy that ships a mismatched
  embedder is visible in the first probe rather than the first complaint.
- **The index is built at startup.** That is fine at 362 chunks and wrong at 362,000.
  At that size you build offline and load a serialised index; the service code does not
  change.
- **Concurrency is the server's problem, not the service's.** `Service.ask` is
  stateless apart from the append-only log, so it is safe to call from threads.

## Other ways to serve this

This project uses the standard library's HTTP server so that it runs anywhere with no
installation. In production you would almost certainly use a framework. The same
`Service` works unchanged; only the routing layer differs:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
service = build_service()

class Ask(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok", **service.manifest()}

@app.post("/ask")
def ask(body: Ask):
    return service.ask(body.question)
```

That is the whole change. Which is the point of keeping the HTTP layer separate from the
pipeline: the framework is a detail, the service is the thing.

You would also, in a real deployment, add authentication, rate limiting, per-tenant
quotas, and tracing. Those are infrastructure concerns and they belong outside
`Service.ask` for the same reason.

## Field notes

From `FIELD-NOTES.md`:

- **AWS Well-Architected** recommends explicit latency budgets per retrieval stage, not
  one budget for the whole request. That is exactly what `timings` provides, and it is
  why a regression in packing shows up as a packing number rather than as "the service
  feels slower".
- **PostHog's decision models** are relevant here in a way they were not in earlier
  stages: a cheap gate in front of an expensive call cut 98.7% of their model calls. Our
  sufficiency gate is that pattern at the lexical end of the spectrum. If generation
  cost dominates, gate harder.
- **Pinning the embedding model** stops being advice and becomes a file: `reports/index.json`
  records what built the index.

## Do this

1. Implement `Service.__init__`, `ask`, `manifest`, and `write_manifest`.
2. Run the tests.
3. Run the benchmark.
4. Start the server and call it by hand.

```sh
python stages/08_service/test_stage08.py
python scripts/bench.py
python scripts/serve.py
```

In another terminal:

```sh
curl -s localhost:8000/health
curl -s -X POST localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question": "how do I set a response header"}'
```

## Check yourself

Expected: 14 tests pass; p95 under 300ms; `/health` returns the manifest; three requests
put three lines in `logs/usage.jsonl`.

One test is worth reading closely: `test_adjacent_topic_is_not_refused`. It asserts
that a question about Spring Boot WebSockets is **not** refused, because the gate answers
it from related material. That is a documented failure, not a passing behaviour, and it
is in the test suite so nobody can quietly claim it does not happen.

This stage also found a real bug in stage 06. Serving the pipeline exposed that natural
questions ("how do I set a response header") were refused while terse ones were answered,
because interrogatives and pronouns were counting toward sufficiency. The fix is recorded
in the stage 06 lesson, and the numbers there were re-measured after it.

## If you are stuck

**Hint 1.** Time the stages with `time.perf_counter()` around each call, not around the
whole function. Three timestamps give you three numbers; one gives you a total.

**Hint 2.** Round milliseconds to two decimals in the response. Sub-microsecond
precision in an API response is noise that makes diffs unreadable.

**Hint 3.** The usage log is append-only and one line per request. Open, write, close;
do not hold the file open across requests, or a crash loses the tail.

## What a senior would notice

**Whether the numbers are honest about what they exclude.** A 3ms p95 with a local
reader is not a latency claim about a language model. Saying so in the same breath is
the difference between a benchmark and marketing.

**Whether refusal is in the logs.** Refusal rate is a product metric: it tells you which
questions your documentation does not answer. A service that logs only successes cannot
improve its corpus.

**Whether the deployment can be rolled back.** The manifest makes an index/embedder
mismatch visible. If you ever change the embedder, the correct procedure is build the
new index alongside, swap, then delete, and the manifest is how you know which one is
live.

---

## The project ends here

You now have eight stages, a service, and a report that states what it can and cannot
prove. One more stage exists after this one: **stage 09 measures whether a reranker
earns its cost, and the honest answer on this corpus is no.** Read it — a measured
negative result is the most transferable thing in the repository. Then, if you want
the next three things to do, in order:

1. **Grow the eval set** to the size the power analysis asks for, written by someone who
   has not read the code. This is the highest-value work in the repository and nobody
   will ever ask you to do it.
2. **Replace the stand-in reader with a real model**, keep the same packer and gate, and
   measure whether citation precision improves. You will finally be measuring the thing
   the earlier stages could only approximate.
3. **Add a reranker** — but only after step 1, so you can tell whether it helped.
