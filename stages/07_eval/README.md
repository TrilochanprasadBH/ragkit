# Stage 07 — The eval suite

```sh
python stages/07_eval/test_stage07.py
```

Sixteen tests, all failing at first.

---

## What you will build

Three statistical functions and a report generator. No new retrieval. This stage is
about whether the numbers you have been quoting since stage 02 mean anything.

## Why this step is next

Every number so far came from the same 32 questions that were used to choose the
retrieval settings. That is a leak, and it is the most common one in practice: you tune
until the number improves, then report the number you tuned.

There is a second problem. "Recall@5 is 83%" is not a measurement. It is a measurement
with a range, and on twenty-odd questions that range is roughly twenty points wide.
Reporting the middle and ignoring the edges is how teams convince themselves that a
two-point change is progress.

## The idea

**A proportion from a small sample is a range.** If you get 6 of 9, your best estimate
is 67%, and the plausible range is roughly 35% to 88%. The range is not uncertainty
about your code; it is the arithmetic consequence of having looked at nine things.

The correct interval for a proportion near 0 or 1 is the Wilson score interval, not the
normal approximation. The normal approximation produces absurd values outside [0, 1]
when your sample is small and your successes are few, which is exactly the situation a
32-question eval is in.

**A held-out set is only held out if you do not tune on it.** Splitting once and then
tuning against the test number turns the test into a second development set. The
discipline is: choose settings on `dev`, then look at `test` once, and report whatever
you see.

**Comparing two configurations needs paired data.** When two retrievers are evaluated on
the same questions, the questions where both succeed and both fail tell you nothing.
Only the disagreements carry information. That is McNemar's test, and it is the same
reasoning as drill 12 in the Agent Lab, applied to your own project.

**Power comes before the experiment.** The number you need is not "how many questions
do I have"; it is "how many questions would it take to detect the effect I care about".
For a ten-point improvement over a 75% baseline, at 95% confidence and 80% power, that
is roughly 250 questions. If your eval has 32, it cannot answer the question you are
asking, no matter how carefully you compute the average.

## What the report found

`evals/report.py` writes `reports/latest.md`. The first run produced this:

| configuration | dev recall@5 | held-out recall@5 |
| --- | --- | --- |
| lexical | 83% | 56% (CI 27–81%) |
| fielded | 96% | 56% (CI 27–81%) |
| dense | 78% | **78%** (CI 45–94%) |
| hybrid | 96% | 67% (CI 35–88%) |

Three findings, in order of how uncomfortable they are.

**1. The development numbers were optimistic.** Hybrid looked like 96% on dev and 67%
on held-out. That is not a bug; it is what tuning on a set does. Every parameter chosen
in stages 03 and 04 — heading weight, candidate depth, fusion constant — was chosen
while looking at questions that are now in the dev split.

**2. The ranking flipped.** On dev, hybrid and fielded tie at 96%. On held-out, dense
retrieval beats both. If you had stopped at stage 04 and reported "hybrid is best", you
would have reported a result that does not survive nine unseen questions.

**3. The differences are not distinguishable.** Hybrid versus fielded on held-out: one
question where hybrid succeeds and fielded fails, zero the other way. Exact McNemar
p = 1.000. With one discordant question, the honest verdict is "no evidence of a
difference", not "hybrid wins".

The intervals tell the same story: hybrid 35–88%, dense 45–94%. They overlap almost
completely. Any claim that one is better is a claim about which nine questions you
happened to pick.

**And the power section puts a number on the whole problem.** Detecting a ten-point
improvement needs about 250 questions per configuration. This report has 9 held out.

## What this means for the project

The bottleneck at this point is not the retriever. It is the eval set.

The correct next action is not to add reranking or tune `k` again. It is to grow the
question set: more questions per document, more categories, and questions written by
someone who has not seen the code. That is unglamorous, and it is what separates a demo
from a measurement.

This is also the most useful interview answer in the whole project. Anyone can say
"recall@5 was 83%". Very few people can say "83% on 32 questions has a 20-point
interval, my held-out number was lower than my dev number because I tuned on dev, and
I need around 250 questions before I can claim a ten-point improvement".

## Field notes

From `FIELD-NOTES.md`:

- **LangChain** noted that about twenty curated datapoints is enough to *start*. This
  stage is the rest of that sentence: twenty is enough to start, not enough to conclude.
- **Oracle** insisted retrieval and generation be evaluated separately. The report
  separates them, and section 2 shows why: retrieval finds evidence for 6 of 9 held-out
  questions, but only 2 of those answers contain the evidence, because the stand-in
  reader picks the wrong sentence. Two different failures, two different fixes.
- **PostHog** benchmarked agents by varying one thing at a time — model, effort,
  harness. The dev/test columns here are the same discipline: a comparison is only
  meaningful when you know what changed.

## Do this

1. Implement `wilson_interval`, `mcnemar_exact_p`, and `required_sample_size` in
   `src/ragkit/stats.py`.
2. Run the tests. The sanity values in the docstrings are real; if your numbers differ,
   the formula is wrong.
3. Generate the report and read it.
4. Answer these three questions in `NOTES.md`:
   - which configuration would you have shipped after stage 04, and does the report
     still support it?
   - which category is worst, and is that a retrieval problem or a reader problem?
   - how many questions would you need to detect a five-point improvement?

```sh
python stages/07_eval/test_stage07.py
IMPL=reference python evals/report.py
cat reports/latest.md
```

## Check yourself

Expected: all tests pass, the report has six sections, every proportion carries an
interval, and the significance test reports a p-value you can interpret.

If your Wilson interval for 5 of 10 is not close to (0.24, 0.76), check the brackets in
the formula. The most common error is dividing the margin by the denominator twice.

## If you are stuck

**Hint 1.** Write the formula out on paper first. The Wilson interval has three pieces:
a denominator that inflates with small n, a centre that is pulled toward 0.5, and a
margin that shrinks with n.

**Hint 2.** McNemar is a binomial test on `n01 + n10` trials. If both are zero, there is
no evidence either way, so return 1.0.

**Hint 3.** `required_sample_size` divides by the square of the effect. That is why
halving the effect you want to detect quadruples the questions you need, and why "let us
just add ten more questions" rarely changes a conclusion.

## What a senior would notice

**Whether the dev/test gap is reported.** The single most informative line in the report
is "96% on dev, 67% held out". A team that hides that line is not doing evaluation; it is
doing marketing with extra steps.

**Whether a flipped ranking triggers a rebuild or a shrug.** Dense beating hybrid on
held-out with nine questions is noise, and the right response is to collect more data,
not to switch architectures. The wrong responses are both available: chase the new
winner, or dismiss the test as unreliable. Only the interval tells you which.

**Whether the next sprint is about code or about data.** At this point the code is
adequate and the eval is not. A senior engineer spends the next day writing questions;
a junior engineer adds a reranker and reports the dev number.
