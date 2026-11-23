"""Stage 07 — statistics.

Read `stages/07_eval/README.md` first.

Three functions, each of which stops a number from lying:

    wilson_interval         a range around a proportion
    mcnemar_exact_p         is the difference between two configurations real
    required_sample_size    how many questions you needed before you started
"""

from __future__ import annotations


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion, Wilson score method.

    Contract:
      - if n <= 0, return (0.0, 1.0)
      - p = successes / n
      - denominator = 1 + z^2 / n
      - centre      = p + z^2 / (2n)
      - margin      = z * sqrt(p(1-p)/n + z^2/(4n^2))
      - return ((centre - margin)/denominator, (centre + margin)/denominator),
        clamped to [0, 1]

    Sanity checks: 5 of 10 gives roughly (0.237, 0.763). 0 of 10 gives (0, 0.278).
    10 of 10 gives (0.722, 1.0).
    """
    raise NotImplementedError("stage 07: implement wilson_interval")


def mcnemar_exact_p(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail results.

    Contract:
      - `n01` counts questions where A passed and B failed; `n10` the reverse
      - if both are zero, return 1.0
      - let d = n01 + n10 and k = max(n01, n10)
      - tail = P(X >= k) for X ~ Binomial(d, 0.5)
      - return min(1.0, 2 * tail)

    Sanity checks: (0, 5) gives 0.0625. (5, 5) gives 1.0. (0, 0) gives 1.0.
    """
    raise NotImplementedError("stage 07: implement mcnemar_exact_p")


def required_sample_size(
    baseline: float,
    minimum_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Approximate number of questions needed to detect `minimum_effect` over `baseline`.

    Contract:
      - raise ValueError when minimum_effect <= 0
      - z_alpha = 1.96 for alpha 0.05, 2.576 for 0.01
      - z_power = 0.8416 for power 0.8, 1.2816 for 0.9
      - p1 = baseline, p2 = min(1.0, baseline + minimum_effect), pooled = (p1+p2)/2
      - n = (z_alpha*sqrt(2*pooled*(1-pooled)) + z_power*sqrt(p1(1-p1)+p2(1-p2)))^2
            / minimum_effect^2
      - return the ceiling as an int

    Sanity check: baseline 0.75, effect 0.10 gives roughly 230.
    """
    raise NotImplementedError("stage 07: implement required_sample_size")
