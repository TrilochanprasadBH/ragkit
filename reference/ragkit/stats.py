"""Stage 07 reference: the statistics an eval report needs.

Three functions, and each one exists because a number without it is misleading.

**wilson_interval** puts a range around a proportion. "Recall@5 was 78%" on nine
questions is not a measurement; it is a measurement with a range roughly twenty points
wide, and saying only the middle is how teams convince themselves that a two-point
change matters.

**mcnemar_exact_p** answers "is this difference real" for two configurations evaluated
on the same questions. The same reasoning as the lab's drill 12: with paired pass/fail
results, only the disagreements carry information.

**required_sample_size** answers the question before the experiment: how many questions
would I need to detect an effect this size? Run it before you build the eval, not after
it fails to show anything.

The formulas are standard. The judgment is in using them.
"""

from __future__ import annotations

import math

Z_95 = 1.96


def wilson_interval(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% confidence interval for a proportion, using the Wilson score method.

    Preferred over the normal approximation because it stays inside [0, 1] and behaves
    sensibly at proportions near 0 or 1, which is exactly where small eval sets live.
    """
    if n <= 0:
        return (0.0, 1.0)

    p = successes / n
    denominator = 1 + (z * z) / n
    centre = p + (z * z) / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))

    low = (centre - margin) / denominator
    high = (centre + margin) / denominator
    return (max(0.0, low), min(1.0, high))


def mcnemar_exact_p(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail results.

    `n01` is the number of questions where configuration A passed and B failed.
    `n10` is the opposite. Questions where both agree carry no information about which
    is better, so they are excluded.
    """
    discordant = n01 + n10
    if discordant == 0:
        return 1.0

    k = max(n01, n10)
    tail = sum(
        math.comb(discordant, i) for i in range(k, discordant + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def required_sample_size(
    baseline: float,
    minimum_effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Approximate questions needed to detect `minimum_effect` over `baseline`.

    This is the two-proportion formula with equal group sizes, applied to a paired
    design as a rough lower bound. It is deliberately simple: the point is the order of
    magnitude, not the third decimal.

    Example: a baseline of 0.75 and a hoped-for 0.85 needs about 230 questions per arm.
    A 32-question eval cannot answer the question you are asking it.
    """
    if minimum_effect <= 0:
        raise ValueError("minimum_effect must be positive")

    z_alpha = {0.05: 1.96, 0.01: 2.576}.get(alpha, 1.96)
    z_power = {0.80: 0.8416, 0.90: 1.2816}.get(power, 0.8416)

    p1 = baseline
    p2 = min(1.0, baseline + minimum_effect)
    pooled = (p1 + p2) / 2

    numerator = (
        z_alpha * math.sqrt(2 * pooled * (1 - pooled))
        + z_power * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return math.ceil(numerator / (minimum_effect**2))
