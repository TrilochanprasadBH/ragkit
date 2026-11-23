"""Stage 07 tests: intervals, paired significance, and the report.

    python stages/07_eval/test_stage07.py
    IMPL=reference python stages/07_eval/test_stage07.py
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
IMPL = os.environ.get("IMPL", "src")
sys.path.insert(0, str(ROOT / IMPL))

from ragkit.stats import (  # noqa: E402
    mcnemar_exact_p,
    required_sample_size,
    wilson_interval,
)


class TestWilson(unittest.TestCase):
    def test_half_of_ten(self):
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.237, places=2)
        self.assertAlmostEqual(high, 0.763, places=2)

    def test_zero_successes(self):
        low, high = wilson_interval(0, 10)
        self.assertEqual(low, 0.0)
        self.assertAlmostEqual(high, 0.278, places=2)

    def test_all_successes(self):
        low, high = wilson_interval(10, 10)
        self.assertAlmostEqual(low, 0.722, places=2)
        self.assertEqual(high, 1.0)

    def test_empty_sample_is_uninformative(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_bounds_stay_inside_zero_and_one(self):
        for successes, n in ((0, 3), (1, 3), (3, 3), (0, 50), (50, 50)):
            low, high = wilson_interval(successes, n)
            self.assertGreaterEqual(low, 0.0)
            self.assertLessEqual(high, 1.0)
            self.assertLessEqual(low, high)

    def test_interval_narrows_with_more_data(self):
        small = wilson_interval(5, 10)
        large = wilson_interval(50, 100)
        self.assertGreater(small[1] - small[0], large[1] - large[0])

    def test_nine_questions_are_wide(self):
        # The point of the stage: a 6-of-9 result cannot support a precise claim.
        low, high = wilson_interval(6, 9)
        self.assertGreater(high - low, 0.45)


class TestMcNemar(unittest.TestCase):
    def test_no_discordance(self):
        self.assertEqual(mcnemar_exact_p(0, 0), 1.0)

    def test_symmetric_disagreement(self):
        self.assertEqual(mcnemar_exact_p(5, 5), 1.0)

    def test_clean_sweep_is_significant(self):
        self.assertEqual(mcnemar_exact_p(0, 5), 0.0625)

    def test_large_imbalance_is_significant(self):
        self.assertLess(mcnemar_exact_p(1, 12), 0.01)

    def test_symmetry(self):
        self.assertAlmostEqual(mcnemar_exact_p(2, 7), mcnemar_exact_p(7, 2))


class TestSampleSize(unittest.TestCase):
    def test_ten_point_effect_needs_hundreds(self):
        needed = required_sample_size(0.75, 0.10)
        self.assertGreater(needed, 150)
        self.assertLess(needed, 400)

    def test_larger_effects_need_fewer(self):
        self.assertGreater(
            required_sample_size(0.75, 0.05),
            required_sample_size(0.75, 0.20),
        )

    def test_zero_effect_raises(self):
        with self.assertRaises(ValueError):
            required_sample_size(0.75, 0.0)


class TestReport(unittest.TestCase):
    def test_report_is_generated_with_the_expected_sections(self):
        with tempfile.TemporaryDirectory() as directory:
            output = pathlib.Path(directory) / "report.md"
            result = subprocess.run(
                [sys.executable, "evals/report.py", "--output", str(output)],
                cwd=ROOT,
                env={**os.environ, "IMPL": IMPL},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr[-800:])
            text = output.read_text()

        for heading in (
            "# Evaluation report",
            "## 1. Retrieval",
            "## 2. End to end",
            "## 3. Unanswerable",
            "## 4. Where hybrid fails",
            "## 5. Is the difference real?",
            "## 6. Power",
        ):
            self.assertIn(heading, text)

        self.assertIn("95% CI", text)
        self.assertIn("McNemar", text)
        self.assertIn("held out", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
