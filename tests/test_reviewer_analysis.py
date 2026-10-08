import math
import random
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from analyze_user_paired_ci import paired_ci
from summarize_history_budget import draw_figure, slope


class ReviewerAnalysisTests(unittest.TestCase):
    def test_paired_bootstrap_keeps_user_queries_together(self):
        scores = {"local": {"a": [2.0, 2], "b": [1.0, 1]},
                  "group": {"a": [0.0, 2], "b": [0.0, 1]}}
        result = paired_ci(["a", "b"], [scores], "local", "group",
                           1000, random.Random(2))
        self.assertAlmostEqual(result["delta"], 1.0)
        self.assertAlmostEqual(result["ci_low"], 1.0)
        self.assertAlmostEqual(result["ci_high"], 1.0)

    def test_within_user_slope_removes_user_intercepts(self):
        data = {user: [(n, intercept + 2 * math.log1p(n))
                       for n in (2, 4, 8, 16)]
                for user, intercept in (("a", 3), ("b", -5))}
        self.assertAlmostEqual(slope(data, ["a", "b"]), 2.0)

    def test_gain_figure_is_well_formed_svg(self):
        row = {"segment_gain": {"delta": 0.03, "ci_low": 0.01, "ci_high": 0.05},
               "individual_gain": {"delta": 0.07, "ci_low": 0.04, "ci_high": 0.10}}
        payload = {"tasks": {"movie_tagging": {"budgets": {str(n): row
                                                               for n in (25, 50, 75, 100)}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gains.svg"
            draw_figure(payload, path)
            self.assertEqual(ET.parse(path).getroot().tag.rsplit("}", 1)[-1], "svg")


if __name__ == "__main__":
    unittest.main()
