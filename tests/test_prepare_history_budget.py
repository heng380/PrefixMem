import unittest

from prepare_history_budget import select_history


class NestedBudgetTests(unittest.TestCase):
    def test_nested_random_selection_is_reproducible(self):
        profile = list(range(40))
        small = select_history(profile, 123, 10, "nested_random", 0)
        medium = select_history(profile, 123, 20, "nested_random", 0)
        large = select_history(profile, 123, 30, "nested_random", 0)
        self.assertEqual(small, select_history(profile, 123, 10, "nested_random", 0))
        self.assertEqual(len(small), 10)
        self.assertTrue(set(small) < set(medium) < set(large))
        self.assertEqual(select_history(profile, 123, 40, "nested_random", 0), profile)

    def test_legacy_prefix_mode_is_unchanged(self):
        self.assertEqual(select_history([0, 1, 2], 7, 2, "prefix", 0), [0, 1])


if __name__ == "__main__":
    unittest.main()
