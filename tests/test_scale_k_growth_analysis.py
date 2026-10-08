import unittest

from scale_k_growth_analysis import (
    PREFIX_BYTES_PER_USER,
    growth_k,
    make_rows,
    projected_storage,
)


class KGrowthCostTests(unittest.TestCase):
    def setUp(self):
        self.anchor = {
            "task": "movie_tagging", "k": 5, "users": 100, "nproc": 8,
            "memory_bytes": PREFIX_BYTES_PER_USER * 100 + 10_000,
            "breakdown": {
                "global_task_adapter_bytes": 20_000,
                "group_task_adapter_bytes": 50_000,
                "local_mediator_adapter_bytes": 50_000,
            },
            "timing": {"shared_seconds": 100, "local_seconds_per_user": 2},
        }

    def test_growth_rule(self):
        self.assertEqual(growth_k(100, 100), 5)
        self.assertEqual(growth_k(500, 100), 12)
        self.assertEqual(growth_k(10_000, 100), 50)
        with self.assertRaises(ValueError):
            growth_k(0, 100)

    def test_storage_reproduces_baseline_and_grows_with_k(self):
        measured_bytes = 20_000 + 50_000 + 50_000 + self.anchor["memory_bytes"]
        self.assertEqual(projected_storage(100, 5, self.anchor), measured_bytes)
        self.assertGreater(projected_storage(100, 10, self.anchor), measured_bytes)

    def test_gpu_hours_account_for_distributed_training(self):
        rows = make_rows([self.anchor], [100, 500])
        self.assertEqual(len(rows), 4)
        self.assertAlmostEqual(rows[0]["update_gpu_hours"], 300 * 8 / 3600)
        self.assertEqual(rows[2]["update_gpu_hours"], rows[3]["update_gpu_hours"])
        self.assertEqual(rows[3]["K"], 12)
        self.assertTrue(rows[3]["k_beyond_measured_range"])


if __name__ == "__main__":
    unittest.main()
