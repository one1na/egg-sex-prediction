"""公平比较和独立测试集的测试：决策效用、共同样本准确率、分批次成绩、查看记录。"""
import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import holdout_lock  # noqa: E402
from evaluation import (CHANCE_ALARM, CHANCE_PASS, CHANCE_UNKNOWN, PRIMARY_METRIC, chance_status,  # noqa: E402
                        common_accuracy, day_metrics, fold_day_metrics, holdout_table, pooled_utility)

NAN = np.nan


class TestFairComparison(unittest.TestCase):
    sex = np.array([1, 0, 1, 0, 1, 0])
    fold = np.array([0, 0, 0, 1, 1, 1])
    deep = np.array([0.9, 0.2, 0.8, 0.7, 0.6, 0.1])       # 6 颗都判断；错的是第 4 颗
    base = np.array([0.8, NAN, 0.3, 0.2, NAN, NAN])       # 只判断 3 颗：对、错、对

    def test_primary_metric_is_utility(self) -> None:
        self.assertEqual(PRIMARY_METRIC, "utility")

    def test_utility_counts_unjudged_as_wrong(self) -> None:
        m = day_metrics(self.base, self.sex)
        self.assertAlmostEqual(m["utility"], 2 / 6)          # 判对 2 颗 / 全部 6 颗
        self.assertAlmostEqual(m["acc"], 2 / 3)               # 条件准确率只看判断了的
        self.assertAlmostEqual(m["coverage"], 0.5)
        self.assertAlmostEqual(day_metrics(self.deep, self.sex)["utility"], 5 / 6)

    def test_utility_zero_when_nothing_judged(self) -> None:
        self.assertEqual(day_metrics(np.full(4, NAN), np.array([0, 1, 0, 1]))["utility"], 0.0)

    def test_per_fold_utility(self) -> None:
        res = fold_day_metrics(self.base[:, None], self.sex, self.fold, 2)
        np.testing.assert_allclose(res["utility"][:, 0], [1 / 3, 1 / 3])

    def test_pooled_utility_differs_from_fold_mean_when_folds_unequal(self) -> None:
        sex = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.2, NAN])          # 第 0 折 4 颗全对；第 1 折 2 颗全错/没判
        fold = np.array([0, 0, 0, 0, 1, 1])
        per_fold = fold_day_metrics(scores[:, None], sex, fold, 2)["utility"][:, 0]
        self.assertAlmostEqual(per_fold.mean(), 0.5)                  # (1 + 0) / 2
        self.assertAlmostEqual(pooled_utility(scores, sex), 4 / 6)   # 合在一起：判对 4 / 全部 6

    def test_common_accuracy_only_on_eggs_both_judge(self) -> None:
        c = common_accuracy(self.deep, self.base, self.sex, self.fold, 2)
        np.testing.assert_array_equal(c.n_common, [2, 1])      # 第 0 折：蛋 0、2；第 1 折：蛋 3
        np.testing.assert_allclose(c.acc_a, [1.0, 0.0])
        np.testing.assert_allclose(c.acc_b, [0.5, 1.0])

    def test_common_accuracy_nan_without_common_eggs(self) -> None:
        c = common_accuracy(np.array([0.9, NAN]), np.array([NAN, 0.1]), np.array([1, 0]), np.zeros(2, int), 1)
        self.assertEqual(c.n_common[0], 0)
        self.assertTrue(np.isnan(c.acc_a[0]))


class TestHoldoutTable(unittest.TestCase):
    def test_overall_and_per_batch(self) -> None:
        scores = np.array([0.9, 0.2, NAN, 0.6, 0.1, 0.8])
        sex = np.array([1, 0, 1, 0, 0, 1])
        batch = np.array(["A", "A", "A", "B", "B", "B"])
        total, a, b = holdout_table(scores, sex, batch)
        self.assertEqual((total.n_eggs, total.judged, total.correct), (6, 5, 4))
        self.assertAlmostEqual(total.utility, 4 / 6)
        self.assertEqual((a.scope, a.n_eggs, a.judged, a.correct), ("A", 3, 2, 2))
        self.assertAlmostEqual(a.coverage, 2 / 3)
        self.assertAlmostEqual(a.accuracy, 1.0)
        self.assertEqual((b.scope, b.judged, b.correct), ("B", 3, 2))
        self.assertAlmostEqual(b.accuracy, 2 / 3)

    def test_no_batch_info_gives_only_overall(self) -> None:
        batch = np.array([f"蛋{i}" for i in range(4)])
        self.assertEqual(len(holdout_table(np.array([0.9, 0.1, 0.8, 0.3]), np.array([1, 0, 1, 0]), batch)), 1)


class TestChanceStatus(unittest.TestCase):
    """“应该是瞎猜”的检查：统计量算不出来时绝不能算通过。"""

    def status(self, fold_acc: list[float]) -> str:
        return chance_status(np.array(fold_acc), 128, 32)[0]

    def test_all_folds_perfect_is_alarm(self) -> None:
        self.assertEqual(self.status([1.0] * 5), CHANCE_ALARM)

    def test_all_folds_exactly_half_is_not_confirmable(self) -> None:
        self.assertEqual(self.status([0.5] * 5), CHANCE_UNKNOWN)

    def test_varied_near_half_passes(self) -> None:
        self.assertEqual(self.status([0.46, 0.55, 0.50, 0.48, 0.53]), CHANCE_PASS)

    def test_clearly_above_half_with_variance_is_alarm(self) -> None:
        self.assertEqual(self.status([0.8, 0.82, 0.79, 0.85, 0.81]), CHANCE_ALARM)

    def test_single_fold_or_nan_is_not_confirmable(self) -> None:
        self.assertEqual(self.status([0.5]), CHANCE_UNKNOWN)
        self.assertEqual(self.status([np.nan, np.nan, np.nan]), CHANCE_UNKNOWN)

    def test_just_below_margin_and_zero_variance_is_not_confirmable(self) -> None:
        self.assertEqual(self.status([0.6] * 4), CHANCE_UNKNOWN)


class TestHoldoutLock(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.path = self.dir / "holdout_lock.json"
        self.ids = ("e2", "e1", "e3")

    def tearDown(self) -> None:
        shutil.rmtree(self.dir)

    def test_hash_ignores_order(self) -> None:
        self.assertEqual(holdout_lock.holdout_hash(self.ids), holdout_lock.holdout_hash(("e1", "e3", "e2")))
        self.assertNotEqual(holdout_lock.holdout_hash(self.ids), holdout_lock.holdout_hash(("e1", "e2")))

    def test_counts_views(self) -> None:
        h = holdout_lock.holdout_hash(self.ids)
        self.assertEqual(holdout_lock.prior_views(self.path, h), 0)
        self.assertEqual(holdout_lock.record_view(self.path, h, "run1", "PLS-DA", {"决策效用": 0.7}), 1)
        self.assertEqual(holdout_lock.prior_views(self.path, h), 1)
        self.assertEqual(holdout_lock.record_view(self.path, h, "run2", "深度-累计", {}), 2)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        evaluations = saved["sets"][h]["evaluations"]
        self.assertEqual([e["运行"] for e in evaluations], ["run1", "run2"])
        self.assertEqual(evaluations[0]["选定方法"], "PLS-DA")

    def test_new_holdout_set_starts_over(self) -> None:
        holdout_lock.record_view(self.path, holdout_lock.holdout_hash(self.ids), "run1", "x", {})
        other = holdout_lock.holdout_hash(("e9",))
        self.assertEqual(holdout_lock.prior_views(self.path, other), 0)
        self.assertEqual(holdout_lock.record_view(self.path, other, "run2", "x", {}), 1)

    def test_switching_holdout_sets_keeps_history(self) -> None:
        a, b = holdout_lock.holdout_hash(self.ids), holdout_lock.holdout_hash(("e9",))
        holdout_lock.record_view(self.path, a, "run1", "x", {})
        holdout_lock.record_view(self.path, b, "run2", "x", {})
        self.assertEqual(holdout_lock.prior_views(self.path, a), 1)
        self.assertEqual(holdout_lock.record_view(self.path, a, "run3", "x", {}), 2)
        self.assertEqual(holdout_lock.prior_views(self.path, b), 1)
        sets = json.loads(self.path.read_text(encoding="utf-8"))["sets"]
        self.assertEqual([e["运行"] for e in sets[a]["evaluations"]], ["run1", "run3"])
        self.assertEqual([e["运行"] for e in sets[b]["evaluations"]], ["run2"])

    def test_old_single_record_format_is_migrated(self) -> None:
        h = holdout_lock.holdout_hash(self.ids)
        old = {"hash": h, "evaluations": [{"时间": "t", "运行": "run0", "选定方法": "x", "成绩": {}}]}
        self.path.write_text(json.dumps(old), encoding="utf-8")
        self.assertEqual(holdout_lock.prior_views(self.path, h), 1)
        self.assertEqual(holdout_lock.record_view(self.path, h, "run1", "x", {}), 2)
        saved = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual([e["运行"] for e in saved["sets"][h]["evaluations"]], ["run0", "run1"])
        self.assertNotIn("hash", saved)

    def test_corrupt_lock_fails_loudly(self) -> None:
        self.path.write_text("not json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "读不了"):
            holdout_lock.prior_views(self.path, "x")

    def test_lock_path(self) -> None:
        self.assertEqual(holdout_lock.lock_path(str(self.dir / "meta.csv")), self.path)       # 不在 _cache 里
        self.assertEqual(holdout_lock.lock_path(None).name, "holdout_lock.json")


if __name__ == "__main__":
    unittest.main()
