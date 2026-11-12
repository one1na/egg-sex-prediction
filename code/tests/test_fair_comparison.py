"""公平比较：基线的同样本量对照（只用深度模型拟合的那部分蛋训练），以及以批次为单位的区间。"""
import argparse
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import report  # noqa: E402
import train_demo  # noqa: E402
from baselines import BASELINE_NAMES, MATCHED_NAMES, matched_name  # noqa: E402
from evaluation import (MIN_BATCHES_FOR_INTERVAL, Paired, ScopeStats, batch_utilities, group_interval,  # noqa: E402
                        inner_split)
from synth_data import EggDataset  # noqa: E402


def stub(n_batches: int, per_batch: int = 4) -> EggDataset:
    n = n_batches * per_batch
    return EggDataset(egg_ids=tuple(f"e{i:03d}" for i in range(n)), x=np.zeros((n, 1, 1, 1, 1), dtype=np.float32),
                      spec=np.zeros((n, 1, 1), dtype=np.float32), mask=np.ones((n, 1), dtype=bool),
                      sex=np.arange(n) % 2, dev=np.zeros((n, 1), dtype=np.float32),
                      dev_mask=np.zeros((n, 1), dtype=bool), batch=np.array([f"B{i // per_batch}" for i in range(n)]),
                      days=np.array([0.0], dtype=np.float32))


class TestGroupInterval(unittest.TestCase):
    def test_t_interval_over_batches(self) -> None:
        g = group_interval(np.array([0.5, 0.7, 0.9]))
        self.assertEqual(g.n, 3)
        self.assertAlmostEqual(g.mean, 0.7)
        half = 4.302652729911275 * 0.2 / np.sqrt(3)          # t(0.975, 2) × 标准差 / √n
        self.assertAlmostEqual(g.ci_low, 0.7 - half, places=6)
        self.assertAlmostEqual(g.ci_high, 0.7 + half, places=6)
        self.assertEqual((g.low, g.high), (0.5, 0.9))

    def test_too_few_batches_gives_no_interval(self) -> None:
        self.assertEqual(MIN_BATCHES_FOR_INTERVAL, 3)
        g = group_interval(np.array([0.5, 0.9]))
        self.assertEqual((g.n, g.low, g.high), (2, 0.5, 0.9))
        self.assertAlmostEqual(g.mean, 0.7)
        self.assertTrue(np.isnan(g.ci_low) and np.isnan(g.ci_high))

    def test_nan_is_ignored_and_empty_is_all_nan(self) -> None:
        self.assertEqual(group_interval(np.array([0.5, np.nan, 0.7, 0.9])).n, 3)
        empty = group_interval(np.array([np.nan]))
        self.assertEqual(empty.n, 0)
        self.assertTrue(np.isnan(empty.mean))

    def test_identical_batches_give_zero_width_interval(self) -> None:
        g = group_interval(np.array([0.75, 0.75, 0.75]))
        self.assertEqual((g.ci_low, g.ci_high), (0.75, 0.75))


class TestBatchUtilities(unittest.TestCase):
    def test_one_value_per_batch_and_unjudged_counts_as_wrong(self) -> None:
        sex = np.array([1, 0, 1, 0, 1, 0])
        batch = np.array(["A", "A", "A", "B", "B", "B"])
        scores = np.array([0.9, 0.1, 0.2, np.nan, 0.8, 0.3])      # A：对 对 错；B：没判 对 对
        values = batch_utilities(scores, sex, batch, np.array(["A", "B"]))
        np.testing.assert_allclose(values, [2 / 3, 2 / 3])
        np.testing.assert_allclose(batch_utilities(scores, sex, batch, np.array(["B", "A"])), [2 / 3, 2 / 3])
        all_wrong = batch_utilities(np.full(6, np.nan), sex, batch, np.array(["A", "B"]))
        np.testing.assert_allclose(all_wrong, [0.0, 0.0])


class TestPerBatchReport(unittest.TestCase):
    def setUp(self) -> None:
        self.names = np.array(["B0", "B1", "B2"])
        self.utility = {"深度-累计": np.array([0.75, 0.5, 1.0]), "PLS-DA": np.array([0.5, 0.5, 0.25])}
        self.diffs = {"PLS-DA": self.utility["深度-累计"] - self.utility["PLS-DA"]}

    def test_rows_have_one_line_per_batch(self) -> None:
        rows = report.per_batch_rows(self.names, np.array([4, 4, 8]), self.utility, self.diffs, "深度-累计")
        self.assertEqual([r["批次"] for r in rows], ["B0", "B1", "B2"])
        self.assertEqual([r["蛋数"] for r in rows], [4, 4, 8])
        self.assertEqual([r["深度-累计减PLS-DA"] for r in rows], [0.25, 0.0, 0.75])

    def test_summary_has_interval_and_win_counts(self) -> None:
        summary = report.per_batch_summary(self.utility, self.diffs, "深度-累计")
        diff = summary["差值"]["深度-累计减PLS-DA"]
        self.assertEqual((diff["更高批次数"], diff["持平批次数"], diff["更低批次数"]), (2, 1, 0))
        self.assertEqual(diff["批次数"], 3)
        low, high = diff["95%区间"]
        self.assertLess(low, diff["平均"])
        self.assertGreater(high, diff["平均"])
        self.assertIn("以批次为单位", summary["区间说明"])
        self.assertEqual(summary["各方法决策效用"]["PLS-DA"]["最低"], 0.25)

    def test_holdout_needs_two_batches(self) -> None:
        total = ScopeStats("全部", 8, 8, 6, 0.75, 0.8)
        self.assertIsNone(report.holdout_batch_interval([total]))
        self.assertIsNone(report.holdout_batch_interval([total, ScopeStats("B1", 8, 8, 6, 0.75, 0.8)]))
        two = report.holdout_batch_interval([total, ScopeStats("B1", 4, 4, 4, 1.0, 1.0),
                                             ScopeStats("B2", 4, 4, 2, 0.5, 0.5)])
        self.assertEqual((two["批次数"], two["平均"], two["95%区间"]), (2, 0.75, [None, None]))


class TestMatchedBaseline(unittest.TestCase):
    def test_names(self) -> None:
        self.assertEqual(len(MATCHED_NAMES), len(BASELINE_NAMES))
        self.assertEqual(MATCHED_NAMES[0], matched_name(BASELINE_NAMES[0]))
        self.assertFalse(set(MATCHED_NAMES) & set(BASELINE_NAMES))

    def test_fold_trains_baseline_on_all_train_eggs_and_on_fit_eggs_only(self) -> None:
        data = stub(10)
        train_idx, test_idx = np.arange(32), np.arange(32, 40)
        fit_idx, val_idx, _ = inner_split(train_idx, data.sex, data.batch, 7)
        nothing = np.zeros((len(test_idx), 1))
        deep = {"frame": nothing, "cum": nothing, "dev": nothing}
        calls = []

        def fake_baselines(spec, mask, sex, train, test):
            calls.append(np.array(train))
            return {name: np.full((len(test), 1), float(len(train))) for name in BASELINE_NAMES}

        with mock.patch.object(train_demo.dt, "train_early_stop", return_value=mock.Mock(epochs=3)), \
                mock.patch.object(train_demo.dt, "predict_scores", return_value=deep), \
                mock.patch.object(train_demo, "per_day_baselines", side_effect=fake_baselines):
            scores, _, epochs, n_fit = train_demo.run_fold(data, train_idx, test_idx, 0.0, 7)
        self.assertEqual((epochs, n_fit), (3, len(fit_idx)))
        self.assertLess(len(fit_idx), len(train_idx))
        np.testing.assert_array_equal(calls[0], train_idx)                     # 正式比较：全部训练蛋
        np.testing.assert_array_equal(calls[1], fit_idx)                       # 同样本量对照：只用拟合的那部分
        self.assertFalse(set(calls[1]) & set(val_idx))                         # 早停验证的蛋没进去
        for name in BASELINE_NAMES:
            self.assertEqual(scores[name][0, 0], len(train_idx))
            self.assertEqual(scores[matched_name(name)][0, 0], len(fit_idx))

    def test_matched_baselines_stay_out_of_self_check(self) -> None:
        metrics = {name: {} for name in ("深度-累计", *BASELINE_NAMES, *MATCHED_NAMES)}
        cv = mock.Mock(n_train=8.0, n_test=4.0)
        args = argparse.Namespace(shuffle_labels=True, data=None)
        with mock.patch.object(train_demo.report, "log_chance", return_value={"深度-累计": "通过"}) as log_chance:
            train_demo.self_check(stub(2), cv, metrics, args)
        checked = set(log_chance.call_args.args[0])
        self.assertEqual(checked, {"深度-累计", *BASELINE_NAMES})

    def test_matched_comparison_reports_and_flags_disagreement(self) -> None:
        clear = Paired(0.2, 0.1, 0.3, 0.001)
        unclear = Paired(0.01, -0.1, 0.12, 0.4)
        self.assertTrue(report.log_matched("深度-累计", "PLS-DA·等样本", clear, "utility", True))
        self.assertFalse(report.log_matched("深度-累计", "PLS-DA·等样本", unclear, "utility", False))
        self.assertFalse(report.log_matched("深度-累计", "PLS-DA·等样本", Paired(0.0, np.nan, np.nan, np.nan),
                                            "utility", False))
        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs("egg_train", level="WARNING") as logs:
                report.log_matched("深度-累计", "PLS-DA·等样本", clear, "utility", False)
        finally:
            logging.disable(logging.WARNING)
        self.assertTrue(any("结论不一致" in line for line in logs.output))


if __name__ == "__main__":
    unittest.main()
