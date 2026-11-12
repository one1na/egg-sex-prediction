"""补充统计分析：概率校准、Holm 校正、逐日比较、失败案例；重复交叉验证。"""
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)

import analysis  # noqa: E402
import holdout_lock  # noqa: E402
import repeat_cv  # noqa: E402
import train_demo  # noqa: E402
from synth_data import make_dataset  # noqa: E402
from test_e2e import FOLDS, EndToEndBase, read_csv  # noqa: E402

nan = float("nan")


class TestCalibration(unittest.TestCase):
    def test_perfectly_confident_and_right(self) -> None:
        c = analysis.calibration(np.array([1.0, 0.0, 1.0]), np.array([1, 0, 1]))
        self.assertEqual((c.n, c.brier, c.ece), (3, 0.0, 0.0))

    def test_always_half(self) -> None:
        c = analysis.calibration(np.full(4, 0.5), np.array([1, 0, 1, 0]))
        self.assertAlmostEqual(c.brier, 0.25)
        self.assertAlmostEqual(c.ece, 0.0)              # 说 50%，实际也是 50%：校准得很好，只是没用

    def test_overconfident(self) -> None:
        c = analysis.calibration(np.full(4, 0.95), np.array([1, 1, 0, 0]))
        self.assertAlmostEqual(c.ece, 0.45)             # 说 95% 是雄，实际只有一半
        self.assertEqual(c.bins, [{"范围": "0.9~1.0", "蛋数": 4, "平均分数": 0.95, "实际雄的比例": 0.5}])

    def test_nan_skipped_and_scores_clipped(self) -> None:
        c = analysis.calibration(np.array([nan, 1.7, -0.3]), np.array([0, 1, 0]))
        self.assertEqual((c.n, c.brier), (2, 0.0))

    def test_report_marks_regression_scores(self) -> None:
        scores = {"深度-累计": np.array([[0.9], [0.1]]), "PLS-DA": np.array([[1.2], [-0.1]])}
        summary, rows = analysis.calibration_report(scores, np.array([1, 0]), 0)
        self.assertTrue(summary["深度-累计"]["分数是概率"])
        self.assertFalse(summary["PLS-DA"]["分数是概率"])
        self.assertEqual({r["方法"] for r in rows}, set(scores))


class TestHolm(unittest.TestCase):
    def test_known_values(self) -> None:
        np.testing.assert_allclose(analysis.holm(np.array([0.01, 0.04, 0.03])), [0.03, 0.06, 0.06])

    def test_capped_and_nan_kept(self) -> None:
        out = analysis.holm(np.array([0.5, nan, 0.6]))
        self.assertTrue(np.isnan(out[1]))
        np.testing.assert_allclose(out[[0, 2]], [1.0, 1.0])

    def test_never_smaller_than_raw(self) -> None:
        p = np.random.default_rng(0).uniform(0, 0.2, 8)
        self.assertTrue(np.all(analysis.holm(p) >= p))


class TestPerDayComparison(unittest.TestCase):
    def test_decision_day_not_adjusted_others_are(self) -> None:
        rng = np.random.default_rng(1)
        deep = 0.7 + rng.normal(0, 0.03, (5, 3))
        base = 0.6 + rng.normal(0, 0.03, (5, 3))
        rows = analysis.per_day_comparison(np.array([0, 4, 8]), deep, base, 1, 80.0, 20.0)
        self.assertEqual(rows[1]["校正后p"], rows[1]["单侧p"])
        self.assertIn("主检验", rows[1]["性质"])
        for r in (rows[0], rows[2]):
            self.assertIn("Holm", r["性质"])
            self.assertGreaterEqual(r["校正后p"], r["单侧p"])

    def test_day_without_data(self) -> None:
        deep = np.array([[0.7, nan], [0.8, nan]])
        rows = analysis.per_day_comparison(np.array([0, 4]), deep, deep - 0.1, 0, 80.0, 20.0)
        self.assertIsNone(rows[1]["差值"])
        self.assertFalse(rows[1]["深度明显更好"])


class TestFailureCases(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = {"深": np.array([0.95, 0.6, nan, 0.2]), "基": np.array([0.7, 0.3, 0.4, 0.1])}
        self.rows = analysis.failure_rows(("E1", "E2", "E3", "E4"), np.array(["B1", "B1", "B2", "B2"]),
                                          np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]),
                                          np.array([True, True, False, True]), np.array([3, 3, 1, 2]), self.scores)

    def test_confident_errors_first(self) -> None:
        self.assertEqual(self.rows[0]["egg_id"], "E1")            # 深 0.95 判雄，实际是雌
        self.assertEqual(self.rows[0]["深结果"], "判错")
        self.assertEqual(self.rows[0]["都判错"], 1)
        e3 = next(r for r in self.rows if r["egg_id"] == "E3")
        self.assertEqual((e3["深结果"], e3["基结果"], e3["深分数"]), ("没判断", "判错", None))

    def test_summary_and_overview(self) -> None:
        summary = analysis.failure_summary(self.rows, ["深", "基"])
        overall = next(s for s in summary if s["方法"] == "深" and s["分组"] == "全部")
        self.assertEqual((overall["蛋数"], overall["判错"], overall["没判断"]), (4, 1, 1))
        self.assertAlmostEqual(overall["错误率（判断了的蛋）"], 1 / 3, places=4)
        self.assertAlmostEqual(overall["决策效用"], 0.5)
        info = analysis.failure_overview(self.rows, summary, ["深", "基"])
        self.assertEqual(info["深"]["很有把握却判错"], 1)
        self.assertEqual(info["都判错的蛋数"], 1)
        self.assertEqual(info["深"]["错误率最高的批次"]["批次"], "B1")
        self.assertEqual(set(info["基"]["按性别的错误率"]), {"雄", "雌"})


class TestRepeatSummary(unittest.TestCase):
    def test_spread_and_stability(self) -> None:
        rows = [{"深度决策效用": 0.8, "基线决策效用": 0.7, "差值": 0.1, "深度明显更好": True, "主比较基线": "A",
                 "按规则部署": "深"},
                {"深度决策效用": 0.6, "基线决策效用": 0.7, "差值": -0.1, "深度明显更好": False, "主比较基线": "A",
                 "按规则部署": "A"}]
        s = repeat_cv.summarize(rows)
        self.assertEqual(s["深度决策效用"]["平均"], 0.7)
        self.assertEqual(s["深度明显更好的遍数"], 1)
        self.assertFalse(s["部署决定是否每遍相同"])
        self.assertNotIn("最好的基线是否每遍相同", s)               # 主比较基线事先指定，不会每遍变


TINY = mock.patch.object(train_demo, "make_dataset", side_effect=lambda seed: make_dataset(32, seed))


class TestPipelineOutputs(EndToEndBase):
    def test_training_writes_supplementary_analysis(self) -> None:
        with TINY:
            run = self.train()
        extra = self.config(run)["补充分析"]
        self.assertIn("深度-累计", extra["概率校准（决策日）"])
        self.assertIn("都判错的蛋数", extra["失败案例（决策日）"])
        days = read_csv(run / "per_day_comparison.csv")
        self.assertEqual(sum("主检验" in r["性质"] for r in days), 1)
        cases = read_csv(run / "failure_cases.csv")
        self.assertEqual(len(cases), len(read_csv(run / "folds.csv")))         # 每颗开发蛋一行
        for name in ("calibration.csv", "failure_summary.csv"):
            self.assertTrue((run / name).exists())

    def run_repeat(self, *args: str) -> Path:
        runs = self.out / "runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        with TINY, mock.patch.object(repeat_cv, "N_FOLDS", FOLDS), \
                mock.patch.object(sys, "argv", ["repeat_cv.py", *args]):
            self.assertEqual(repeat_cv.main(), 0)
        (new,) = set(runs.iterdir()) - before
        return new

    def test_repeat_cv_first_repeat_matches_training_and_holdout_untouched(self) -> None:
        with TINY:
            train_run = self.train("--skip-holdout")
        lock = self.out / "holdout_lock.json"
        before = lock.read_bytes()
        run = self.run_repeat("--repeats", "2")
        rows = read_csv(run / "repeat_cv.csv")
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["分折种子"], rows[1]["分折种子"])
        formal = self.config(train_run)["决策日结果"]["深度模型决策效用"]
        self.assertAlmostEqual(float(rows[0]["深度决策效用"]), formal, places=4)      # 第 1 遍 = 正式训练的分折
        self.assertEqual(lock.read_bytes(), before)                                   # 没碰独立测试集的记录
        self.assertFalse(list(run.glob("*.pt")) + list(run.glob("*.pkl")))
        config = self.config(run)
        self.assertEqual(config["类型"], "重复交叉验证")
        self.assertIn("复现", config)
        self.assertEqual(holdout_lock.read_split(lock) is not None, True)

    def test_repeats_must_be_at_least_two(self) -> None:
        with mock.patch.object(sys, "argv", ["repeat_cv.py", "--repeats", "1"]), \
                mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            repeat_cv.parse_args()


if __name__ == "__main__":
    unittest.main()
