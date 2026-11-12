"""可信度门控：自检/背景负对照没通过不发布模型；独立测试集只能正式评估一次；真实数据缺批次信息要明确报错。

这里是不训练模型的快速测试；整条链上的同类检查在 test_e2e.py。
"""
import argparse
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import gui  # noqa: E402
import holdout_lock  # noqa: E402
import real_data  # noqa: E402
import report  # noqa: E402
import train_demo  # noqa: E402
from evaluation import ScopeStats  # noqa: E402
from run_record import EXIT_HOLDOUT_USED, EXIT_SELF_CHECK_FAILED  # noqa: E402
from synth_data import EggDataset  # noqa: E402

PASS, ALARM, UNKNOWN = "通过", "报警", "无法确认"


def stub(n_batches: int, per_batch: int = 4) -> EggDataset:
    """只有标签和批次有意义的小数据集：批次 B0、B1……"""
    n = n_batches * per_batch
    return EggDataset(egg_ids=tuple(f"e{i:03d}" for i in range(n)), x=np.zeros((n, 1, 1, 1, 1), dtype=np.float32),
                      spec=np.zeros((n, 1, 1), dtype=np.float32), mask=np.ones((n, 1), dtype=bool),
                      sex=np.arange(n) % 2, dev=np.zeros((n, 1), dtype=np.float32),
                      dev_mask=np.zeros((n, 1), dtype=bool), batch=np.array([f"B{i // per_batch}" for i in range(n)]),
                      days=np.array([0.0], dtype=np.float32))


def background(status: str) -> dict:
    return {"批次可分性": None, "背景对照": {"各折准确率": [0.9, 0.9], "平均准确率": 0.9, "单侧p": 0.001, "结果": status}}


NO_CONTROLS = {"批次可分性": None, "背景对照": None}


class TestReleaseBlockers(unittest.TestCase):
    def test_nothing_checked_or_all_passed_releases(self) -> None:
        self.assertEqual(train_demo.release_blockers(None, NO_CONTROLS), [])
        check = train_demo.check_record({"a": PASS, "b": PASS})
        self.assertEqual(train_demo.release_blockers(check, background(PASS)), [])

    def test_failed_self_check_blocks(self) -> None:
        for status in (ALARM, UNKNOWN):
            reasons = train_demo.release_blockers(train_demo.check_record({"a": PASS, "b": status}), NO_CONTROLS)
            self.assertEqual(len(reasons), 1)
            self.assertIn("流程自检未通过", reasons[0])
            self.assertIn(f"b：{status}", reasons[0])
            self.assertNotIn("a：", reasons[0])

    def test_failed_background_control_blocks_even_if_self_check_passed(self) -> None:
        for status in (ALARM, UNKNOWN):
            reasons = train_demo.release_blockers(train_demo.check_record({"a": PASS}), background(status))
            self.assertEqual(len(reasons), 1)
            self.assertIn("背景负对照", reasons[0])

    def test_both_failures_are_both_reported(self) -> None:
        reasons = train_demo.release_blockers(train_demo.check_record({"a": ALARM}), background(ALARM))
        self.assertEqual(len(reasons), 2)


class TestFinishGate(unittest.TestCase):
    """finish() 在门控没通过时，不训练最终模型、不看独立测试集、不保存。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.data, self.hold = stub(4), stub(2)
        self.cv = mock.Mock(fold_of_egg=np.zeros(len(self.data.sex), dtype=int), n_train=8.0, n_test=8.0,
                            best_epochs=(1, 1), dev_pred=np.zeros((len(self.data.sex), 1)))
        self.args = argparse.Namespace(data=None, decision_day=0, debug_holdout=False, fertility_check=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def finish(self, statuses: dict | None, controls: dict) -> tuple[dict, mock.Mock, mock.Mock, mock.Mock]:
        decision = {"深度模型明显更好": False, "最好的基线": "PLS-DA"}
        holdout_extra = {"holdout_views": 1}
        with mock.patch.object(train_demo, "decide", return_value=decision), \
                mock.patch.object(train_demo.report, "log_dev", return_value=[]), \
                mock.patch.object(train_demo, "self_check", return_value=statuses), \
                mock.patch.object(train_demo, "run_controls", return_value=controls), \
                mock.patch.object(train_demo.deploy, "train_final") as train_final, \
                mock.patch.object(train_demo, "run_holdout", return_value=holdout_extra) as run_holdout, \
                mock.patch.object(train_demo.deploy, "save") as save:
            extra = train_demo.finish(self.data, self.hold, self.cv, {}, self.args, 0.0, self.dir)
        return extra, train_final, run_holdout, save

    def assert_blocked(self, statuses: dict | None, controls: dict, reason: str) -> None:
        extra, train_final, run_holdout, save = self.finish(statuses, controls)
        train_final.assert_not_called()
        run_holdout.assert_not_called()
        save.assert_not_called()
        self.assertFalse(extra["模型发布"]["已发布"])
        self.assertIn(reason, "；".join(extra["模型发布"]["未发布原因"]))
        self.assertNotIn("独立测试集", extra)
        self.assertNotIn("holdout_views", extra)

    def test_failed_self_check_stops_before_final_model(self) -> None:
        self.assert_blocked({"a": ALARM}, NO_CONTROLS, "流程自检未通过")
        self.assert_blocked({"a": UNKNOWN}, NO_CONTROLS, "流程自检未通过")

    def test_failed_background_control_stops_before_final_model(self) -> None:
        self.assert_blocked(None, background(ALARM), "背景负对照")
        self.assert_blocked({"a": PASS}, background(UNKNOWN), "背景负对照")

    def test_passing_checks_train_evaluate_and_save(self) -> None:
        extra, train_final, run_holdout, save = self.finish({"a": PASS}, background(PASS))
        train_final.assert_called_once()
        run_holdout.assert_called_once()
        save.assert_called_once()
        self.assertEqual(extra["模型发布"], {"已发布": True, "未发布原因": []})
        self.assertEqual(extra["holdout_views"], 1)


class TestHoldoutOnlyOnce(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.csv = str(self.dir / "meta.csv")
        self.lock = self.dir / "holdout_lock.json"
        data = stub(10)
        self.hold = train_demo.subset(data, np.flatnonzero(np.isin(data.batch, ["B2", "B7"])))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def args(self, skip: bool = False, debug: bool = False) -> argparse.Namespace:
        return argparse.Namespace(data=self.csv, skip_holdout=skip, debug_holdout=debug)

    def view(self, debug: bool = False) -> dict:
        table = {"x": [ScopeStats("全部", len(self.hold.sex), len(self.hold.sex), 1, 0.5, 0.5)]}
        with mock.patch.object(train_demo.deploy, "evaluate_holdout", return_value=([], table)) as evaluate:
            extra = train_demo.run_holdout(None, self.hold, self.args(debug=debug), 0, "x", self.dir)
        self.formal_flag = evaluate.call_args.args[-1]
        return extra

    def views(self) -> int:
        return holdout_lock.prior_views(self.lock, holdout_lock.view_key(self.hold.batch))

    def test_unviewed_holdout_passes_the_guard(self) -> None:
        self.assertIs(train_demo.guard_holdout(self.hold, self.args()), self.hold)
        self.assertIsNone(train_demo.guard_holdout(None, self.args()))

    def test_first_view_is_formal(self) -> None:
        extra = self.view()
        self.assertEqual((extra["holdout_views"], extra["正式结果"], self.formal_flag), (1, True, True))
        self.assertEqual(extra["性质"], report.FORMAL_RESULT)

    def test_second_formal_evaluation_is_blocked_before_training(self) -> None:
        self.view()
        with self.assertRaises(train_demo.HoldoutUsedError) as caught:
            train_demo.guard_holdout(self.hold, self.args())
        for option in ("--holdout-batches", "--skip-holdout", "--debug-holdout"):
            self.assertIn(option, str(caught.exception))
        self.assertEqual(self.views(), 1)

    def test_second_formal_evaluation_is_blocked_at_evaluation_too(self) -> None:
        self.view()
        with self.assertRaises(train_demo.HoldoutUsedError):
            self.view()
        self.assertEqual(self.views(), 1)                                   # 被拦下的不算查看

    def test_skip_holdout_returns_no_holdout_and_does_not_count(self) -> None:
        self.view()
        self.assertIsNone(train_demo.guard_holdout(self.hold, self.args(skip=True)))
        self.assertEqual(self.views(), 1)

    def test_debug_mode_views_again_but_never_formal(self) -> None:
        self.view()
        self.assertIs(train_demo.guard_holdout(self.hold, self.args(debug=True)), self.hold)
        extra = self.view(debug=True)
        self.assertEqual((extra["holdout_views"], extra["正式结果"], self.formal_flag), (2, False, False))
        self.assertEqual(extra["性质"], report.NOT_INDEPENDENT)
        evaluations = holdout_lock._load(self.lock)[0][holdout_lock.view_key(self.hold.batch)]["evaluations"]
        self.assertEqual([e["正式结果"] for e in evaluations], [True, False])

    def test_debug_rows_are_labelled_not_independent(self) -> None:
        tables = {"x": [ScopeStats("全部", 8, 8, 6, 0.75, 0.8), ScopeStats("B2", 4, 4, 3, 0.75, 0.8)]}
        formal = report.holdout_rows(4, tables, "x", 1)
        debug = report.holdout_rows(4, tables, "x", 2, formal=False)
        self.assertEqual([r["正式结果"] for r in formal], [1, 1])
        self.assertEqual([r["正式结果"] for r in debug], [0, 0])
        self.assertTrue(all(r["性质"] == report.NOT_INDEPENDENT for r in debug))
        self.assertIn("非独立验证", train_demo.HOLDOUT_DEBUG_CSV)
        self.assertNotEqual(train_demo.HOLDOUT_DEBUG_CSV, train_demo.HOLDOUT_CSV)

    def test_exit_code_and_gui_message(self) -> None:
        self.assertEqual((EXIT_SELF_CHECK_FAILED, EXIT_HOLDOUT_USED), (2, 3))
        message = gui.failure_message("训练", EXIT_HOLDOUT_USED)
        self.assertTrue(message.startswith(gui.HOLDOUT_MARK))
        self.assertNotIn("截图", message)


class TestBatchRequired(unittest.TestCase):
    """真实数据缺批次信息：默认明确报错，不再悄悄退化成每颗蛋自成一组。"""

    @staticmethod
    def args(data: str | None = "meta.csv", allow: bool = False) -> argparse.Namespace:
        return argparse.Namespace(data=data, allow_no_batch=allow)

    @staticmethod
    def without_batch(data: EggDataset, eggs: tuple[str, ...]) -> EggDataset:
        batch = np.array([real_data._solo_batch(e) if e in eggs else str(b) for e, b in zip(data.egg_ids, data.batch)])
        return train_demo.replace(data, batch=batch)

    def test_complete_batches_pass(self) -> None:
        self.assertEqual(real_data.eggs_without_batch(stub(3)), [])
        train_demo.check_batches(stub(3), self.args())

    def test_missing_batch_is_an_explicit_error(self) -> None:
        data = stub(3)
        solo = self.without_batch(data, data.egg_ids)
        self.assertEqual(real_data.eggs_without_batch(solo), list(data.egg_ids))
        with self.assertRaisesRegex(ValueError, "batch") as caught:
            train_demo.check_batches(solo, self.args())
        self.assertIn("--allow-no-batch", str(caught.exception))
        self.assertIn(f"{len(data.egg_ids)} 颗蛋", str(caught.exception))

    def test_partly_missing_batch_names_the_eggs(self) -> None:
        partial = self.without_batch(stub(3), ("e005",))
        with self.assertRaisesRegex(ValueError, "e005"):
            train_demo.check_batches(partial, self.args())

    def test_allow_no_batch_downgrades_to_loud_warning(self) -> None:
        data = stub(3)
        solo = self.without_batch(data, data.egg_ids)
        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs("egg_train", level="WARNING") as logs:
                train_demo.check_batches(solo, self.args(allow=True))
        finally:
            logging.disable(logging.WARNING)
        self.assertTrue(any("没有批次信息" in line for line in logs.output))

    def test_synthetic_data_is_not_checked(self) -> None:
        data = stub(3)
        train_demo.check_batches(self.without_batch(data, data.egg_ids), self.args(data=None))


if __name__ == "__main__":
    unittest.main()
