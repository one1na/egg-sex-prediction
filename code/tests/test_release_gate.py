"""可信度门控：自检/背景负对照没通过不发布模型；独立测试集只能正式评估一次；真实数据缺批次信息要明确报错。

这里是不训练模型的快速测试；整条链上的同类检查在 test_e2e.py。
"""
import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import gui_messages  # noqa: E402
import holdout_lock  # noqa: E402
import real_data  # noqa: E402
import report  # noqa: E402
import run_record  # noqa: E402
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
DEMO = argparse.Namespace(data=None, fertility_check=False)
FORMAL = argparse.Namespace(data="meta.csv", fertility_check=False)
REPRO = {"代码版本": {"代码指纹": "code1"}, "数据指纹": {"整体": "data1"}}


def gate(args: argparse.Namespace, statuses: dict | None, controls: dict, data: EggDataset | None = None,
         repro: dict | None = None) -> dict:
    return train_demo.gate_checks(args, data or stub(2), statuses, controls, repro)


def blockers(*a, **k) -> list[str]:
    return train_demo.release_blockers(gate(*a, **k))


class TestReleaseBlockers(unittest.TestCase):
    def test_all_passed_releases(self) -> None:
        self.assertEqual(blockers(DEMO, {"a": PASS, "b": PASS}, background(PASS)), [])

    def test_failed_self_check_blocks(self) -> None:
        for status in (ALARM, UNKNOWN):
            reasons = blockers(DEMO, {"a": PASS, "b": status}, NO_CONTROLS)
            self.assertEqual(len(reasons), 1)
            self.assertIn("流程自检未通过", reasons[0])
            self.assertIn(f"b：{status}", reasons[0])
            self.assertNotIn("a：", reasons[0])

    def test_failed_background_control_blocks_even_if_self_check_passed(self) -> None:
        for status in (ALARM, UNKNOWN):
            reasons = blockers(DEMO, {"a": PASS}, background(status))
            self.assertEqual(len(reasons), 1)
            self.assertIn("背景负对照", reasons[0])

    def test_both_failures_are_both_reported(self) -> None:
        self.assertEqual(len(blockers(DEMO, {"a": ALARM}, background(ALARM))), 2)

    def test_four_states(self) -> None:
        with_bg = train_demo.replace(stub(2), bg_spec=np.zeros((8, 1, 1), dtype=np.float32))
        g = gate(DEMO, None, NO_CONTROLS)
        self.assertEqual((g["模式"], g["流程自检"]["状态"], g["背景负对照"]["状态"]), ("演示", "未执行", "不适用"))
        self.assertEqual(gate(DEMO, None, NO_CONTROLS, data=with_bg)["背景负对照"]["状态"], "未执行")
        self.assertEqual(gate(DEMO, {"a": ALARM}, background(PASS))["流程自检"]["状态"], "未通过")

    def test_not_run_only_warns_in_demo_mode(self) -> None:
        self.assertEqual(blockers(DEMO, None, NO_CONTROLS), [])

    def test_not_run_blocks_in_formal_mode(self) -> None:
        """真实数据：没做打乱标签自检就不发布——"没有检查"不能当作"通过检查"。"""
        with mock.patch.object(train_demo, "find_self_check", return_value=None):
            reasons = blockers(FORMAL, None, background(PASS), repro=REPRO)
        self.assertEqual(len(reasons), 1)
        self.assertIn("流程自检未执行", reasons[0])
        self.assertIn("--shuffle-labels", reasons[0])
        for fertility in (True, False):
            mode = train_demo.gate_mode(argparse.Namespace(data=None, fertility_check=fertility))
            self.assertEqual(mode, "正式" if fertility else "演示")

    def test_background_not_run_blocks_in_formal_mode_but_not_applicable_does_not(self) -> None:
        with_bg = train_demo.replace(stub(2), bg_spec=np.zeros((8, 1, 1), dtype=np.float32))
        found = {"运行": "r1", "总体": PASS}
        with mock.patch.object(train_demo, "find_self_check", return_value=found):
            self.assertEqual(blockers(FORMAL, None, NO_CONTROLS, repro=REPRO), [])                 # 没有背景光谱
            reasons = blockers(FORMAL, None, NO_CONTROLS, data=with_bg, repro=REPRO)
        self.assertEqual(len(reasons), 1)
        self.assertIn("背景负对照未执行", reasons[0])


class TestFindSelfCheck(unittest.TestCase):
    """正式模式的发布条件：以前跑过一次打乱标签的自检，代码指纹和打乱前的数据指纹都和这次一样。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.patch = mock.patch.object(run_record, "RUNS_DIR", self.dir)
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()
        shutil.rmtree(self.dir, ignore_errors=True)

    def add(self, name: str, overall: str, code: str = "code1", data: str = "data1", shuffled: bool = True) -> None:
        (self.dir / name).mkdir()
        config = {"流程自检（打乱标签）": shuffled, "自检": {"总体": overall},
                  "复现": {"代码版本": {"代码指纹": code}, "数据指纹": {"整体": "shuffled"}, "打乱前数据指纹": {"整体": data}}}
        (self.dir / name / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    def test_matching_run_found_latest_wins(self) -> None:
        self.assertIsNone(train_demo.find_self_check(REPRO))
        self.add("2026-01-01_100000_流程自检", "未通过")
        self.add("2026-01-02_100000_流程自检", PASS)
        self.assertEqual(train_demo.find_self_check(REPRO), {"运行": "2026-01-02_100000_流程自检", "总体": PASS})
        self.add("2026-01-03_100000_流程自检", "未通过")
        found = train_demo.find_self_check(REPRO)
        self.assertEqual(found["总体"], "未通过")
        g = gate(FORMAL, None, NO_CONTROLS, repro=REPRO)
        self.assertEqual(g["流程自检"]["状态"], "未通过")

    def test_other_code_data_or_normal_runs_do_not_count(self) -> None:
        self.add("2026-01-01_100000_流程自检", PASS, code="old")
        self.add("2026-01-02_100000_流程自检", PASS, data="other")
        self.add("2026-01-03_100000_真实数据", PASS, shuffled=False)
        (self.dir / "2026-01-04_100000_坏的").mkdir()
        (self.dir / "2026-01-04_100000_坏的" / "config.json").write_text("{", encoding="utf-8")
        self.assertIsNone(train_demo.find_self_check(REPRO))
        self.assertIsNone(train_demo.find_self_check(None))


class TestFinishGate(unittest.TestCase):
    """finish() 在门控没通过时，不训练最终模型、不看独立测试集、不保存。"""

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.data, self.hold = stub(4), stub(2)
        self.cv = mock.Mock(fold_of_egg=np.zeros(len(self.data.sex), dtype=int), n_train=8.0, n_test=8.0,
                            best_epochs=(1, 1), dev_pred=np.zeros((len(self.data.sex), 1)))
        self.args = argparse.Namespace(data=None, decision_day=0, debug_holdout=False, fertility_check=False,
                                       shuffle_labels=False)

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def finish(self, statuses: dict | None, controls: dict) -> tuple[dict, mock.Mock, mock.Mock, mock.Mock]:
        decision = {"深度模型明显更好": False, "主比较基线": "PLS-DA"}
        holdout_extra = {"holdout_views": 1}
        with mock.patch.object(train_demo, "decide", return_value=decision), \
                mock.patch.object(train_demo.report, "log_dev", return_value=[]), \
                mock.patch.object(train_demo, "self_check", return_value=statuses), \
                mock.patch.object(train_demo, "run_controls", return_value=controls), \
                mock.patch.object(train_demo, "supplementary", return_value={}), \
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

    def test_formal_mode_without_self_check_stops_before_final_model(self) -> None:
        self.args.data = str(self.dir / "meta.csv")
        with mock.patch.object(run_record, "RUNS_DIR", self.dir / "runs"):
            self.assert_blocked(None, background(PASS), "流程自检未执行")

    def test_passing_checks_train_evaluate_and_save(self) -> None:
        extra, train_final, run_holdout, save = self.finish({"a": PASS}, background(PASS))
        train_final.assert_called_once()
        run_holdout.assert_called_once()
        save.assert_called_once()
        self.assertEqual(extra["模型发布"], {"已发布": True, "未发布原因": []})
        self.assertEqual(extra["holdout_views"], 1)
        self.assertEqual(extra["发布检查"]["流程自检"]["状态"], "通过")


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

    def test_interrupted_evaluation_counts_as_viewed(self) -> None:
        """先记后看：算成绩时程序崩了，这次也按看过算，下次正式评估被拦下。"""
        with mock.patch.object(train_demo.deploy, "evaluate_holdout", side_effect=RuntimeError("断电")), \
                self.assertRaises(RuntimeError):
            train_demo.run_holdout(None, self.hold, self.args(), 0, "x", self.dir)
        key = holdout_lock.view_key(self.hold.batch)
        self.assertEqual((self.views(), holdout_lock.unfinished_views(self.lock, key)), (1, 1))
        with self.assertRaises(train_demo.HoldoutUsedError) as caught:
            train_demo.guard_holdout(self.hold, self.args())
        self.assertIn("中途中断", str(caught.exception))
        self.assertTrue(holdout_lock.read_split(self.lock) is None or holdout_lock.read_split(self.lock).holdout_viewed)

    def test_finished_evaluation_is_marked_done_with_result(self) -> None:
        self.view()
        (entry,) = holdout_lock._load(self.lock)[0][holdout_lock.view_key(self.hold.batch)]["evaluations"]
        self.assertEqual((entry["状态"], entry["成绩"]["蛋数"]), ("完成", len(self.hold.sex)))

    def test_concurrent_runs_get_only_one_formal_view(self) -> None:
        """两个（这里是八个）训练同时跑到独立测试集：只有一个拿得到正式查看，其余被拦下。"""
        key = holdout_lock.view_key(self.hold.batch)
        start, results = threading.Barrier(8), []

        def claim(i: int) -> None:
            start.wait()
            try:
                results.append(holdout_lock.begin_view(self.lock, key, f"run{i}", "x", 8, "h"))
            except holdout_lock.AlreadyViewed:
                results.append("blocked")

        threads = [threading.Thread(target=claim, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(sorted(results, key=str), [0] + ["blocked"] * 7)
        self.assertEqual(self.views(), 1)
        self.assertFalse(list(self.dir.glob("*.busy")) + list(self.dir.glob("*.tmp")))

    def test_stale_busy_marker_is_cleared(self) -> None:
        busy = self.lock.with_name(self.lock.name + ".busy")
        busy.write_text("12345")
        old = busy.stat().st_mtime - holdout_lock.BUSY_STALE - 10
        os.utime(busy, (old, old))
        holdout_lock.record_view(self.lock, "k", "run1", "x", {})
        self.assertFalse(busy.exists())

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
        message = gui_messages.failure_message("训练", EXIT_HOLDOUT_USED)
        self.assertTrue(message.startswith(gui_messages.HOLDOUT_MARK))
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
