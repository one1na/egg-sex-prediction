"""固定的独立测试批次：第一次划分后记在锁文件里，加了新数据也不会变；指定新测试批次的各种检查；旧锁文件搬家。"""
import argparse
import json
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

import holdout_lock  # noqa: E402
import train_demo  # noqa: E402
from evaluation import ScopeStats  # noqa: E402
from synth_data import EggDataset  # noqa: E402

PER_BATCH = 4


def stub(n_batches: int, per_batch: int = PER_BATCH) -> EggDataset:
    """只有标签和批次有意义的小数据集：批次 B0、B1……每批 4 颗蛋。"""
    n = n_batches * per_batch
    return EggDataset(egg_ids=tuple(f"e{i:03d}" for i in range(n)), x=np.zeros((n, 1, 1, 1, 1), dtype=np.float32),
                      spec=np.zeros((n, 1, 1), dtype=np.float32), mask=np.ones((n, 1), dtype=bool),
                      sex=np.arange(n) % 2, dev=np.zeros((n, 1), dtype=np.float32),
                      dev_mask=np.zeros((n, 1), dtype=bool), batch=np.array([f"B{i // per_batch}" for i in range(n)]),
                      days=np.array([0.0], dtype=np.float32))


class SplitBase(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        self.csv = str(self.dir / "meta.csv")
        self.lock = self.dir / "holdout_lock.json"

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)

    def split(self, n_batches: int, holdout_batches: tuple[str, ...] | None = None,
              shuffle: bool = False) -> tuple[EggDataset, EggDataset | None]:
        args = argparse.Namespace(data=self.csv, shuffle_labels=shuffle, holdout_batches=holdout_batches)
        return train_demo.split_holdout(stub(n_batches), args)

    @staticmethod
    def batches(data: EggDataset | None) -> set[str]:
        return set() if data is None else set(map(str, data.batch))


class TestFixedHoldout(SplitBase):
    def test_first_run_records_split(self) -> None:
        dev, hold = self.split(10)
        record = holdout_lock.read_split(self.lock)
        self.assertEqual(set(record.holdout_batches), self.batches(hold))
        self.assertEqual(set(record.dev_batches), self.batches(dev))
        self.assertEqual(len(record.holdout_batches) + len(record.dev_batches), 10)
        self.assertFalse(self.batches(dev) & self.batches(hold))

    def test_added_batches_go_to_dev_and_holdout_stays(self) -> None:
        _, hold = self.split(10)
        dev2, hold2 = self.split(12)
        self.assertEqual(self.batches(hold2), self.batches(hold))
        self.assertTrue({"B10", "B11"} <= self.batches(dev2))
        self.assertFalse({"B10", "B11"} & self.batches(hold2))
        record = holdout_lock.read_split(self.lock)
        self.assertTrue({"B10", "B11"} <= set(record.dev_batches))

    def test_missing_holdout_batch_is_error(self) -> None:
        _, hold = self.split(10)
        lost = sorted(self.batches(hold))[0]
        smaller = stub(10)
        keep = smaller.batch != lost
        args = argparse.Namespace(data=self.csv, shuffle_labels=False, holdout_batches=None)
        with self.assertRaisesRegex(ValueError, lost):
            train_demo.split_holdout(train_demo.subset(smaller, np.flatnonzero(keep)), args)

    def test_requested_former_dev_batch_is_error(self) -> None:
        dev, _ = self.split(10)
        former = sorted(self.batches(dev))[0]
        with self.assertRaisesRegex(ValueError, "以前用于开发"):
            self.split(12, ("B10", former))

    def test_requested_new_batch_is_accepted_and_recorded(self) -> None:
        _, old_hold = self.split(10)
        dev, hold = self.split(12, ("B10", "B11"))
        self.assertEqual(self.batches(hold), {"B10", "B11"})
        record = holdout_lock.read_split(self.lock)
        self.assertEqual(set(record.holdout_batches), {"B10", "B11"})
        self.assertFalse(record.holdout_viewed)
        self.assertTrue(self.batches(old_hold) <= set(record.dev_batches))      # 旧测试批次从此算开发批次
        _, again = self.split(12)                                               # 以后沿用新的
        self.assertEqual(self.batches(again), {"B10", "B11"})

    def test_viewed_holdout_cannot_be_requested_again(self) -> None:
        _, hold = self.split(10)
        holdout_lock.record_view(self.lock, "h", "run1", "x", {})
        self.assertTrue(holdout_lock.read_split(self.lock).holdout_viewed)
        with self.assertRaisesRegex(ValueError, "已经被查看过"):
            self.split(12, ("B10", sorted(self.batches(hold))[0]))

    def test_requested_unknown_batch_is_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "找不到"):
            self.split(10, ("B99",))

    def test_few_batches_no_holdout_but_old_batches_never_become_holdout(self) -> None:
        dev, hold = self.split(3)
        self.assertIsNone(hold)
        self.assertEqual(len(dev.sex), 3 * PER_BATCH)
        _, hold2 = self.split(8)                       # 批次够了，但前 3 个批次已用于开发，不自动留测试集
        self.assertIsNone(hold2)

    def test_new_batches_can_be_designated_after_few_batch_run(self) -> None:
        self.split(3)
        dev, hold = self.split(8, ("B5", "B6", "B7"))
        self.assertEqual(self.batches(hold), {"B5", "B6", "B7"})
        self.assertEqual(self.batches(dev), {f"B{i}" for i in range(5)})

    def test_shuffle_does_not_touch_lock(self) -> None:
        dev, hold = self.split(10, shuffle=True)
        self.assertFalse(self.lock.exists())
        self.assertIsNotNone(hold)


class TestLockMigration(SplitBase):
    def test_old_cache_lock_is_migrated(self) -> None:
        old = self.dir / "_cache" / "holdout_lock.json"
        old.parent.mkdir()
        sets = {"abc": {"egg_ids_n": 8, "evaluations": [{"时间": "t", "运行": "run0", "选定方法": "x", "成绩": {}}]}}
        old.write_text(json.dumps({"sets": sets}), encoding="utf-8")
        with self.assertLogs("egg_train", level="INFO") as logs:
            logging.disable(logging.NOTSET)
            try:
                self.split(10)
            finally:
                logging.disable(logging.WARNING)
        self.assertTrue(any("搬到" in line for line in logs.output))
        self.assertEqual(holdout_lock.prior_views(self.lock, "abc"), 1)             # 查看次数跟着搬过来
        self.assertIsNotNone(holdout_lock.read_split(self.lock))
        self.assertTrue(old.exists())

    def test_lock_not_in_cache_dir(self) -> None:
        self.assertEqual(holdout_lock.lock_path(self.csv), self.lock)
        with mock.patch.object(holdout_lock, "SYNTH_HOLDOUT_LOCK", self.dir / "s.json"):
            self.assertEqual(holdout_lock.lock_path(None), self.dir / "s.json")


class TestViewKey(SplitBase):
    """查看次数按测试批次名单记；往测试批次里加蛋不清零，只提醒。没有批次信息时按蛋编号指纹。"""

    def view(self, hold: EggDataset) -> int:
        args = argparse.Namespace(data=self.csv, debug_holdout=True)     # 调试模式才允许重复查看；这里只测计数
        table = {"x": [ScopeStats("全部", len(hold.sex), len(hold.sex), 1, 0.5, 0.5)]}
        with mock.patch.object(train_demo.deploy, "evaluate_holdout", return_value=([], table)):
            extra = train_demo.run_holdout(None, hold, args, 0, "x", self.dir)
        return extra["holdout_views"]

    def hold_of(self, per_batch: int) -> EggDataset:
        data = stub(10, per_batch)
        return train_demo.subset(data, np.flatnonzero(np.isin(data.batch, ["B2", "B7"])))

    def test_adding_eggs_to_viewed_batch_continues_count_with_warning(self) -> None:
        self.assertEqual(self.view(self.hold_of(4)), 1)
        logging.disable(logging.NOTSET)
        try:
            with self.assertLogs("egg_train", level="WARNING") as logs:
                self.assertEqual(self.view(self.hold_of(5)), 2)
        finally:
            logging.disable(logging.WARNING)
        self.assertTrue(any("和上次查看时不一样" in line for line in logs.output))
        sets = json.loads(self.lock.read_text(encoding="utf-8"))["sets"]
        self.assertEqual(list(sets), ["batches:B2,B7"])
        self.assertEqual(self.view(self.hold_of(5)), 3)

    def test_without_batch_info_key_is_fixed_name_list(self) -> None:
        data = stub(2, 4)
        solo = train_demo.replace(data, batch=np.array([f"蛋{e}" for e in data.egg_ids]))
        key = holdout_lock.view_key(solo.batch)
        self.assertTrue(key.startswith(holdout_lock.BATCH_KEY_PREFIX))
        self.assertEqual(self.view(solo), 1)
        self.assertIn(key, json.loads(self.lock.read_text(encoding="utf-8"))["sets"])
        many = train_demo.replace(stub(10, 4), batch=np.array([f"蛋{i}" for i in range(40)]))
        self.assertEqual(holdout_lock.view_key(many.batch), holdout_lock.view_key(many.batch[::-1]))
        self.assertLess(len(holdout_lock.view_key(many.batch)), 60)              # 名单太长时记指纹

    def test_old_hash_keyed_record_is_merged_into_batch_key(self) -> None:
        hold = self.hold_of(4)
        old_hash = holdout_lock.holdout_hash(hold.egg_ids)
        entry = {"时间": "t", "运行": "run0", "选定方法": "x", "成绩": {}}
        other = {"egg_ids_n": 3, "evaluations": [entry]}
        self.lock.write_text(json.dumps({"sets": {old_hash: {"egg_ids_n": 8, "evaluations": [entry]},
                                                  "unrelated": other}}), encoding="utf-8")
        self.assertEqual(self.view(hold), 2)
        sets = json.loads(self.lock.read_text(encoding="utf-8"))["sets"]
        self.assertNotIn(old_hash, sets)
        self.assertIn("unrelated", sets)                          # 对不上的旧记录原样保留
        runs = [e["运行"] for e in sets["batches:B2,B7"]["evaluations"]]
        self.assertEqual(runs[0], "run0")
        self.assertEqual(len(runs), 2)

    def test_shrinking_to_one_egg_per_batch_keeps_count(self) -> None:
        self.assertEqual(self.view(self.hold_of(4)), 1)
        self.assertEqual(self.view(self.hold_of(1)), 2)                          # 每批只剩一颗蛋，键不变
        self.assertEqual(list(json.loads(self.lock.read_text(encoding="utf-8"))["sets"]), ["batches:B2,B7"])


class TestLegacyLockWithoutSplit(SplitBase):
    """旧版本的锁只有查看次数、没有批次划分：补划分时不能让训练过的批次混进测试集。"""

    def write_legacy(self, key: str) -> None:
        entry = {"时间": "t", "运行": "run0", "选定方法": "x", "成绩": {}}
        self.lock.write_text(json.dumps({"sets": {key: {"egg_ids_n": 8, "evaluations": [entry]}}}), encoding="utf-8")

    def test_recognised_old_holdout_is_kept_and_marked_viewed(self) -> None:
        data = stub(10)
        old = train_demo._fresh_holdout_batches(data)
        eggs = tuple(e for e, b in zip(data.egg_ids, data.batch) if b in old)
        self.write_legacy(holdout_lock.holdout_hash(eggs))
        _, hold = self.split(10)
        self.assertEqual(self.batches(hold), set(old))
        record = holdout_lock.read_split(self.lock)
        self.assertTrue(record.holdout_viewed)
        _, hold2 = self.split(12)                                                # 加数据后测试批次不变
        self.assertEqual(self.batches(hold2), set(old))

    def test_unrecognised_old_holdout_makes_all_batches_dev(self) -> None:
        self.write_legacy("old-unknown-hash")
        dev, hold = self.split(12)
        self.assertIsNone(hold)
        self.assertEqual(set(holdout_lock.read_split(self.lock).dev_batches), self.batches(dev))
        with self.assertRaisesRegex(ValueError, "以前用于开发"):
            self.split(12, ("B0",))
        _, fresh = self.split(14, ("B12", "B13"))                                # 新批次可以当测试集
        self.assertEqual(self.batches(fresh), {"B12", "B13"})


if __name__ == "__main__":
    unittest.main()
