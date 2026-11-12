"""复现记录：数据指纹、代码版本、依赖版本核对、同样种子训练结果逐位相同。"""
import logging
import shutil
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import deep_train as dt  # noqa: E402
import run_record  # noqa: E402
from synth_data import make_dataset  # noqa: E402


class TestDataFingerprint(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = make_dataset(n_eggs=12, seed=0)

    def test_same_data_same_fingerprint(self) -> None:
        again = make_dataset(n_eggs=12, seed=0)
        self.assertEqual(run_record.data_fingerprint(self.data), run_record.data_fingerprint(again))

    def test_label_change_only_moves_label_hash(self) -> None:
        sex = self.data.sex.copy()
        sex[0] = 1 - sex[0]
        a, b = run_record.data_fingerprint(self.data), run_record.data_fingerprint(replace(self.data, sex=sex))
        self.assertEqual(a["图像"], b["图像"])
        self.assertNotEqual(a["标签"], b["标签"])
        self.assertNotEqual(a["整体"], b["整体"])

    def test_image_change_only_moves_image_hash(self) -> None:
        x = self.data.x.copy()
        x[0, 0, 0, 0, 0] += 1e-6
        a, b = run_record.data_fingerprint(self.data), run_record.data_fingerprint(replace(self.data, x=x))
        self.assertNotEqual(a["图像"], b["图像"])
        self.assertEqual(a["标签"], b["标签"])

    def test_batch_rename_changes_labels(self) -> None:
        batch = self.data.batch.astype(object).copy()
        batch[0] = "另一个批次"
        a, b = run_record.data_fingerprint(self.data), run_record.data_fingerprint(replace(self.data, batch=batch))
        self.assertNotEqual(a["标签"], b["标签"])

    def test_csv_hash(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "meta.csv").write_text("egg_id,day\nE1,0\n", encoding="utf-8")
        before = run_record.data_fingerprint(self.data, str(tmp / "meta.csv"))["数据表"]
        (tmp / "meta.csv").write_text("egg_id,day\nE1,2\n", encoding="utf-8")
        self.assertNotEqual(before, run_record.data_fingerprint(self.data, str(tmp / "meta.csv"))["数据表"])


class TestCodeVersion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_code_hash_sees_any_edit(self) -> None:
        (self.tmp / "a.py").write_text("X = 1\n", encoding="utf-8")
        (self.tmp / "sub").mkdir()
        (self.tmp / "sub" / "b.py").write_text("Y = 2\n", encoding="utf-8")
        before = run_record.code_hash(self.tmp)
        (self.tmp / "sub" / "b.py").write_text("Y = 3\n", encoding="utf-8")
        self.assertNotEqual(before, run_record.code_hash(self.tmp))

    def test_without_git(self) -> None:
        with mock.patch.object(run_record, "_git", return_value=None):
            info = run_record.code_version(self.tmp)
        self.assertIn("不可用", info["git"])
        self.assertIn("代码指纹", info)
        self.assertFalse((self.tmp / run_record.PATCH_NAME).exists())

    def test_clean_tree(self) -> None:
        answers = {"rev-parse": "abc123\n", "status": ""}
        with mock.patch.object(run_record, "_git", side_effect=lambda *a: answers.get(a[0])):
            info = run_record.code_version(self.tmp)
        self.assertEqual((info["git提交"], info["有未提交的改动"]), ("abc123", False))
        self.assertFalse((self.tmp / run_record.PATCH_NAME).exists())

    def test_dirty_tree_saves_patch(self) -> None:
        answers = {"rev-parse": "abc123\n", "status": " M code/roi.py\n?? code/新文件.py\n",
                   "diff": "diff --git a/code/roi.py b/code/roi.py\n+改动\n"}
        with mock.patch.object(run_record, "_git", side_effect=lambda *a: answers.get(a[0])):
            info = run_record.code_version(self.tmp)
        self.assertTrue(info["有未提交的改动"])
        self.assertEqual(info["改动的文件"], ["code/roi.py", "code/新文件.py"])
        self.assertIn("+改动", (self.tmp / run_record.PATCH_NAME).read_text(encoding="utf-8"))
        self.assertIn("说明", info)       # 新文件不在 patch 里，要提醒

    def test_real_repo_runs(self) -> None:
        info = run_record.code_version()      # 在这台机器的真实仓库上跑：不报错，有指纹
        self.assertEqual(len(info["代码指纹"]), 16)


class TestDependencyCheck(unittest.TestCase):
    def setUp(self) -> None:
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        self.req = tmp / "requirements.txt"

    def test_parse_pins(self) -> None:
        self.req.write_text("# 注释\nnumpy==2.4.6\nscipy >= 1.0\ntorch == 2.15.0  # cpu\n", encoding="utf-8")
        self.assertEqual(run_record.pinned_versions(self.req), {"numpy": "2.4.6", "torch": "2.15.0"})

    def test_local_suffix_ignored_and_mismatch_reported(self) -> None:
        self.req.write_text("torch==2.15.0\nnumpy==0.0.1\n", encoding="utf-8")
        versions = {"torch": "2.15.0+cpu", "numpy": "2.4.6"}
        with mock.patch.object(run_record.metadata, "version", side_effect=versions.__getitem__):
            res = run_record.dependency_check(self.req)
        self.assertFalse(res["一致"])
        self.assertEqual(len(res["不一致"]), 1)
        self.assertIn("numpy", res["不一致"][0])

    def test_shipped_requirements_are_pinned(self) -> None:
        pins = run_record.pinned_versions()
        self.assertEqual(set(pins), {"numpy", "scipy", "scikit-learn", "torch"})


class TestDeterminism(unittest.TestCase):
    def test_same_seed_bitwise_identical(self) -> None:
        data = make_dataset(n_eggs=12, seed=0)
        idx = np.arange(len(data.sex))
        outs = []
        for _ in range(2):
            run_record.set_determinism(7)
            trained = dt.train_fixed_epochs(data, idx, 0.5, seed=7, epochs=2)
            outs.append(dt.predict_scores(trained, data.x, data.mask, data.days)["cum"])
        np.testing.assert_array_equal(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
