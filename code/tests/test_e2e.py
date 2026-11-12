"""端到端：极小的数据、极少的轮数，把 训练 → 保存 → 读取 → 判断 整条链跑一遍。所有输出都写到临时文件夹。"""
import contextlib
import csv
import json
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import baseline_fertility  # noqa: E402
import calibration as cal  # noqa: E402
import deep_train as dt  # noqa: E402
import deploy  # noqa: E402
import export_sample  # noqa: E402
import holdout_lock  # noqa: E402
import predict  # noqa: E402
import preprocess_binding  # noqa: E402
import real_data  # noqa: E402
import run_record  # noqa: E402
import synth_data  # noqa: E402
import train_demo  # noqa: E402
from real_data import EggFrames  # noqa: E402

N_EGGS, EGGS_PER_BATCH = 32, 4        # 8 个批次：够留出独立测试集
FOLDS, EPOCHS = 3, 2


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


class EndToEndBase(unittest.TestCase):
    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp())
        self.handlers = list(logging.getLogger().handlers)
        self.stack = contextlib.ExitStack()
        patches = [
            mock.patch.object(deploy, "MODEL_PATH", self.out / "egg_model.pt"),
            mock.patch.object(deploy, "BASELINE_MODEL_PATH", self.out / "baseline_model.pkl"),
            mock.patch.object(predict, "MODEL_PATH", self.out / "egg_model.pt"),
            mock.patch.object(predict, "BASELINE_MODEL_PATH", self.out / "baseline_model.pkl"),
            mock.patch.object(predict, "PREDICTIONS_CSV", self.out / "predictions.csv"),
            mock.patch.object(predict, "DECISION_CSV", self.out / "decision.csv"),
            mock.patch.object(run_record, "RUNS_DIR", self.out / "runs"),
            mock.patch.object(holdout_lock, "SYNTH_HOLDOUT_LOCK", self.out / "holdout_lock.json"),
            mock.patch.object(synth_data, "EGGS_PER_BATCH", EGGS_PER_BATCH),
            mock.patch.object(dt, "MAX_EPOCHS", EPOCHS), mock.patch.object(dt, "PATIENCE", EPOCHS),
            mock.patch.object(train_demo, "N_FOLDS", FOLDS),
        ]
        for p in patches:
            self.stack.enter_context(p)

    def tearDown(self) -> None:
        self.stack.close()
        root = logging.getLogger()
        for h in root.handlers[:]:
            if h not in self.handlers:
                h.close()
                root.removeHandler(h)
        shutil.rmtree(self.out, ignore_errors=True)

    def train(self, *args: str, force_release: bool = True) -> Path:
        """跑一次训练，返回这次新建的运行文件夹。

        force_release：这么小的数据上自检和负对照的统计结果说不准，测整条链时不让发布门控挡路；
        测门控本身的用例传 False。
        """
        runs = self.out / "runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(sys, "argv", ["train_demo.py", *args]))
            if force_release:
                stack.enter_context(mock.patch.object(train_demo, "release_blockers", return_value=[]))
            self.exit_code = train_demo.main()
        (new,) = set(runs.iterdir()) - before
        return new

    @staticmethod
    def config(run: Path) -> dict:
        return json.loads((run / "config.json").read_text(encoding="utf-8"))


class TestRealDataChain(EndToEndBase):
    @classmethod
    def setUpClass(cls) -> None:
        """样例数据只导出、预处理一次；每个测试用自己的副本（缓存和查看记录互不影响）。"""
        cls.master = Path(tempfile.mkdtemp())
        with mock.patch.object(synth_data, "EGGS_PER_BATCH", EGGS_PER_BATCH),                 mock.patch.object(export_sample, "OUT_DIR", cls.master),                 mock.patch.object(export_sample, "N_EGGS", N_EGGS):
            export_sample.main()
        real_data.load_real_dataset(str(cls.master / "meta.csv"))

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.master, ignore_errors=True)

    def setUp(self) -> None:
        super().setUp()
        self.sample = self.out / "sample"
        shutil.copytree(self.master, self.sample)
        self.csv = str(self.sample / "meta.csv")

    def test_load_has_wavelengths_and_qc(self) -> None:
        data = real_data.load_real_dataset(self.csv)
        self.assertEqual(data.wavelengths.shape, (data.x.shape[2],))
        qc = next((self.sample / "_cache").glob("*_roi_qc.csv"))
        self.assertTrue(all(r["失败原因"] == "" for r in read_csv(qc)))

    def test_train_save_load_predict(self) -> None:
        run = self.train("--data", self.csv)
        ckpt = torch.load(self.out / "egg_model.pt", weights_only=True)
        self.assertEqual(len(ckpt["wavelengths"]), ckpt["n_bands"])
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["holdout_views"], 1)
        self.assertEqual(config["主指标"], "utility")
        self.assertTrue((run / "decision_day_comparison.csv").exists())
        holdout = read_csv(run / "holdout.csv")
        self.assertIn("全部", {r["范围"] for r in holdout})
        self.assertTrue(any(r["范围"].startswith("B") for r in holdout))     # 有分批次的行
        self.assertEqual({r["方法"] for r in holdout}, {ckpt["chosen"]})       # 只评估事先选定的方法
        self.assertTrue(all(r["正式结果"] == "1" for r in holdout))
        self.assertIn("深度模型决策效用（合并按蛋）", config["决策日结果"])
        matched = config["决策日结果"]["同样本量对照"]                        # 基线只用深度模型拟合的那部分蛋
        self.assertEqual(matched["基线"], config["决策日结果"]["最好的基线"] + "·等样本")
        self.assertLess(matched["平均每折拟合蛋数"], matched["平均每折训练蛋数"])
        self.assertIn(matched["基线"], {r["方法"] for r in read_csv(run / "results_per_day.csv")})
        self.assertIn(matched["基线"], {r["对比基线"] for r in read_csv(run / "decision_day_comparison.csv")})
        per_batch = read_csv(run / "decision_day_per_batch.csv")              # 每个开发批次一行
        self.assertEqual(len(per_batch), config["开发集批次数"])
        self.assertEqual(config["决策日结果"]["按批次"]["差值"][f"深度-累计减{matched['基线']}"]["批次数"],
                         config["开发集批次数"])
        hold_batches = len(config["独立测试集"]["批次"])                      # 不到 2 个测试批次时不做按批次汇总
        by_batch = config["独立测试集"]["按批次决策效用"]
        self.assertEqual(by_batch["批次数"] if hold_batches >= 2 else by_batch, hold_batches if hold_batches >= 2 else None)
        self.assertIsNotNone(config["负对照"]["背景对照"])                     # 真实数据格式有背景光谱
        self.assertIn("认批次准确率", config["负对照"]["批次可分性"])
        self.assertTrue((self.sample / "holdout_lock.json").exists())
        self.assertFalse((self.sample / "_cache" / "holdout_lock.json").exists())

        with mock.patch.object(sys, "argv", ["predict.py", "--data", self.csv]):
            predict.main()
        rows = read_csv(self.out / "decision.csv")
        self.assertEqual(len(rows), N_EGGS)
        self.assertTrue((self.out / "predictions.csv").exists())

    def lock_evaluations(self) -> list[dict]:
        lock = json.loads((self.sample / "holdout_lock.json").read_text(encoding="utf-8"))
        return [e for record in lock["sets"].values() for e in record["evaluations"]]

    def assert_no_model_in(self, run: Path) -> None:
        for name in ("egg_model.pt", "baseline_model.pkl", train_demo.HOLDOUT_CSV, train_demo.HOLDOUT_DEBUG_CSV):
            self.assertFalse((run / name).exists(), name)

    def test_second_formal_evaluation_is_blocked(self) -> None:
        self.train("--data", self.csv)
        model = self.out / "egg_model.pt"
        saved = model.read_bytes()
        run2 = self.train("--data", self.csv)
        self.assertEqual(self.exit_code, 3)
        self.assertIn("--skip-holdout", self.config(run2)["已拦截"])
        self.assertEqual(model.read_bytes(), saved)                          # 已有的模型没被动过
        self.assert_no_model_in(run2)
        self.assertEqual(len(self.lock_evaluations()), 1)                    # 被拦下的不算查看

    def test_debug_mode_views_again_and_labels_everything(self) -> None:
        self.train("--data", self.csv)
        run2 = self.train("--data", self.csv, "--debug-holdout")
        config = self.config(run2)
        self.assertEqual(config["holdout_views"], 2)
        self.assertFalse(config["独立测试集"]["正式结果"])
        self.assertFalse((run2 / train_demo.HOLDOUT_CSV).exists())           # 不生成看起来像正式结果的文件
        rows = read_csv(run2 / train_demo.HOLDOUT_DEBUG_CSV)
        self.assertTrue(rows)
        self.assertTrue(all(r["正式结果"] == "0" and "非独立验证" in r["性质"] for r in rows))
        self.assertEqual([e["正式结果"] for e in self.lock_evaluations()], [True, False])

    def test_skip_holdout_retrains_without_viewing(self) -> None:
        self.train("--data", self.csv)
        run2 = self.train("--data", self.csv, "--skip-holdout")
        self.assertNotIn("holdout_views", self.config(run2))
        self.assertTrue((run2 / "egg_model.pt").exists())
        self.assertFalse((run2 / train_demo.HOLDOUT_CSV).exists())
        self.assertEqual(len(self.lock_evaluations()), 1)

    def test_failed_self_check_does_not_save_or_touch_model(self) -> None:
        previous = b"previous model that passed its checks"
        model = self.out / "egg_model.pt"
        model.write_bytes(previous)
        statuses = {"深度-累计": "报警", "PLS-DA": "通过"}
        with mock.patch.object(train_demo, "self_check", return_value=statuses):
            run = self.train("--data", self.csv, force_release=False)
        config = self.config(run)
        self.assertEqual(self.exit_code, 2)
        self.assertFalse(config["模型发布"]["已发布"])
        self.assertIn("流程自检未通过", config["模型发布"]["未发布原因"][0])
        self.assertEqual(model.read_bytes(), previous)                       # 上一个通过检查的模型原样保留
        self.assertFalse((self.out / "baseline_model.pkl").exists())
        self.assert_no_model_in(run)
        self.assertNotIn("独立测试集", config)
        self.assertEqual(self.lock_evaluations(), [])                        # 独立测试集没被查看
        self.assertTrue((run / "results_per_day.csv").exists())              # 诊断记录还在
        self.assertTrue((run / "log.txt").exists())

    def test_failed_background_control_does_not_save_model(self) -> None:
        controls = {"批次可分性": None,
                    "背景对照": {"各折准确率": [0.9, 0.9, 0.9], "平均准确率": 0.9, "单侧p": 0.001, "结果": "报警"}}
        with mock.patch.object(train_demo, "run_controls", return_value=controls):
            run = self.train("--data", self.csv, force_release=False)
        config = self.config(run)
        self.assertEqual(self.exit_code, 2)                                  # 自检没做也算没通过
        self.assertIn("背景负对照", config["模型发布"]["未发布原因"][0])
        self.assertFalse((self.out / "egg_model.pt").exists())
        self.assert_no_model_in(run)
        self.assertEqual(self.lock_evaluations(), [])

    def test_missing_batch_column_is_an_explicit_error(self) -> None:
        meta = self.sample / "meta.csv"
        rows = read_csv(meta)
        with meta.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[c for c in rows[0] if c != "batch"], extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "没有填 batch"):
            self.train("--data", self.csv)
        self.assertFalse((self.out / "egg_model.pt").exists())
        self.assertFalse((self.sample / "holdout_lock.json").exists())

    def test_failed_self_check_gives_nonzero_exit_and_is_recorded(self) -> None:
        statuses = {"深度-累计": "无法确认", "PLS-DA": "通过"}
        with mock.patch.object(train_demo, "self_check", return_value=statuses):
            run = self.train("--data", self.csv, "--shuffle-labels")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual((self.exit_code, config["自检"]["总体"], config["自检"]["各方法"]), (2, "未通过", statuses))

    def test_exit_code_rule(self) -> None:
        check = train_demo.check_record
        self.assertEqual([train_demo.exit_code(check(s)) for s in
                          ({"a": "通过"}, {"a": "报警"}, {"a": "通过", "b": "无法确认"}, None)], [0, 2, 2, 0])

    def test_shuffle_labels_does_not_touch_lock(self) -> None:
        run = self.train("--data", self.csv, "--shuffle-labels")
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertNotIn("holdout_views", config)
        self.assertFalse((self.sample / "holdout_lock.json").exists())
        self.assertFalse((self.out / "egg_model.pt").exists())


class TestFertilityCheck(EndToEndBase):
    """流水线检验：伊利诺伊格式的平均光谱（标签是受精状态）跑完整条流程，但绝不产出判断新蛋用的模型。"""
    DATES, PER_DATE, BANDS = 8, 6, 8

    def setUp(self) -> None:
        super().setUp()
        folder = self.out / "illinois"
        folder.mkdir()
        self.csv = folder / "spectra.csv"
        rng = np.random.default_rng(0)
        header = ["Sample ID", "Mass (g)", "Fertility status", *(f"{400 + 10 * b}.5" for b in range(self.BANDS))]
        rows = []
        for d in range(self.DATES):
            for i in range(self.PER_DATE):
                fertile = i % 2 == 0
                spectrum = 100 + 5 * d + rng.normal(0, 3, self.BANDS) + (8 * np.arange(self.BANDS) if fertile else 0)
                rows.append([f"Egg-{'F' if fertile else 'IF'}-{d + 1:02d}Jan-{len(rows) + 1:04d}'", "60.1",
                             "Fertile" if fertile else "Infertile", *(f"{v:.4f}" for v in spectrum)])
        with self.csv.open("w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerows([header, *rows])
        self.stack.enter_context(mock.patch.object(baseline_fertility, "FERTILITY_CSV", self.csv))

    def test_dataset_layout(self) -> None:
        data = baseline_fertility.as_egg_dataset()
        n = self.DATES * self.PER_DATE
        self.assertEqual(data.x.shape, (n, 1, self.BANDS, 1, 1))
        self.assertEqual(data.spec.shape, (n, 1, self.BANDS))
        self.assertEqual((int(data.sex.sum()), len(set(data.batch)), list(data.days)), (n // 2, self.DATES, [0.0]))
        self.assertTrue(data.mask.all() and not data.dev_mask.any())
        self.assertEqual(data.egg_ids[0], "Egg-F-01Jan-0001")                 # 编号末尾的引号去掉了
        self.assertAlmostEqual(float(data.wavelengths[1]), 410.5, places=3)
        self.assertEqual(real_data.eggs_without_batch(data), [])

    def test_runs_whole_pipeline_without_publishing_a_model(self) -> None:
        previous = b"sex model that must stay untouched"
        (self.out / "egg_model.pt").write_bytes(previous)
        run = self.train("--fertility-check")
        config = self.config(run)
        self.assertTrue(run.name.endswith("真实光谱检验"))
        self.assertIn("受精状态", config["标签含义"])
        self.assertIn("不是性别", config["标签含义"])
        self.assertEqual((config["决策日"], config["holdout_views"]), (0, 1))
        self.assertEqual((self.out / "egg_model.pt").read_bytes(), previous)   # 判断新蛋用的模型没被换掉
        self.assertFalse((self.out / "baseline_model.pkl").exists())
        self.assertTrue((run / "egg_model.pt").exists())                       # 只在运行文件夹里留一份
        self.assertTrue((run / train_demo.HOLDOUT_CSV).exists())
        self.assertTrue((self.csv.parent / "holdout_lock.json").exists())      # 锁文件在数据旁边
        self.assertFalse((self.out / "holdout_lock.json").exists())            # 没碰模拟数据的锁
        self.assertTrue((run / "decision_day_per_batch.csv").exists())

    def test_self_check_variant_and_conflicting_arguments(self) -> None:
        run = self.train("--fertility-check", "--shuffle-labels")
        self.assertTrue(run.name.endswith("真实光谱自检"))
        self.assertFalse((self.csv.parent / "holdout_lock.json").exists())
        self.assertFalse((run / "egg_model.pt").exists())
        with mock.patch.object(sys, "argv", ["train_demo.py", "--fertility-check", "--data", "x.csv"]), \
                mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
            train_demo.parse_args()


class TestSyntheticChain(EndToEndBase):
    def test_train_save_load_predict(self) -> None:
        tiny = lambda seed: synth_data.make_dataset(n_eggs=N_EGGS, seed=seed)  # noqa: E731
        with mock.patch.object(train_demo, "make_dataset", tiny):
            run = self.train()
        self.assertTrue((self.out / "holdout_lock.json").exists())
        trained, ckpt = predict.load_model()
        self.assertEqual(len(ckpt["wavelengths"]), ckpt["n_bands"])
        data = tiny(train_demo.SEED)
        frames = EggFrames(data.egg_ids, data.days, data.x, data.spec, data.mask, data.wavelengths)
        chosen, main_scores, _, _ = predict.score(frames, trained, ckpt)
        self.assertEqual(main_scores.shape, data.mask.shape)
        self.assertEqual(chosen, ckpt["chosen"])
        self.assertEqual(ckpt["preprocess"], preprocess_binding.preprocess_settings())
        no_axis = EggFrames(data.egg_ids, data.days, data.x, data.spec, data.mask)     # 新数据缺波长轴：必须拒绝
        with self.assertRaisesRegex(ValueError, "只有一边"):
            predict.score(no_axis, trained, ckpt)
        self.assertTrue((run / "holdout.csv").exists())

    def test_predict_refuses_changed_preprocessing(self) -> None:
        with mock.patch.object(train_demo, "make_dataset", lambda seed: synth_data.make_dataset(n_eggs=N_EGGS, seed=seed)):
            self.train()
        with mock.patch.object(cal, "MIN_REF_SNR", 99.0), self.assertRaisesRegex(ValueError, "MIN_REF_SNR"):
            predict.load_model()
        ckpt = torch.load(self.out / "egg_model.pt", weights_only=True)
        torch.save({k: v for k, v in ckpt.items() if k != "preprocess"}, self.out / "egg_model.pt")
        with self.assertRaisesRegex(ValueError, "没有记录预处理设置"):
            predict.load_model()


if __name__ == "__main__":
    unittest.main()
