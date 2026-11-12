"""消融实验：变体怎么定、数据怎么改、结论怎么下；以及整套跑一遍时不碰独立测试集、不保存模型。"""
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import ablation  # noqa: E402
import deep_train as dt  # noqa: E402
import deploy  # noqa: E402
import holdout_lock  # noqa: E402
import run_record  # noqa: E402
import synth_data  # noqa: E402
import train_demo  # noqa: E402
from baselines import BASELINE_NAMES, per_day_baselines, snv  # noqa: E402
from evaluation import ALPHA, Paired  # noqa: E402
from synth_data import EggDataset, make_dataset  # noqa: E402


def tiny(n_bands: int = 8, size: int = 2, with_dev: bool = True) -> EggDataset:
    n, t = 8, 2
    rng = np.random.default_rng(0)
    x = rng.random((n, t, n_bands, size, size)).astype(np.float32)
    return EggDataset(egg_ids=tuple(f"e{i}" for i in range(n)), x=x, spec=x.mean(axis=(3, 4)),
                      mask=np.ones((n, t), dtype=bool), sex=np.arange(n) % 2, dev=np.zeros((n, t), dtype=np.float32),
                      dev_mask=np.full((n, t), with_dev), batch=np.arange(n) // 2, days=np.array([0.0, 4.0], dtype=np.float32),
                      wavelengths=np.linspace(400.0, 1000.0, n_bands))


class TestDataChanges(unittest.TestCase):
    def test_band_groups_cover_every_band_once(self) -> None:
        groups = ablation.band_groups(32)
        self.assertEqual(len(groups), ablation.BAND_GROUPS)
        self.assertEqual(sorted(np.concatenate(groups).tolist()), list(range(32)))

    def test_too_few_bands_gives_no_groups(self) -> None:
        self.assertEqual(ablation.band_groups(ablation.BAND_GROUPS * ablation.MIN_BANDS_PER_GROUP - 1), [])

    def test_drop_bands_removes_them_everywhere(self) -> None:
        data = tiny()
        out = ablation.drop_bands(np.array([2, 3]))(data)
        self.assertEqual(out.x.shape[2], 6)
        self.assertEqual(out.spec.shape[2], 6)
        np.testing.assert_array_equal(out.x, data.x[:, :, [0, 1, 4, 5, 6, 7]])
        np.testing.assert_array_equal(out.wavelengths, data.wavelengths[[0, 1, 4, 5, 6, 7]])
        self.assertEqual(data.x.shape[2], 8)                    # 原数据不动

    def test_no_spatial_keeps_only_the_mean_spectrum(self) -> None:
        data = tiny()
        out = ablation.no_spatial(data)
        self.assertEqual(out.x.shape, (8, 2, 8, 1, 1))
        np.testing.assert_allclose(out.x[:, :, :, 0, 0], data.spec, rtol=1e-6)

    def test_baseline_without_snv_uses_raw_spectra(self) -> None:
        data = make_dataset(n_eggs=48, seed=1)
        train, test = np.arange(32), np.arange(32, 48)
        with_snv = per_day_baselines(data.spec, data.mask, data.sex, train, test)
        raw = per_day_baselines(data.spec, data.mask, data.sex, train, test, use_snv=False)
        pre_snv = per_day_baselines(snv(data.spec), data.mask, data.sex, train, test, use_snv=False)
        for name in BASELINE_NAMES:
            self.assertFalse(np.allclose(with_snv[name], raw[name], equal_nan=True))
            np.testing.assert_allclose(pre_snv[name], with_snv[name], equal_nan=True)     # 开关只管 SNV 这一步


class TestVariants(unittest.TestCase):
    def test_full_set(self) -> None:
        variants = ablation.build_variants(tiny(), dev_weight=5.0)
        names = [v.name for v in variants]
        self.assertEqual(names[0], ablation.FULL_DEEP)          # 完整模型必须最先训练，单帧变体要复用它
        self.assertIn(ablation.FULL_BASE, names)
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum("去掉空间" in n for n in names), 1)
        self.assertEqual(sum("去掉发育" in n for n in names), 1)
        self.assertEqual(sum("nm" in n for n in names), 2 * ablation.BAND_GROUPS)
        self.assertEqual(ablation.n_comparisons(variants), len(variants) - 2)

    def test_unsupported_items_are_left_out(self) -> None:
        names = [v.name for v in ablation.build_variants(tiny(n_bands=4, size=1), dev_weight=0.0)]
        self.assertFalse(any("去掉空间" in n or "去掉发育" in n or "nm" in n for n in names))
        self.assertIn("深度：去掉时序（单帧输出）", names)

    def test_each_variant_is_compared_with_its_own_family(self) -> None:
        for v in ablation.build_variants(tiny(), dev_weight=5.0):
            self.assertEqual(ablation.reference_of(v), ablation.FULL_DEEP if v.family == "深度" else ablation.FULL_BASE)


class TestVerdict(unittest.TestCase):
    def test_needs_to_pass_the_corrected_threshold(self) -> None:
        self.assertEqual(ablation.verdict(Paired(0.1, 0.0, 0.2, 0.004), 0.005), "有明确贡献")
        self.assertEqual(ablation.verdict(Paired(0.1, -0.1, 0.3, 0.02), 0.005), "看不出贡献")    # 不校正的话 0.02 会算显著

    def test_unknown_p_is_never_a_contribution(self) -> None:
        nan = float("nan")
        self.assertTrue(ablation.verdict(Paired(0.3, nan, nan, nan), 0.05).startswith("无法判断"))


class TestSummarize(unittest.TestCase):
    def setUp(self) -> None:
        n = 40
        self.data = EggDataset(egg_ids=tuple(f"e{i}" for i in range(n)), x=np.zeros((n, 1, 1, 1, 1), dtype=np.float32),
                               spec=np.zeros((n, 1, 1), dtype=np.float32), mask=np.ones((n, 1), dtype=bool),
                               sex=np.arange(n) % 2, dev=np.zeros((n, 1), dtype=np.float32),
                               dev_mask=np.zeros((n, 1), dtype=bool), batch=np.arange(n) // 4,
                               days=np.array([4.0], dtype=np.float32))
        right = self.data.sex.astype(float)[:, None]
        # 变体每折错的蛋数不一样（1、2、3、2 颗），各折差值才有方差、p 值才算得出来
        wrong = right.copy()
        fold = np.arange(n) // 10
        for k, n_wrong in enumerate((1, 2, 3, 2)):
            idx = np.flatnonzero(fold == k)[:n_wrong]
            wrong[idx] = 1 - wrong[idx]
        self.variants = (ablation.Variant(ablation.FULL_DEEP, "深度", ""), ablation.Variant("深度：变体", "深度", ""),
                         ablation.Variant(ablation.FULL_BASE, "基线", ""), ablation.Variant("基线：变体", "基线", ""))
        self.res = ablation.AblationResult(self.variants, {ablation.FULL_DEEP: right, "深度：变体": wrong,
                                                          ablation.FULL_BASE: wrong, "基线：变体": right},
                                           fold, 4, 30.0, 10.0)

    def test_drop_is_full_minus_variant(self) -> None:
        rows = {r["变体"]: r for r in ablation.summarize(self.data, self.res, 0)}
        self.assertEqual(rows[ablation.FULL_DEEP]["决策效用"], 1.0)
        self.assertEqual(rows["深度：变体"]["决策效用"], 0.8)
        self.assertAlmostEqual(rows["深度：变体"]["去掉后下降"], 0.2)
        self.assertAlmostEqual(rows["基线：变体"]["去掉后下降"], -0.2)        # 去掉后反而更好：负数
        self.assertEqual(rows["基线：变体"]["结论"], "看不出贡献")

    def test_reference_rows_have_no_comparison(self) -> None:
        rows = {r["变体"]: r for r in ablation.summarize(self.data, self.res, 0)}
        for name in (ablation.FULL_DEEP, ablation.FULL_BASE):
            self.assertEqual((rows[name]["对照"], rows[name]["结论"], rows[name]["去掉后下降"]), ("", "", None))

    def test_threshold_is_alpha_over_number_of_comparisons(self) -> None:
        rows = ablation.summarize(self.data, self.res, 0)
        self.assertEqual(rows[1]["校正后门槛"], round(ALPHA / 2, 4))

    def test_per_day_rows_have_one_column_per_day(self) -> None:
        rows = ablation.per_day_rows(self.data, self.res)
        self.assertEqual(list(rows[0]), ["变体", "第4天"])
        self.assertEqual(len(rows), 4)


class TestEndToEnd(unittest.TestCase):
    """整套跑一遍：极小的数据、极少的轮数。"""

    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp())
        self.handlers = list(logging.getLogger().handlers)
        self.stack = contextlib.ExitStack()
        self.lock = self.out / "holdout_lock.json"
        small = lambda seed=0: make_dataset(n_eggs=32, seed=seed)  # noqa: E731
        for p in (mock.patch.object(run_record, "RUNS_DIR", self.out / "runs"),
                  mock.patch.object(holdout_lock, "SYNTH_HOLDOUT_LOCK", self.lock),
                  mock.patch.object(deploy, "MODEL_PATH", self.out / "egg_model.pt"),
                  mock.patch.object(deploy, "BASELINE_MODEL_PATH", self.out / "baseline_model.pkl"),
                  mock.patch.object(synth_data, "EGGS_PER_BATCH", 4),
                  mock.patch.object(train_demo, "make_dataset", small),
                  mock.patch.object(dt, "MAX_EPOCHS", 2), mock.patch.object(dt, "PATIENCE", 2),
                  mock.patch.object(ablation, "N_FOLDS", 3),
                  mock.patch.object(sys, "argv", ["ablation.py"])):
            self.stack.enter_context(p)

    def tearDown(self) -> None:
        self.stack.close()
        root = logging.getLogger()
        for h in root.handlers[:]:
            if h not in self.handlers:
                h.close()
                root.removeHandler(h)
        shutil.rmtree(self.out, ignore_errors=True)

    def run_dir(self) -> Path:
        (run,) = (self.out / "runs").iterdir()
        return run

    def test_writes_tables_and_nothing_else(self) -> None:
        self.assertEqual(ablation.main(), 0)
        run = self.run_dir()
        self.assertTrue(run.name.endswith("消融实验"))
        with (run / "ablation.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(len(rows), len(config["变体"]))
        self.assertEqual(config["比较项数"], len(rows) - 2)
        self.assertIn("模拟数据", config["说明"])
        self.assertTrue(all(r["决策效用"] != "" for r in rows))
        self.assertTrue((run / "ablation_per_day.csv").exists())
        # 不保存任何模型
        self.assertFalse((self.out / "egg_model.pt").exists())
        self.assertFalse(list(run.glob("*.pt")) + list(run.glob("*.pkl")))

    def test_holdout_is_excluded_and_never_viewed(self) -> None:
        seen = []
        real = ablation.run_ablation
        with mock.patch.object(ablation, "run_ablation", side_effect=lambda d, *rest: seen.append(d) or real(d, *rest)):
            ablation.main()
        record = holdout_lock.read_split(self.lock)
        self.assertTrue(record.holdout_batches)
        self.assertFalse(set(seen[0].batch.astype(str)) & set(record.holdout_batches))       # 测试批次没进消融
        before = self.lock.read_bytes()
        ablation.main()                                          # 再跑一次：锁文件一个字节都不变
        self.assertEqual(self.lock.read_bytes(), before)

    def test_all_variants_share_the_same_folds(self) -> None:
        captured = {}
        real = ablation.summarize
        with mock.patch.object(ablation, "summarize", side_effect=lambda d, r, i: captured.update(res=r) or real(d, r, i)):
            ablation.main()
        res = captured["res"]
        self.assertTrue((res.fold_of_egg >= 0).all())
        day = 2                                                  # 第 4 天
        for name, scores in res.scores.items():
            if "单帧" in name:
                continue                                         # 单帧输出当天没拍就判断不了，允许是 nan
            self.assertEqual(scores.shape, res.scores[ablation.FULL_DEEP].shape, name)
        self.assertTrue(np.isfinite(res.scores[ablation.FULL_DEEP][:, day]).any())


if __name__ == "__main__":
    unittest.main()
