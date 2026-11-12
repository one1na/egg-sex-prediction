"""流程的自动化测试：数据入口、校正、缓存、决策日规则、防泄漏、统计检验。

运行：双击 5-运行测试.bat，或在 code 文件夹里 py -3.11 -m unittest discover -s tests -v
只用 Python 自带的 unittest，不需要另装包。每次改了代码都应该全部通过。
"""
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

import calibration as cal  # noqa: E402
import roi  # noqa: E402
import deep_train as dt  # noqa: E402
import predict  # noqa: E402
import preprocess_binding  # noqa: E402
import real_data  # noqa: E402
from evaluation import compare_paired, day_metrics, holdout_split, inner_split, outer_folds  # noqa: E402
from model import EggSeqModel  # noqa: E402
from real_data import EggFrames  # noqa: E402

class TestCalibration(unittest.TestCase):
    def test_normal(self) -> None:
        refl = cal.calibrate(np.full((2, 3, 2), 60.0), np.full((5, 3, 2), 110.0), np.full((5, 3, 2), 10.0))
        np.testing.assert_allclose(refl, 0.5)

    def test_white_equals_dark(self) -> None:
        white = np.full((5, 3, 2), 110.0)
        white[:, 1, 0] = 10.0
        with self.assertRaisesRegex(ValueError, "几乎一样"):
            cal.calibrate(np.full((2, 3, 2), 60.0), white, np.full((5, 3, 2), 10.0))

    def test_white_dark_swapped(self) -> None:
        with self.assertRaisesRegex(ValueError, "放反"):
            cal.calibrate(np.full((2, 3, 2), 60.0), np.full((5, 3, 2), 10.0), np.full((5, 3, 2), 110.0))

    def test_non_finite_reference(self) -> None:
        white = np.full((5, 3, 2), 110.0)
        white[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            cal.calibrate(np.full((2, 3, 2), 60.0), white, np.full((5, 3, 2), 10.0))

    def test_reflectance_out_of_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "合理范围"):
            cal.calibrate(np.full((2, 3, 2), 1000.0), np.full((5, 3, 2), 110.0), np.full((5, 3, 2), 10.0))

    def test_tiny_reference_signal_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "噪声"):
            cal.calibrate(np.full((2, 3, 2), 1e-6), np.full((5, 3, 2), 1e-6) + 10, np.full((5, 3, 2), 10.0))

    def test_reference_signal_below_dark_noise_rejected(self) -> None:
        dark = 100 + np.random.default_rng(0).normal(0, 20, (50, 3, 2))
        with self.assertRaisesRegex(ValueError, "噪声"):
            cal.calibrate(np.full((2, 3, 2), 150.0), np.full((50, 3, 2), 250.0), dark)

    def test_reflectance_scale_references_pass(self) -> None:
        refl = cal.calibrate(np.full((2, 3, 2), 0.5), np.ones((5, 3, 2)), np.zeros((5, 3, 2)))
        np.testing.assert_allclose(refl, 0.5)

    def test_cropped_image_with_full_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "先校正再裁剪"):
            cal.calibrate(np.ones((20, 15, 4)), np.ones((5, 28, 4)) * 2, np.zeros((5, 28, 4)))


class TestDevLabels(unittest.TestCase):
    def row(self, value: str) -> dict[str, str]:
        return {"egg_id": "e1", "day": "4", "dev": value}

    def test_rejects_nan_inf_out_of_range(self) -> None:
        for value in ("nan", "inf", "-inf", "2.0", "-0.1", "abc"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                real_data._read_dev(self.row(value))

    def test_accepts_valid(self) -> None:
        self.assertEqual(real_data._read_dev(self.row("0.5")), 0.5)

    def test_loss_finite_with_nan_in_masked_position(self) -> None:
        torch.manual_seed(0)
        model = EggSeqModel(4)
        x, mask = torch.randn(2, 3, 4, 16, 16), torch.ones(2, 3, dtype=torch.bool)
        dev = torch.tensor([[0.1, float("nan"), 0.3]] * 2)
        loss = dt._loss(model(x, mask, torch.tensor([0.0, 2.0, 4.0])), mask, torch.tensor([0, 1]),
                        dev, ~torch.isnan(dev), 5.0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))


class TestCache(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        np.save(self.dir / "c.npy", np.zeros((2, 2, 2)))
        self.csv = self.dir / "meta.csv"
        self.rows = [{"egg_id": "e1", "day": "0", "sex": "0", "cube": "c.npy"}]

    def tearDown(self) -> None:
        shutil.rmtree(self.dir)

    def test_label_change_keeps_cache(self) -> None:
        changed = [{**self.rows[0], "sex": "1"}]
        self.assertEqual(real_data._fingerprint(self.csv, self.rows), real_data._fingerprint(self.csv, changed))

    def test_preprocessing_code_change_invalidates_cache(self) -> None:
        before = real_data._fingerprint(self.csv, self.rows)
        edited = self.dir / "roi.py"
        edited.write_text(Path(roi.__file__).read_text(encoding="utf-8").replace(
            "MIN_CONTRAST = 5.0", "MIN_CONTRAST = 50.0"), encoding="utf-8")
        with mock.patch.object(real_data.roi, "__file__", str(edited)):
            self.assertNotEqual(before, real_data._fingerprint(self.csv, self.rows))


class TestDecisionDayRule(unittest.TestCase):
    def test_cumulative_uses_history_single_frame_does_not(self) -> None:
        torch.manual_seed(0)
        trained = dt.TrainedModel(EggSeqModel(4), dt.Normalizer(np.zeros((1, 4, 1, 1)), np.ones((1, 4, 1, 1))), 1)
        mask = np.array([[True, True, False], [False, False, False]])
        out = dt.predict_scores(trained, np.random.rand(2, 3, 4, 16, 16).astype(np.float32), mask,
                                np.array([0, 2, 4], dtype=np.float32))
        self.assertTrue(np.isnan(out["frame"][0, 2]))         # 第 4 天没拍：单帧判断不了
        self.assertFalse(np.isnan(out["cum"][0, 2]))          # 累计用前两天的图像
        self.assertTrue(np.isnan(out["cum"][1]).all())        # 从没拍过：都判断不了

    def test_coverage_counts_all_eggs(self) -> None:
        m = day_metrics(np.array([0.9, np.nan]), np.array([1, 0]))
        self.assertEqual((m["coverage"], m["acc"], m["n"]), (0.5, 1.0, 1.0))


class TestCumulativeValidity(unittest.TestCase):
    """累计判断在"当天或之前拍过"的位置都有定义：训练的损失和预测的 nan 用同一条规则。"""

    MASK = torch.tensor([[True, False, True], [False, True, True]])

    def test_cum_valid(self) -> None:
        self.assertEqual(dt.cum_valid(self.MASK).tolist(), [[True, True, True], [False, True, True]])

    def test_loss_supervises_cum_on_missing_day_but_not_frame(self) -> None:
        out = {"frame_sex": torch.zeros(2, 3, 2, requires_grad=True), "cum_sex": torch.zeros(2, 3, 2, requires_grad=True),
               "dev": torch.zeros(2, 3)}
        dt._loss(out, self.MASK, torch.tensor([0, 1]), torch.zeros(2, 3), torch.zeros(2, 3, dtype=torch.bool),
                 0.0).backward()
        has_grad = lambda g: (g.abs().sum(-1) > 0)  # noqa: E731
        self.assertEqual(has_grad(out["frame_sex"].grad).tolist(), self.MASK.tolist())
        self.assertEqual(has_grad(out["cum_sex"].grad).tolist(), dt.cum_valid(self.MASK).tolist())

    def test_predict_scores_nan_exactly_where_cum_invalid(self) -> None:
        torch.manual_seed(0)
        trained = dt.TrainedModel(EggSeqModel(4), dt.Normalizer(np.zeros((1, 4, 1, 1)), np.ones((1, 4, 1, 1))), 1)
        out = dt.predict_scores(trained, np.random.rand(2, 3, 4, 16, 16).astype(np.float32), self.MASK.numpy(),
                                np.array([0, 2, 4], dtype=np.float32))
        self.assertEqual((~np.isnan(out["cum"])).tolist(), dt.cum_valid(self.MASK).tolist())
        self.assertEqual((~np.isnan(out["frame"])).tolist(), self.MASK.tolist())

    def test_missing_day_output_uses_only_past_frames(self) -> None:
        torch.manual_seed(0)
        model = EggSeqModel(4).eval()
        x, days = torch.randn(1, 3, 4, 16, 16), torch.tensor([0.0, 2.0, 4.0])
        mask = torch.tensor([[True, False, True]])
        future = x.clone()
        future[:, 2] += 5.0                                  # 只改第 3 天的图像
        with torch.no_grad():
            a, b = model(x, mask, days)["cum_sex"], model(future, mask, days)["cum_sex"]
        self.assertEqual(float((a[:, :2] - b[:, :2]).abs().max()), 0.0)


class TestPreprocessBinding(unittest.TestCase):
    def test_unchanged_passes(self) -> None:
        preprocess_binding.check_preprocess(preprocess_binding.preprocess_settings(), "模型")

    def test_changed_constant_is_named(self) -> None:
        trained = preprocess_binding.preprocess_settings()
        for module, name, value in ((cal, "MIN_REF_SNR", 99.0), (cal, "REFL_RANGE", (0.0, 1.0)),
                                    (roi, "ROI_MODE", "none"), (roi, "MIN_CROP_IN_MASK", 0.5),
                                    (cal, "TARGET_SIZE", 32)):
            with self.subTest(name=name), mock.patch.object(module, name, value):
                with self.assertRaisesRegex(ValueError, name) as ctx:
                    preprocess_binding.check_preprocess(trained, "模型")
                self.assertIn("重新训练", str(ctx.exception))

    def test_code_change_alone_is_rejected(self) -> None:
        trained = preprocess_binding.preprocess_settings()
        with mock.patch.object(real_data, "_code_version", return_value=b"edited"):
            with self.assertRaisesRegex(ValueError, "预处理代码改过了"):
                preprocess_binding.check_preprocess(trained, "模型")

    def test_old_model_without_record_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "重新训练"):
            preprocess_binding.check_preprocess(None, "模型")

    def test_settings_survive_json_roundtrip(self) -> None:
        import json
        settings = preprocess_binding.preprocess_settings()
        self.assertEqual(json.loads(json.dumps(settings)), settings)


class TestNoLeakage(unittest.TestCase):
    def test_future_frame_and_other_egg_do_not_change_outputs_in_training_mode(self) -> None:
        torch.manual_seed(0)
        model = EggSeqModel(8).train()
        x, mask, days = torch.randn(3, 4, 8, 16, 16), torch.ones(3, 4, dtype=torch.bool), torch.arange(4.0)
        base = model(x, mask, days)
        future, other = x.clone(), x.clone()
        future[:, 3] += 5.0
        other[1:] += 5.0
        for changed, idx in ((model(future, mask, days), (slice(None), slice(0, 3))),
                             (model(other, mask, days), (0, slice(None)))):
            for key in ("frame_sex", "cum_sex", "dev"):
                self.assertEqual(float((base[key][idx] - changed[key][idx]).abs().max().detach()), 0.0)

    def test_splits_never_share_batches(self) -> None:
        groups = np.repeat(np.arange(10), 6)
        sex = np.tile([0, 1], 30)
        dev_idx, hold_idx = holdout_split(sex, groups, 0)
        self.assertFalse(set(groups[dev_idx]) & set(groups[hold_idx]))
        for train, test in outer_folds(sex[dev_idx], groups[dev_idx], 5, 0):
            self.assertFalse(set(groups[dev_idx][train]) & set(groups[dev_idx][test]))
            fit, val, grouped = inner_split(train, sex[dev_idx], groups[dev_idx], 0)
            self.assertTrue(grouped)
            self.assertFalse(set(groups[dev_idx][fit]) & set(groups[dev_idx][val]))

    def test_no_holdout_with_few_batches(self) -> None:
        self.assertIsNone(holdout_split(np.tile([0, 1], 15), np.repeat(np.arange(3), 10), 0))


class TestStatistics(unittest.TestCase):
    def test_zero_variance_gives_nan(self) -> None:
        cmp = compare_paired(np.full(5, 0.51), np.full(5, 0.50), 128, 32)
        self.assertTrue(np.isnan(cmp.p) and np.isnan(cmp.ci_low))

    def test_ci_contains_mean(self) -> None:
        cmp = compare_paired(np.array([0.7, 0.75, 0.72, 0.8, 0.74]), np.full(5, 0.65), 128, 32)
        self.assertLess(cmp.ci_low, cmp.diff)
        self.assertLess(cmp.diff, cmp.ci_high)
        self.assertLess(cmp.p, 0.05)


class TestWavelengths(unittest.TestCase):
    def frames(self, wl: np.ndarray | None) -> EggFrames:
        return EggFrames(("e",), np.zeros(1), np.zeros((1, 1, 3, 2, 2)), np.zeros((1, 1, 3)),
                         np.ones((1, 1), bool), wl)

    def test_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "波长轴"):
            predict.check_wavelengths(self.frames(np.array([400.0, 500, 600])), {"wavelengths": [400, 510, 600]})

    def test_match_accepted(self) -> None:
        predict.check_wavelengths(self.frames(np.array([400.0, 500, 600])), {"wavelengths": [400, 500, 600.2]})

    def test_one_side_only_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "只有一边"):
            predict.check_wavelengths(self.frames(np.array([400.0, 500, 600])), {"wavelengths": None})
        with self.assertRaisesRegex(ValueError, "只有一边"):
            predict.check_wavelengths(self.frames(None), {"wavelengths": [400, 500, 600]})

    def test_neither_side_allowed(self) -> None:
        predict.check_wavelengths(self.frames(None), {"wavelengths": None})
        predict.check_wavelengths(self.frames(None), {})


if __name__ == "__main__":
    unittest.main()
