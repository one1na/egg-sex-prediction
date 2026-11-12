"""两个负对照：批次可分性、背景对照。背景里混进性别信息要报警；批次光谱形状不同要认得出来。"""
import logging
import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import controls  # noqa: E402
from evaluation import CHANCE_ALARM, CHANCE_PASS  # noqa: E402
from synth_data import EggDataset  # noqa: E402

N_BATCHES, PER_BATCH, BANDS, N_FOLDS = 6, 20, 12, 5


def stub(seed: int = 0, batch_shape: float = 0.0, bg_sex: float = 0.0) -> tuple[EggDataset, np.ndarray]:
    """6 批 × 20 颗、一天。batch_shape：各批光谱形状不同的程度；bg_sex：背景亮度跟性别挂钩的程度。"""
    rng = np.random.default_rng(seed)
    n = N_BATCHES * PER_BATCH
    batch = np.repeat([f"B{i}" for i in range(N_BATCHES)], PER_BATCH)
    sex = rng.permutation(np.tile([0, 1], n // 2))
    bands = np.linspace(0, 1, BANDS)
    shape = np.array([np.sin((i + 1) * 3 * bands) for i in range(N_BATCHES)])[np.repeat(np.arange(N_BATCHES), PER_BATCH)]
    spec = 1 + batch_shape * shape + rng.normal(0, 0.05, (n, BANDS))
    bg = 0.05 + bg_sex * sex[:, None] + rng.normal(0, 0.01, (n, BANDS))
    data = EggDataset(egg_ids=tuple(f"e{i}" for i in range(n)), x=np.zeros((n, 1, 1, 1, 1), np.float32),
                      spec=spec[:, None].astype(np.float32), mask=np.ones((n, 1), bool), sex=sex,
                      dev=np.zeros((n, 1), np.float32), dev_mask=np.zeros((n, 1), bool), batch=batch,
                      days=np.array([4.0], np.float32), bg_spec=bg[:, None].astype(np.float32))
    fold_of_egg = np.repeat(np.arange(N_FOLDS), n // N_FOLDS)[rng.permutation(n)]
    return data, fold_of_egg


class TestBatchProbe(unittest.TestCase):
    def test_distinct_batches_are_flagged(self) -> None:
        probe = controls.batch_probe(stub(batch_shape=0.5)[0], 0)
        self.assertTrue(probe["批次差异很大"])
        self.assertGreater(probe["认批次准确率"], 0.9)

    def test_identical_batches_are_not_flagged(self) -> None:
        probe = controls.batch_probe(stub()[0], 0)
        self.assertFalse(probe["批次差异很大"])
        self.assertAlmostEqual(probe["瞎猜水平"], 1 / N_BATCHES, places=3)

    def test_single_batch_skipped(self) -> None:
        data = stub()[0]
        self.assertIsNone(controls.batch_probe(replace(data, batch=np.full(len(data.sex), "B0")), 0))


class TestBackgroundControl(unittest.TestCase):
    def run_control(self, **kw: float) -> dict:
        data, folds = stub(**kw)
        n = len(data.sex)
        return controls.background_control(data, folds, 0, n * 0.8, n * 0.2)

    def test_clean_background_passes(self) -> None:
        self.assertEqual(self.run_control()["结果"], CHANCE_PASS)

    def test_background_tied_to_sex_alarms(self) -> None:
        result = self.run_control(bg_sex=0.02)
        self.assertEqual(result["结果"], CHANCE_ALARM)
        self.assertGreater(result["平均准确率"], 0.8)

    def test_no_background_skipped(self) -> None:
        data, folds = stub()
        self.assertIsNone(controls.background_control(replace(data, bg_spec=None), folds, 0, 96, 24))
        nan_bg = np.full_like(data.bg_spec, np.nan)
        self.assertIsNone(controls.background_control(replace(data, bg_spec=nan_bg), folds, 0, 96, 24))


if __name__ == "__main__":
    unittest.main()
