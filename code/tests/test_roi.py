"""找蛋和裁剪的测试：任意朝向、局部对比度、多个候选、固定蛋位、失败记录。"""
import csv
import logging
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)

import calibration as cal  # noqa: E402
import real_data  # noqa: E402
import roi  # noqa: E402
from helpers import RNG, ellipse, image  # noqa: E402

SHAPE = (70, 90)
EGG_RADII = (10, 19)          # 长短轴之比 1.9


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / (a | b).sum())


class TestEggMask(unittest.TestCase):
    shape = (40, 70)

    def test_ignores_larger_bright_rectangle(self) -> None:
        egg = ellipse(self.shape, (20, 15), (8, 10))
        rect = np.zeros(self.shape, bool)
        rect[8:28, 35:60] = True                     # 面积比蛋大的矩形亮块
        found, qc = roi.egg_mask(image([(egg, 0.5), (rect, 0.6)], self.shape))
        self.assertGreater(iou(found, egg), 0.9)
        self.assertLess(qc.fill, 0.9)

    def test_only_rectangle_is_rejected(self) -> None:
        rect = np.zeros(self.shape, bool)
        rect[8:28, 20:45] = True
        with self.assertRaisesRegex(ValueError, "不像椭圆"):
            roi.egg_mask(image([(rect, 0.6)], self.shape))

    def test_egg_touching_border_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "边缘"):
            roi.egg_mask(image([(ellipse(self.shape, (20, 5), (8, 10)), 0.5)], self.shape))

    def test_egg_only_image_suggests_roi_none(self) -> None:
        with self.assertRaisesRegex(ValueError, "ROI_MODE"):
            roi.egg_mask(0.5 + RNG.normal(0, 0.01, (16, 16, 4)))


class TestRotation(unittest.TestCase):
    def test_elongated_egg_passes_at_any_angle(self) -> None:
        for angle in (0, 15, 30, 45, 60, 75, 90):
            with self.subTest(angle=angle):
                egg = ellipse(SHAPE, (35, 45), EGG_RADII, angle)
                found, qc = roi.egg_mask(image([(egg, 0.5)], SHAPE))
                self.assertGreater(iou(found, egg), 0.9)
                self.assertGreater(qc.fill, 0.7)
                self.assertLess(qc.aspect, roi.MAX_EGG_ASPECT)

    def test_crop_is_clean_when_rotated(self) -> None:
        for angle in (30, 45):
            with self.subTest(angle=angle):
                egg = ellipse(SHAPE, (35, 45), EGG_RADII, angle)
                _, qc = roi.egg_mask(image([(egg, 0.5)], SHAPE))
                self.assertGreaterEqual(qc.crop_in_mask, roi.MIN_CROP_IN_MASK)
                truth = roi.crop_inside(egg[..., None].astype(float), qc)      # 用真实椭圆独立核对
                self.assertGreaterEqual(truth.mean(), roi.MIN_CROP_IN_MASK)

    def test_egg_retention_recorded_and_lower_when_tilted(self) -> None:
        kept = {}
        for angle in (0, 45):
            egg = ellipse(SHAPE, (35, 45), EGG_RADII, angle)
            _, qc = roi.egg_mask(image([(egg, 0.5)], SHAPE))
            r0, r1, c0, c1 = qc.crop
            self.assertAlmostEqual(qc.egg_retention, egg[r0:r1, c0:c1].sum() / egg.sum(), delta=0.03)
            kept[angle] = qc.egg_retention
        self.assertGreater(kept[0], 0.5)
        self.assertLess(kept[45], kept[0] - 0.1)                 # 斜放的长蛋丢得多，正是要记录它的原因

    def test_crop_shrinks_until_clean(self) -> None:
        egg = ellipse(SHAPE, (35, 45), EGG_RADII)
        start = roi._inscribed_crop(egg, roi._shape(egg))[0]
        bitten = egg.copy()
        bitten[start[0]:start[0] + 4, start[2]:start[2] + 6] = False      # 咬掉起始框的一个角
        box, inside = roi._inscribed_crop(bitten, roi._shape(bitten))
        self.assertGreaterEqual(inside, roi.MIN_CROP_IN_MASK)
        self.assertLess((box[1] - box[0]) * (box[3] - box[2]), (start[1] - start[0]) * (start[3] - start[2]))

    def test_unfillable_region_gives_no_crop(self) -> None:
        sliver = np.zeros(SHAPE, bool)
        sliver[30:40, 10:80:2] = True               # 一排断断续续的点：怎么缩也找不到干净的内部
        self.assertIsNone(roi._inscribed_crop(sliver, roi._shape(sliver)))


class TestAmbiguityAndFixed(unittest.TestCase):
    def setUp(self) -> None:
        self.egg = ellipse(SHAPE, (35, 22), (10, 16))
        self.circle = ellipse(SHAPE, (35, 68), (14, 14))      # 更大的圆形亮物，形状上和蛋分不开

    def test_two_egg_like_regions_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "多个像蛋的区域.*fixed"):
            roi.egg_mask(image([(self.egg, 0.5), (self.circle, 0.6)], SHAPE))

    def test_fixed_window_picks_egg_inside(self) -> None:
        img = image([(self.egg, 0.5), (self.circle, 0.6)], SHAPE)
        with mock.patch.object(roi, "ROI_MODE", "fixed"), mock.patch.object(roi, "FIXED_ROI", (5, 65, 0, 45)):
            _, _, mask, qc = cal.process_image(img)
        self.assertGreater(iou(mask, self.egg), 0.9)
        self.assertLessEqual(qc.bbox[3], 45)

    def test_fixed_without_window_is_config_error(self) -> None:
        with mock.patch.object(roi, "FIXED_ROI", None), self.assertRaisesRegex(ValueError, "FIXED_ROI"):
            cal.process_image(image([(self.egg, 0.5)], SHAPE), "fixed")

    def test_egg_cut_by_window_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "边缘"):
            roi.egg_mask(image([(self.egg, 0.5)], SHAPE), (5, 65, 0, 30))


class TestBackgroundGradient(unittest.TestCase):
    def test_moderate_gradients_detect_egg(self) -> None:
        egg = ellipse(SHAPE, (35, 45), EGG_RADII, 30)
        for gradient in (0.0, 0.15, 0.35):
            with self.subTest(gradient=gradient):
                found, _ = roi.egg_mask(image([(egg, 0.5)], SHAPE, gradient=gradient))
                self.assertGreater(iou(found, egg), 0.9)

    def test_extreme_gradient_does_not_crash(self) -> None:
        """梯度 0.60：背景右端比蛋还亮，Otsu 把背景一起当前景、碰边，被拒绝并报错（不会静默给出错的区域）。"""
        egg = ellipse(SHAPE, (35, 45), EGG_RADII, 30)
        try:
            found, _ = roi.egg_mask(image([(egg, 0.5)], SHAPE, gradient=0.6))
        except ValueError:
            return
        self.assertGreater(iou(found, egg), 0.9)


class TestFlatBackground(unittest.TestCase):
    def test_tiny_brightness_difference_rejected(self) -> None:
        egg = ellipse(SHAPE, (35, 45), EGG_RADII)
        img = np.full((*SHAPE, 4), 0.3)
        img[egg] += 1e-7                              # 背景完全平坦、蛋只亮 1e-7
        with self.assertRaisesRegex(ValueError, "亮度只差"):
            roi.egg_mask(img)


class TestReferenceNoiseThroughLoader(unittest.TestCase):
    """真实数据入口：暗电流的扫描噪声必须在校正时被看到（不能先平均掉）。"""
    BANDS, COLS, LINES = 4, 20, 50

    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        np.save(self.dir / "cube.npy", np.full((8, self.COLS, self.BANDS), 300.0, dtype=np.float32))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir)

    def reflectance(self, white_level: float, dark_noise: float) -> np.ndarray:
        shape = (self.LINES, self.COLS, self.BANDS)
        dark = 100 + dark_noise * np.where(np.arange(self.LINES) % 2 == 0, 1.0, -1.0)[:, None, None] * np.ones(shape)
        np.save(self.dir / "white.npy", np.full(shape, white_level, dtype=np.float32))
        np.save(self.dir / "dark.npy", dark.astype(np.float32))
        row = {"egg_id": "e1", "day": "0", "cube": "cube.npy", "white": "white.npy", "dark": "dark.npy"}
        return real_data._reflectance(self.dir, row, {})

    def test_noisy_dark_with_weak_white_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "噪声.*涉及波段序号"):
            self.reflectance(white_level=105.0, dark_noise=20.0)

    def test_normal_reference_accepted(self) -> None:
        refl = self.reflectance(white_level=1000.0, dark_noise=2.0)
        self.assertEqual(refl.shape, (8, self.COLS, self.BANDS))
        self.assertTrue(np.isfinite(refl).all())


class TestFailureRecords(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())
        good = image([(ellipse(SHAPE, (35, 45), EGG_RADII), 0.5)], SHAPE).astype(np.float32)
        rect = np.zeros(SHAPE, bool)
        rect[15:55, 20:70] = True
        bad = image([(rect, 0.6)], SHAPE).astype(np.float32)
        for name, img in (("good", good), ("bad1", bad), ("bad2", bad)):
            np.save(self.dir / f"{name}.npy", img)
        rows = [("e1", "good"), ("e2", "bad1"), ("e3", "bad2")]
        with (self.dir / "meta.csv").open("w", encoding="utf-8", newline="") as f:
            f.write("egg_id,day,sex,cube\n" + "".join(f"{e},0,0,{c}.npy\n" for e, c in rows))

    def tearDown(self) -> None:
        shutil.rmtree(self.dir)

    def test_all_failures_summarised_and_logged(self) -> None:
        with self.assertRaisesRegex(ValueError, "3 张图像里有 2 张处理失败"):
            real_data.load_real_dataset(str(self.dir / "meta.csv"))
        cache = self.dir / "_cache"
        self.assertEqual(list(cache.glob("*.npz")), [])             # 有失败就不写缓存
        with next(cache.glob("*_roi_qc.csv")).open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        failed = [r for r in rows if r["失败原因"]]
        self.assertEqual({r["egg_id"] for r in failed}, {"e2", "e3"})
        self.assertIn("不像椭圆", failed[0]["失败原因"])
        self.assertEqual([r["失败原因"] for r in rows if r["egg_id"] == "e1"], [""])
        ok = next(r for r in rows if r["egg_id"] == "e1")
        self.assertTrue(0 < float(ok["蛋保留率"]) <= 1)
        self.assertEqual(failed[0]["蛋保留率"], "")


if __name__ == "__main__":
    unittest.main()
