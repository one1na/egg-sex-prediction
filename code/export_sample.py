"""把一小份模拟数据导出成"真实数据"的文件格式，用来演示和测试 real_data.py 的整条处理链。

模拟的是线扫相机：每次采集拍一张整幅原始图像（蛋在黑背景上，读数是 uint16 整数），
同一批次、同一天再拍一张白板、一张暗电流（只扫 REF_LINES 行、整幅宽度；两边光照弱，各波段灵敏度不同）。
读取时必须先校正、再找出蛋、裁剪缩放，才能还原出模拟数据里的那一帧。
运行：py -3.11 export_sample.py
生成：data/sample/meta.csv + data/sample/files/*.npy
"""
import logging

import numpy as np

from paths import SAMPLE_DIR
from run_record import save_csv
from synth_data import NOISE_STD, EggDataset, make_dataset

N_EGGS = 40
SEED = 1
OUT_DIR = SAMPLE_DIR
IMAGE_SIZE = 28          # 整幅图像的边长（像素），蛋放在正中间
BACKGROUND = 0.03        # 黑背景的反射率
BACKGROUND_NOISE = 0.005
WHITE_LEVEL = 4000.0     # 白板在画面中间、最灵敏波段的读数
DARK_LEVEL = 100.0       # 暗电流读数
REF_LINES = 20           # 白板、暗电流各扫多少行
FALLOFF = 0.3            # 光照从中间到两边减弱的比例
REF_NOISE = 20.0         # 参考板读数的噪声
WAVELENGTH_FILE = "files/wavelengths.npy"

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("export_sample")


def egg_image(frame_chw: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """把一帧放进圆形的蛋里（这一帧正好是圆的内接正方形），蛋放在黑背景上。返回反射率 (行, 列, 波段)。"""
    bands, size, _ = frame_chw.shape
    center, radius = IMAGE_SIZE / 2, size / np.sqrt(2)
    yy, xx = np.mgrid[:IMAGE_SIZE, :IMAGE_SIZE] + 0.5
    inside = (yy - center) ** 2 + (xx - center) ** 2 <= radius ** 2
    edge = np.concatenate([frame_chw[:, 0], frame_chw[:, -1], frame_chw[:, :, 0], frame_chw[:, :, -1]], axis=1)
    img = BACKGROUND + rng.normal(0, BACKGROUND_NOISE, size=(IMAGE_SIZE, IMAGE_SIZE, bands))
    img[inside] = edge.mean(axis=1) + rng.normal(0, NOISE_STD, size=(int(inside.sum()), bands))  # 蛋壳
    top = int(center - size / 2)
    img[top:top + size, top:top + size] = frame_chw.transpose(1, 2, 0)
    return img.astype(np.float32)


def session_levels(bands: int) -> tuple[np.ndarray, np.ndarray]:
    """相机看 100% 反射白板和不进光时的读数 (列, 波段)：两边光照弱，中间波段最灵敏。"""
    cols = (np.arange(IMAGE_SIZE) + 0.5 - IMAGE_SIZE / 2) / (IMAGE_SIZE / 2)
    light = 1 - FALLOFF * cols ** 2
    response = 0.5 + 0.5 * np.exp(-((np.arange(bands) - bands / 2) / (bands / 3)) ** 2)
    return WHITE_LEVEL * light[:, None] * response[None, :], np.full((IMAGE_SIZE, bands), DARK_LEVEL)


def to_counts(values: np.ndarray) -> np.ndarray:
    """相机读数是整数。"""
    return np.clip(np.round(values), 0, np.iinfo(np.uint16).max).astype(np.uint16)


def export_session(data: EggDataset, b: int, t: int, rng: np.random.Generator) -> list[dict]:
    """一个批次某一天的采集：一张白板、一张暗电流，加上这批蛋当天的图像。返回 meta.csv 的行。"""
    batch, day = f"B{b + 1:02d}", int(data.days[t])
    white_level, dark_level = session_levels(data.x.shape[2])
    shape = (REF_LINES, *white_level.shape)
    white, dark = f"files/white_{batch}_d{day}.npy", f"files/dark_{batch}_d{day}.npy"
    np.save(OUT_DIR / white, to_counts(white_level + rng.normal(0, REF_NOISE, shape)))
    np.save(OUT_DIR / dark, to_counts(dark_level + rng.normal(0, REF_NOISE / 4, shape)))
    rows = []
    for i in np.flatnonzero((data.batch == b) & data.mask[:, t]):
        cube = f"files/egg{i:03d}_d{day}.npy"
        raw = egg_image(data.x[i, t], rng) * (white_level - dark_level) + dark_level
        np.save(OUT_DIR / cube, to_counts(raw))
        rows.append({"egg_id": f"egg{i:03d}", "day": day, "batch": batch, "sex": int(data.sex[i]),
                     "dev": f"{data.dev[i, t]:.4f}", "cube": cube, "white": white, "dark": dark,
                     "wavelengths": WAVELENGTH_FILE})
    return rows


def main() -> None:
    data = make_dataset(n_eggs=N_EGGS, seed=SEED)
    rng = np.random.default_rng(SEED)
    (OUT_DIR / "files").mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / WAVELENGTH_FILE, data.wavelengths)
    rows =[row for b in np.unique(data.batch) for t in range(len(data.days))
            for row in export_session(data, int(b), t, rng)]
    save_csv(OUT_DIR / "meta.csv", rows)
    log.info(f"已导出 {len(rows)} 次采集（{len(np.unique(data.batch))} 个批次）到 {OUT_DIR}")


if __name__ == "__main__":
    main()
