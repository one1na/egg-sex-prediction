"""生成模拟的四维种蛋高光谱数据：(蛋, 孵化日, 波段, 高, 宽)。

只用来跑通流程，不代表真实光谱。规律设定为：
- 胚胎区域随发育逐渐变大，光谱随发育变化；
- 性别差异只体现在少数波段，且随发育增强（第 0 天完全没有，早期难、后期易）；
- 每次采集的光源强度有轻微漂移，部分蛋会缺某些天；
- 蛋分批入孵，每批有自己的整体亮度和光谱倾斜（批次效应，和性别无关）。
"""
from dataclasses import dataclass

import numpy as np

DAYS = np.array([0, 2, 4, 6, 8, 10], dtype=np.float32)
N_BANDS = 32
WAVELENGTH_RANGE = (400.0, 1000.0)   # 模拟相机的波长范围（nm）
IMG_SIZE = 16
SEX_BANDS = slice(18, 22)       # 性别差异所在波段
SEX_EFFECT = 0.012              # 性别信号强度（发育完成时）
NOISE_STD = 0.06
LIGHT_DRIFT_STD = 0.04          # 每次采集的光源强度漂移
MISSING_RATE = 0.1              # 某天未采集的概率
EGGS_PER_BATCH = 16             # 每个入孵批次的蛋数
BATCH_LEVEL_STD = 0.03          # 批次之间的整体亮度差异
BATCH_TILT_STD = 0.03           # 批次之间的光谱倾斜（短波、长波此消彼长）


@dataclass(frozen=True)
class EggDataset:
    egg_ids: tuple[str, ...]    # (N,)
    x: np.ndarray               # (N, T, C, H, W) 反射率
    spec: np.ndarray            # (N, T, C) 每帧蛋区域的平均光谱，基线方法用
    mask: np.ndarray            # (N, T) 该天是否有数据
    sex: np.ndarray             # (N,) 0=雌, 1=雄
    dev: np.ndarray             # (N, T) 发育指数，0~1
    dev_mask: np.ndarray        # (N, T) 该帧有没有发育指数的实测值
    batch: np.ndarray           # (N,) 入孵批次，交叉验证按它分组
    days: np.ndarray            # (T,)
    wavelengths: np.ndarray | None = None   # (C,) 每个波段的波长（nm）
    bg_spec: np.ndarray | None = None       # (N, T, C) 蛋外面一圈背景的平均光谱，只做背景对照；模拟数据没有


def _shell_spectrum(bands: np.ndarray) -> np.ndarray:
    return 0.5 + 0.2 * np.sin(bands / 6.0)


def _embryo_spectrum(bands: np.ndarray, dev: float) -> np.ndarray:
    # 发育越成熟，吸收峰越深、位置略右移
    center = 10 + 4 * dev
    return -0.25 * dev * np.exp(-((bands - center) ** 2) / 18.0)


def _embryo_mask(dev: float, rng: np.random.Generator) -> np.ndarray:
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    cy, cx = IMG_SIZE / 2 + rng.normal(0, 0.5, size=2)
    radius = 1.5 + 5.0 * dev
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return np.exp(-dist2 / (2 * radius**2)).astype(np.float32)


def _frame(sex: int, dev: float, rng: np.random.Generator) -> np.ndarray:
    bands = np.arange(N_BANDS, dtype=np.float32)
    embryo = _embryo_spectrum(bands, dev)
    sex_signal = np.zeros(N_BANDS, dtype=np.float32)
    sex_signal[SEX_BANDS] = (1 if sex else -1) * SEX_EFFECT * dev
    spatial = _embryo_mask(dev, rng)

    cube = (_shell_spectrum(bands)[:, None, None]
            + (embryo + sex_signal)[:, None, None] * spatial[None])
    light = 1.0 + rng.normal(0, LIGHT_DRIFT_STD)
    noise = rng.normal(0, NOISE_STD, size=cube.shape)
    return (cube * light + noise).astype(np.float32)


def _batch_gains(n_batches: int, rng: np.random.Generator) -> np.ndarray:
    """每个批次一条乘到光谱上的曲线：整体亮度 × (1 + 倾斜)，形状 (批次数, 波段)。"""
    slope = np.linspace(-1, 1, N_BANDS)
    level = 1 + rng.normal(0, BATCH_LEVEL_STD, size=(n_batches, 1))
    tilt = rng.normal(0, BATCH_TILT_STD, size=(n_batches, 1))
    return (level * (1 + tilt * slope)).astype(np.float32)


def make_dataset(n_eggs: int = 160, seed: int = 0) -> EggDataset:
    rng = np.random.default_rng(seed)
    sex = rng.integers(0, 2, size=n_eggs)
    growth_rate = rng.uniform(0.8, 1.2, size=n_eggs)  # 个体发育快慢不同
    dev = np.clip(growth_rate[:, None] * DAYS[None] / DAYS.max(), 0, 1).astype(np.float32)
    mask = rng.random((n_eggs, len(DAYS))) > MISSING_RATE
    batch = np.arange(n_eggs) // EGGS_PER_BATCH
    gains = _batch_gains(int(batch.max()) + 1, rng)

    x = np.zeros((n_eggs, len(DAYS), N_BANDS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i in range(n_eggs):
        for t in range(len(DAYS)):
            if mask[i, t]:
                x[i, t] = _frame(int(sex[i]), float(dev[i, t]), rng) * gains[batch[i], :, None, None]
    return EggDataset(egg_ids=tuple(f"syn{i:03d}" for i in range(n_eggs)), x=x,
                      spec=x.mean(axis=(3, 4)), mask=mask, sex=sex, dev=dev,
                      dev_mask=mask.copy(), batch=batch, days=DAYS,
                      wavelengths=np.linspace(*WAVELENGTH_RANGE, N_BANDS))
