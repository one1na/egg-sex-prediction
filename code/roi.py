"""在反射率图像里找出蛋，并算出一块"全是蛋"的裁剪区域。只做计算，不读写文件。

数组形状约定为 (行, 列, 波段)。蛋是一块不碰图像边缘、大致椭圆的亮区（可以是任意朝向）。
找蛋的流程：Otsu 阈值分出前景 → 逐块检查形状（旋转不变）和局部对比度 → 恰好一块合格才算找到。
合格的区域不止一块时直接报错，不替用户挑：形状分不出蛋和别的圆形亮物。
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

# 找蛋的方式："auto" 自动找；"fixed" 只在 FIXED_ROI 窗口里找（蛋位固定的夹具，最可靠）；
# "none" 整幅图都当成蛋（图像已经只剩蛋时用）
ROI_MODE = "auto"
FIXED_ROI: tuple[int, int, int, int] | None = None   # (行起, 行止, 列起, 列止)，ROI_MODE 为 "fixed" 时必填
ROI_MODES = ("auto", "fixed", "none")
HIST_BINS = 256

# 找蛋的合格标准
MIN_CONTRAST = 5.0           # 蛋和周围背景的局部对比度。这只是初始值，必须用真实图像标定
CONTRAST_NOISE_FLOOR = 0.002     # 对比度分母（波动）的下限，反射率单位；背景完全平坦时不让对比度爆炸。初始值，必须用真实图像标定
MIN_EGG_BG_DIFF = 0.02           # 蛋和背景亮度中位数的绝对差下限，反射率单位。初始值，必须用真实图像标定
EGG_AREA_RANGE = (0.02, 0.9)     # 蛋占整幅图（fixed 模式是窗口）面积的比例
EGG_FILL_RANGE = (0.65, 0.9)     # 面积 / 沿主轴摆正后的外接矩形面积；椭圆约 0.785，矩形是 1
MAX_EGG_ASPECT = 2.0             # 摆正后长边 / 短边；鸡蛋约 1.3
ROI_CANDIDATES = 5               # 最多检查面积最大的几块亮区
RING_WIDTH = 4                   # 局部对比度：取蛋外面多宽（像素）的一圈当背景
BG_GAP = 2                       # 背景对照取的那一圈离蛋边多远（像素），避开蛋边缘的过渡像素
MAD_TO_SIGMA = 1.4826            # 中位数绝对偏差换算成标准差

# 裁剪区域
MIN_CROP_IN_MASK = 0.98      # 裁剪区域里属于蛋的像素至少占这么多
CROP_SHRINK = 0.95           # 不够干净时每次缩小到原来的多少
MIN_CROP_SIDE = 4            # 裁剪区域最短边至少多少像素

ROI_HINT = "如果图像已经裁剪到只剩蛋，请把 roi.py 里的 ROI_MODE 改成 \"none\""
FIXED_HINT = "建议用固定蛋位夹具，并把 roi.py 里的 ROI_MODE 设成 \"fixed\"、填好 FIXED_ROI"


@dataclass(frozen=True)
class RoiQC:
    """找蛋结果的质量记录，写进缓存旁边的 roi_qc.csv 供人工抽检。"""
    area_fraction: float     # 蛋占整幅图的比例
    fill: float              # 面积 / 沿主轴摆正后的外接矩形面积
    aspect: float            # 摆正后长边 / 短边
    angle: float             # 长轴和列方向的夹角（度，0~180）
    bbox: tuple[int, int, int, int]    # 行起、行止、列起、列止（不旋转的外接框）
    crop: tuple[int, int, int, int]    # 裁剪区域，同上
    crop_in_mask: float      # 裁剪区域里属于蛋的像素比例（纯度：有没有混进背景）
    contrast: float          # 局部对比度
    egg_retention: float     # 整个蛋区域里被裁剪保留下来的比例（保留率：丢了多少蛋）。斜放的长蛋会明显偏低


@dataclass(frozen=True)
class _Shape:
    center: np.ndarray       # (行, 列) 质心
    cov: np.ndarray          # 像素坐标的协方差，(行, 列) 顺序
    fill: float
    aspect: float
    angle: float


def check_config(mode: str | None = None) -> None:
    """ROI_MODE 和 FIXED_ROI 的设置必须合法，否则一开始就报错。"""
    mode = ROI_MODE if mode is None else mode
    if mode not in ROI_MODES:
        raise ValueError(f"ROI_MODE 只能是 {ROI_MODES}，现在是 {mode!r}")
    if mode != "fixed":
        return
    window = FIXED_ROI
    if window is None or len(window) != 4 or not (0 <= window[0] < window[1] and 0 <= window[2] < window[3]):
        raise ValueError(f"ROI_MODE 是 \"fixed\"，FIXED_ROI 必须填 (行起, 行止, 列起, 列止)，现在是 {window!r}")


# ---------- 形状 ----------
def otsu_threshold(values: np.ndarray) -> float:
    """Otsu 阈值：把数值分成高低两类，让两类之间差别最大。"""
    hist, edges = np.histogram(values, bins=HIST_BINS)
    centers = (edges[:-1] + edges[1:]) / 2
    n_low = np.cumsum(hist)
    n_high = n_low[-1] - n_low
    sum_low = np.cumsum(hist * centers)
    mean_low = sum_low / np.maximum(n_low, 1)
    mean_high = (sum_low[-1] - sum_low) / np.maximum(n_high, 1)
    return float(edges[np.argmax(n_low * n_high * (mean_low - mean_high) ** 2) + 1])


def _bbox(region: np.ndarray) -> tuple[int, int, int, int]:
    rows, cols = np.nonzero(region)
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def _shape(region: np.ndarray) -> _Shape:
    """用二阶矩（主成分）求朝向，在摆正后的坐标系里量填充率和长宽比：转多少度都一样。"""
    pts = np.argwhere(region).astype(float)
    center = pts.mean(axis=0)
    cov = np.cov(pts.T)
    _, vecs = np.linalg.eigh(cov)                  # 特征值从小到大：第 0 列是短轴，第 1 列是长轴
    proj = (pts - center) @ vecs
    minor, major = np.ptp(proj, axis=0) + 1
    angle = float(np.degrees(np.arctan2(vecs[0, 1], vecs[1, 1])) % 180)
    return _Shape(center, cov, float(len(pts) / (major * minor)), float(major / minor), angle)


def _local_contrast(brightness: np.ndarray, region: np.ndarray) -> tuple[float, float]:
    """蛋和它外面一圈背景的 (局部对比度, 亮度中位数之差的绝对值)。

    对比度 = 亮度差 / 两边各自的波动，用中位数和 MAD，背景不均匀、有杂点时也稳；波动不低于 CONTRAST_NOISE_FLOOR。
    """
    ring = ndimage.binary_dilation(region, iterations=RING_WIDTH) & ~region
    if not ring.any():
        return 0.0, 0.0
    inside, outside = brightness[region], brightness[ring]
    mad = [np.median(np.abs(v - np.median(v))) * MAD_TO_SIGMA for v in (inside, outside)]
    spread = max(float(np.sqrt((mad[0] ** 2 + mad[1] ** 2) / 2)), CONTRAST_NOISE_FLOOR)
    diff = float(abs(np.median(inside) - np.median(outside)))
    return diff / spread, diff


# ---------- 裁剪 ----------
def _crop_box(center: np.ndarray, half: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0 = np.ceil(center - half).astype(int)
    r1, c1 = np.floor(center + half).astype(int) + 1
    return max(int(r0), 0), min(int(r1), shape[0]), max(int(c0), 0), min(int(c1), shape[1])


def _inscribed_crop(region: np.ndarray, shape: _Shape) -> tuple[tuple[int, int, int, int], float] | None:
    """以质心为中心、沿矩给出的椭圆内接框起步，逐步缩小到里面至少 MIN_CROP_IN_MASK 是蛋；缩得太小返回 None。"""
    var = np.diag(shape.cov)
    corr = shape.cov[0, 1] / np.sqrt(var[0] * var[1])
    half = 2 * np.sqrt(var) * np.sqrt((1 - abs(corr)) / 2)     # 椭圆的轴对齐内接矩形
    while True:
        box = _crop_box(shape.center, half, region.shape)
        r0, r1, c0, c1 = box
        if min(r1 - r0, c1 - c0) < MIN_CROP_SIDE:
            return None
        inside = float(region[r0:r1, c0:c1].mean())
        if inside >= MIN_CROP_IN_MASK:
            return box, inside
        half = half * CROP_SHRINK


def background_ring(egg: np.ndarray) -> np.ndarray:
    """蛋外面一圈背景（离蛋边 BG_GAP 像素起、宽 RING_WIDTH），只用来做"背景对照"，不进模型。"""
    near = ndimage.binary_dilation(egg, iterations=BG_GAP)
    return ndimage.binary_dilation(near, iterations=RING_WIDTH) & ~near


def crop_inside(refl: np.ndarray, qc: RoiQC) -> np.ndarray:
    """裁出蛋里面的内接矩形：全是蛋，四角不带背景。"""
    r0, r1, c0, c1 = qc.crop
    return refl[r0:r1, c0:c1]


# ---------- 找蛋 ----------
def _check_candidate(brightness: np.ndarray, region: np.ndarray) -> tuple[RoiQC | None, str]:
    """一块区域像不像蛋；返回质量记录和不合格的原因（合格时原因为空字符串）。"""
    r0, r1, c0, c1 = _bbox(region)
    if r0 == 0 or c0 == 0 or r1 == region.shape[0] or c1 == region.shape[1]:
        return None, "碰到了图像边缘（蛋可能没拍全）"
    area = float(region.mean())
    if not EGG_AREA_RANGE[0] <= area <= EGG_AREA_RANGE[1]:
        return None, f"面积占整幅图 {area:.1%}，不在 {EGG_AREA_RANGE} 之内"
    shape = _shape(region)
    if not EGG_FILL_RANGE[0] <= shape.fill <= EGG_FILL_RANGE[1]:
        return None, f"形状不像椭圆（填充率 {shape.fill:.2f}，椭圆约 0.79，矩形是 1）"
    if shape.aspect > MAX_EGG_ASPECT:
        return None, f"太细长（长宽比 {shape.aspect:.1f}）"
    contrast, diff = _local_contrast(brightness, region)
    if diff < MIN_EGG_BG_DIFF:
        return None, f"和周围背景的亮度只差 {diff:.2g}（反射率），不到 {MIN_EGG_BG_DIFF}，分不清蛋和背景"
    if contrast < MIN_CONTRAST:
        return None, f"和周围背景分不清（局部对比度 {contrast:.1f}，至少要 {MIN_CONTRAST:.0f}）"
    crop = _inscribed_crop(region, shape)
    if crop is None:
        return None, f"裁不出至少 {MIN_CROP_IN_MASK:.0%} 是蛋的内部区域"
    return RoiQC(area, shape.fill, shape.aspect, shape.angle, (r0, r1, c0, c1), crop[0], crop[1], contrast,
                 _retention(region, crop[0])), ""


def _retention(region: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """蛋保留率：裁剪框里的蛋像素 / 整个蛋的像素。"""
    r0, r1, c0, c1 = box
    return float(region[r0:r1, c0:c1].sum() / region.sum())


def _foreground(brightness: np.ndarray) -> np.ndarray:
    fg = brightness > otsu_threshold(brightness)
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    fg = ~fg if border.mean() > 0.5 else fg          # 背景比蛋亮，反过来
    if fg.all() or not fg.any():
        raise ValueError(f"整幅图亮度都一样，分不出蛋和背景。{ROI_HINT}")
    return fg


def _find_egg(refl: np.ndarray) -> tuple[np.ndarray, RoiQC]:
    """在面积最大的几块前景里找像蛋的；必须恰好一块合格，0 块或多块都报错并说明原因。"""
    brightness = refl.mean(axis=2)
    labels, _ = ndimage.label(_foreground(brightness))
    sizes = np.bincount(labels.ravel())[1:]
    found, reasons = [], []
    for label in np.argsort(sizes)[::-1][:ROI_CANDIDATES] + 1:
        region = ndimage.binary_fill_holes(labels == label)
        qc, reason = _check_candidate(brightness, region)
        if qc is None:
            reasons.append(f"面积第 {len(reasons) + len(found) + 1} 大的一块{reason}")
        else:
            found.append((region, qc))
    if len(found) > 1:
        boxes = "；".join(f"外接框 {qc.bbox}" for _, qc in found)
        raise ValueError(f"找到多个像蛋的区域（{len(found)} 个：{boxes}），无法确定哪个是蛋。{FIXED_HINT}")
    if not found:
        raise ValueError("没找到像蛋的区域：" + "；".join(reasons) + f"。{ROI_HINT}")
    return found[0]


def _shift(qc: RoiQC, dr: int, dc: int) -> RoiQC:
    def move(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return box[0] + dr, box[1] + dr, box[2] + dc, box[3] + dc
    return replace(qc, bbox=move(qc.bbox), crop=move(qc.crop))


def egg_mask(refl: np.ndarray, window: tuple[int, int, int, int] | None = None) -> tuple[np.ndarray, RoiQC]:
    """找出蛋。给了 window (行起, 行止, 列起, 列止) 就只在窗口里找，蛋必须完整落在窗口内。

    返回整幅图大小的蛋掩码和质量记录；窗口模式下面积占比是相对窗口算的。
    """
    if window is None:
        return _find_egg(refl)
    r0, r1, c0, c1 = window
    if r1 > refl.shape[0] or c1 > refl.shape[1]:
        raise ValueError(f"FIXED_ROI {window} 超出了图像（{refl.shape[0]} 行 × {refl.shape[1]} 列）")
    region, qc = _find_egg(refl[r0:r1, c0:c1])
    full = np.zeros(refl.shape[:2], dtype=bool)
    full[r0:r1, c0:c1] = region
    return full, _shift(qc, r0, c0)
