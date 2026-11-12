"""单张高光谱图像的预处理：白板/暗电流校正 → 找出蛋 → 裁剪缩放。只做计算，不读写文件。

数组形状约定为 (行, 列, 波段)，行 = 线扫相机的扫描方向。
白板/暗电流一般只扫几十行，先沿扫描方向取平均，再逐列校正；
所以图像必须是相机拍出来的整幅宽度：先校正，再裁剪。
任何一步数据不合格都直接报错，不靠调小分母、换掉异常值之类的办法糊过去。
"""
import numpy as np
import torch
import torch.nn.functional as F

import roi

TARGET_SIZE = 16             # 模型输入的边长

# 校正的合格标准
MIN_REF_SPAN_RATIO = 0.05    # 每个像素、每个波段的 (白板-暗电流) 至少是整体中位数的 5%
MIN_REF_SNR = 10.0           # (白板-暗电流) 至少是暗电流扫描噪声（沿扫描方向的标准差）的这么多倍。要按相机标定
REF_NOISE_FLOOR = 1e-3       # 噪声估计的下限（暗电流只有平均值、或各行完全一样时用），反射率量级的数据也不会误报
REFL_RANGE = (-0.2, 3.0)     # 反射率的合理范围（高光点可以略超过 1）
MAX_OUT_OF_RANGE = 0.01      # 超出合理范围的读数最多占 1%


# ---------- 校正 ----------
def reduce_reference(ref: np.ndarray) -> np.ndarray:
    """白板/暗电流 -> (1, 列, 波段)：沿扫描方向取平均。也接受已经平均好的 (列, 波段) 或 (波段,)。"""
    if ref.ndim == 3:
        return ref.mean(axis=0, keepdims=True)
    if ref.ndim in (1, 2):
        return ref.reshape((1,) * (3 - ref.ndim) + ref.shape)
    raise ValueError(f"参考板应该是 1~3 维数组，实际形状 {ref.shape}")


def _dark_noise(dark: np.ndarray, like: np.ndarray) -> np.ndarray:
    """暗电流沿扫描方向的标准差 (1, 列, 波段)；只有平均值（或只有一行）时没法估计，用下限。"""
    noise = np.zeros_like(like)
    if dark.ndim == 3 and dark.shape[0] > 1:
        noise = dark.std(axis=0, ddof=1, keepdims=True)
    return np.maximum(noise, REF_NOISE_FLOOR)


def _check_span(span: np.ndarray, noise: np.ndarray) -> None:
    """白板必须明显比暗电流亮（相对整体、也相对暗电流噪声），否则除出来的反射率没有意义。"""
    if not np.isfinite(span).all():
        raise ValueError("白板或暗电流里有 NaN / 无穷大")
    median = float(np.median(span))
    if median <= 0:
        raise ValueError("白板整体不比暗电流亮，白板和暗电流是不是放反了？")
    bad = span < MIN_REF_SPAN_RATIO * median
    if bad.any():
        raise ValueError(f"有 {int(bad.sum())} 处白板和暗电流几乎一样（不到中位数的 {MIN_REF_SPAN_RATIO:.0%}），"
                         f"涉及波段序号 {_band_list(bad)}。请检查参考板采集")
    weak = span < MIN_REF_SNR * noise
    if weak.any():
        raise ValueError(f"有 {int(weak.sum())} 处白板比暗电流只亮一点点（不到暗电流噪声的 {MIN_REF_SNR:.0f} 倍），"
                         f"信号太弱，反射率会被噪声淹没。涉及波段序号 {_band_list(weak)}。请检查白板曝光")


def _band_list(bad: np.ndarray) -> str:
    bands = sorted({int(b) for b in np.nonzero(bad)[-1]})
    return f"{bands[:10]}{'……' if len(bands) > 10 else ''}"


def check_reflectance(refl: np.ndarray) -> None:
    """反射率必须全是有限数，超出合理范围的读数不能太多。"""
    if not np.isfinite(refl).all():
        raise ValueError("反射率里有 NaN / 无穷大")
    low, high = REFL_RANGE
    outside = float(((refl < low) | (refl > high)).mean())
    if outside > MAX_OUT_OF_RANGE:
        raise ValueError(f"{outside:.1%} 的反射率超出合理范围 {REFL_RANGE}，校正或曝光可能有问题")


def calibrate(raw: np.ndarray, white: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """原始读数 -> 反射率：(raw - dark) / (white - dark)，逐列、逐波段计算。"""
    if raw.ndim != 3:
        raise ValueError(f"图像应该是 (行, 列, 波段) 三维数组，实际形状 {raw.shape}")
    refs = {"白板": reduce_reference(white), "暗电流": reduce_reference(dark)}
    for name, ref in refs.items():
        if ref.shape[2] != raw.shape[2]:
            raise ValueError(f"{name}有 {ref.shape[2]} 个波段，图像有 {raw.shape[2]} 个")
        if ref.shape[1] not in (1, raw.shape[1]):
            raise ValueError(f"{name}宽 {ref.shape[1]} 列，图像宽 {raw.shape[1]} 列。"
                             "请用相机拍出来的整幅图像，先校正再裁剪")
    span = refs["白板"] - refs["暗电流"]
    _check_span(span, _dark_noise(dark, span))
    refl = (raw - refs["暗电流"]) / span
    check_reflectance(refl)
    return refl


# ---------- 裁剪缩放 ----------
def to_model_frame(refl: np.ndarray) -> np.ndarray:
    """(行, 列, 波段) -> (波段, TARGET_SIZE, TARGET_SIZE)，按面积平均缩放。"""
    chw = np.ascontiguousarray(refl.transpose(2, 0, 1), dtype=np.float32)
    resized = F.interpolate(torch.from_numpy(chw)[None], size=(TARGET_SIZE, TARGET_SIZE), mode="area")
    return resized[0].numpy()


def process_image(refl: np.ndarray, roi_mode: str | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, roi.RoiQC | None]:
    """反射率图像 -> (模型输入帧 (波段, S, S), 蛋区域平均光谱 (波段,), 蛋的掩码, 质量记录)。

    找蛋方式见 roi.py 的 ROI_MODE；"none" 时整幅图都当成蛋，没有掩码和质量记录。
    """
    mode = roi.ROI_MODE if roi_mode is None else roi_mode
    roi.check_config(mode)
    check_reflectance(refl)
    if mode == "none":
        return to_model_frame(refl), refl.reshape(-1, refl.shape[2]).mean(axis=0), None, None
    egg, qc = roi.egg_mask(refl, roi.FIXED_ROI if mode == "fixed" else None)
    return to_model_frame(roi.crop_inside(refl, qc)), refl[egg].mean(axis=0), egg, qc
