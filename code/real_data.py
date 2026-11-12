"""读取真实高光谱数据（一张 meta.csv + 若干图像文件），整理成和模拟数据一样的 EggDataset。

meta.csv 每行是一次采集，列为：
    egg_id       蛋编号
    day          孵化日（整数）
    sex          0=雌, 1=雄（同一颗蛋每行必须一致；判断新蛋时不需要）
    batch        入孵批次（可选）。交叉验证按批次分组；没有这一列或留空，就每颗蛋自成一组
    dev          发育指标（可选，可留空）。只填 0~1 的实测值；留空的帧不参与发育训练，全部留空就关掉发育这一项
    cube         相机拍出来的整幅高光谱图像（相对 meta.csv 的路径），不要事先裁剪
    white        这次采集的白板文件；white 和 dark 都留空，表示 cube 已经是反射率
    dark         这次采集的暗电流文件
    wavelengths  波长文件（可选，.npy / .txt / .csv，每个波段一个数，单位 nm；也可以直接填 ENVI 的 .hdr，从里面读）。
                 填了就检查所有采集的波长轴一致
文件格式支持 .npy、.mat（数组形状为 (行, 列, 波段)）和 ENVI（填 .hdr，数据文件放在同一文件夹，见 envi.py）。
处理顺序：校正 → 找出蛋 → 裁剪缩放（见 calibration.py、roi.py）。
处理结果缓存在 meta.csv 旁边的 _cache 文件夹，图像文件和预处理代码都没变就直接读缓存。
同一文件夹里还有 <编号>_roi_qc.csv（每张图找蛋的质量记录，处理失败的图也有一行，写明失败原因）和 <编号>_roi_masks.npz（找到的蛋的掩码），供人工抽检。
"""
import csv
import hashlib
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import calibration
import envi
import roi
from calibration import calibrate, process_image
from paths import CACHE_DIRNAME
from run_record import save_csv
from synth_data import EggDataset

CACHE_VERSION = 3            # 缓存文件格式变了就加 1；预处理代码的改动会自动让缓存作废
LOG_EVERY = 50               # 预处理时每多少次采集报一次进度
MAT_VARIABLE: str | None = None   # .mat 文件里图像变量的名字（如 "spec"）；None = 文件里只有一个变量时自动用它
WAVELENGTH_TOLERANCE = 0.5   # 不同采集的波长允许相差多少 nm
TRAIN_COLUMNS = ("egg_id", "day", "sex", "cube")
PREDICT_COLUMNS = ("egg_id", "day", "cube")

Row = dict[str, str]
log = logging.getLogger("real_data")


@dataclass(frozen=True)
class EggFrames:
    """预处理后的图像，不带标签；预测新蛋时直接用。"""
    egg_ids: tuple[str, ...]
    days: np.ndarray    # (T,)
    x: np.ndarray       # (N, T, C, H, W) 反射率
    spec: np.ndarray    # (N, T, C) 蛋区域的平均光谱
    mask: np.ndarray    # (N, T)
    wavelengths: np.ndarray | None = None   # (C,)
    bg_spec: np.ndarray | None = None       # (N, T, C) 蛋外面一圈背景的平均光谱，只做背景对照；整幅都是蛋时为 nan


# ---------- 读表 ----------
def _cell(row: Row, column: str) -> str:
    return (row.get(column) or "").strip()


def _where(row: Row) -> str:
    return f"蛋 {row['egg_id']} 第 {_cell(row, 'day')} 天"


def _day(row: Row) -> int:
    try:
        return int(_cell(row, "day"))
    except ValueError:
        raise ValueError(f"蛋 {row['egg_id']} 的孵化日 {_cell(row, 'day')!r} 不是整数") from None


def _check_unique(rows: list[Row]) -> None:
    seen: set[tuple[str, int]] = set()
    for r in rows:
        key = (r["egg_id"], _day(r))
        if key in seen:
            raise ValueError(f"{_where(r)}在表里出现了不止一行")
        seen.add(key)


def read_meta(csv_path: Path, required: tuple[str, ...]) -> list[Row]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到数据表：{csv_path}")
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{csv_path} 是空表")
    missing = [c for c in required if c not in rows[0]]
    if missing:
        raise ValueError(f"{csv_path} 缺少列：{missing}")
    _check_unique(rows)
    return rows


# ---------- 读文件 ----------
def _load_mat(path: Path) -> np.ndarray:
    from scipy.io import loadmat
    arrays = {k: v for k, v in loadmat(path).items() if not k.startswith("__")}
    if MAT_VARIABLE is not None:
        if MAT_VARIABLE not in arrays:
            raise ValueError(f"{path} 里没有变量 {MAT_VARIABLE!r}，只有 {sorted(arrays)}")
        return np.asarray(arrays[MAT_VARIABLE], dtype=np.float32)
    if len(arrays) != 1:
        raise ValueError(f"{path} 里有 {len(arrays)} 个变量 {sorted(arrays)}，"
                         "请在 real_data.py 里把 MAT_VARIABLE 设成图像变量的名字")
    return np.asarray(next(iter(arrays.values())), dtype=np.float32)


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"找不到文件：{path}")
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix == ".mat":
        return _load_mat(path)
    if path.suffix.lower() == ".hdr":
        return envi.load_envi(path)
    raise ValueError(f"暂不支持的格式 {path.suffix}：{path}（目前支持 .npy / .mat / ENVI 的 .hdr）")


def _source_files(path: Path) -> list[Path]:
    """一个数据表格子对应的磁盘文件：ENVI 的 .hdr 还要连上它的数据文件（数据文件变了缓存也要作废）。"""
    return [path, envi.data_file(path)] if path.suffix.lower() == ".hdr" else [path]


def _load_wavelengths(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".hdr":
        values = envi.envi_wavelengths(path)
    elif path.suffix == ".npy":
        values = np.load(path)
    else:
        values = np.loadtxt(path, delimiter="," if path.suffix == ".csv" else None)
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError(f"{path} 里的波长必须是递增的有限数")
    return values


def _ref_names(row: Row) -> tuple[str, str]:
    white, dark = _cell(row, "white"), _cell(row, "dark")
    if bool(white) != bool(dark):
        raise ValueError(f"{_where(row)}：white 和 dark 要么都填，要么都留空")
    return white, dark


def _reflectance(base: Path, row: Row, refs: dict[Path, np.ndarray]) -> np.ndarray:
    """读一次采集并校正成反射率。refs 记着本次已读过的白板/暗电流，同一张只读一次。

    白板/暗电流保持原始的逐行扫描数据交给 calibrate：暗电流的噪声要靠这些行估计，先平均就看不到了。
    """
    raw = load_array(base / _cell(row, "cube"))
    white, dark = _ref_names(row)
    if not white:
        return raw
    for name in (white, dark):
        if base / name not in refs:
            refs[base / name] = load_array(base / name)
    return calibrate(raw, refs[base / white], refs[base / dark])


# ---------- 波长轴 ----------
def _wavelength_axis(base: Path, rows: list[Row]) -> np.ndarray | None:
    """所有采集的波长轴必须一致。没有 wavelengths 列返回 None；有这一列就每行都要填。"""
    if "wavelengths" not in rows[0]:
        return None
    axes: dict[str, np.ndarray] = {}
    for r in rows:
        name = _cell(r, "wavelengths")
        if not name:
            raise ValueError(f"{_where(r)}：wavelengths 列有的行填了有的没填")
        if name not in axes:
            axes[name] = _load_wavelengths(base / name)
    first_name, first = next(iter(axes.items()))
    for name, axis in axes.items():
        if axis.shape != first.shape or np.abs(axis - first).max() > WAVELENGTH_TOLERANCE:
            raise ValueError(f"波长轴不一致：{name} 和 {first_name} 不同，不同设置拍的数据不能混在一起")
    return first


# ---------- 预处理 ----------
QC_COLUMNS = ("egg_id", "day", "cube", "面积占比", "填充率", "长宽比", "长轴角度", "对比度",
              "裁剪区内蛋占比", "蛋保留率", "外接框(行起,行止,列起,列止)", "裁剪框(行起,行止,列起,列止)", "失败原因")
LOW_EGG_RETENTION = 0.4      # 蛋保留率低于它就提醒：裁剪只用了不到这么多的蛋区域。初始值，要用真实图像再定
RETENTION_SPREAD = 0.15      # 各张图保留率最高和最低差这么多就提醒：摆放角度不同，模型看到的蛋面积不一样


def _qc_row(r: Row, qc: roi.RoiQC | None, error: str = "") -> dict:
    """一张图的质量记录；所有行的列都一样（失败的行只有失败原因）。"""
    row = dict.fromkeys(QC_COLUMNS, "")
    row.update(egg_id=r["egg_id"], day=_day(r), cube=_cell(r, "cube"), 失败原因=error)
    if qc is not None:
        row.update({"面积占比": round(qc.area_fraction, 4), "填充率": round(qc.fill, 3),
                    "长宽比": round(qc.aspect, 3), "长轴角度": round(qc.angle, 1), "对比度": round(qc.contrast, 1),
                    "裁剪区内蛋占比": round(qc.crop_in_mask, 3), "蛋保留率": round(qc.egg_retention, 3),
                    "外接框(行起,行止,列起,列止)": " ".join(map(str, qc.bbox)),
                    "裁剪框(行起,行止,列起,列止)": " ".join(map(str, qc.crop))})
    return row


def _background(refl: np.ndarray, egg: np.ndarray | None) -> np.ndarray:
    """蛋外面一圈背景的平均光谱；没有蛋的掩码（ROI_MODE="none"）或那一圈出了图像时是 nan。"""
    ring = None if egg is None else roi.background_ring(egg)
    if ring is None or not ring.any():
        return np.full(refl.shape[2], np.nan, dtype=np.float32)
    return refl[ring].mean(axis=0)


def _process_rows(csv_file: Path, rows: list[Row]) -> tuple[EggFrames | None, list[dict], dict[str, np.ndarray]]:
    """逐行：读图 → 校正 → 找蛋 → 裁剪缩放，拼成 (蛋, 天, ...) 数组。另返回质量记录（含失败的）和掩码。

    某张图处理失败不会中断，全部处理完再汇总；有失败时第一项返回 None。
    """
    roi.check_config()
    egg_ids = tuple(sorted({r["egg_id"] for r in rows}))
    days = np.array(sorted({_day(r) for r in rows}), dtype=np.float32)
    egg_index = {e: i for i, e in enumerate(egg_ids)}
    day_index = {int(d): t for t, d in enumerate(days)}
    refs: dict[Path, np.ndarray] = {}
    mask = np.zeros((len(egg_ids), len(days)), dtype=bool)
    x = spec = bg = None
    qc_rows, masks = [], {}
    for k, r in enumerate(rows, 1):
        try:
            refl = _reflectance(csv_file.parent, r, refs)
            frame, s, egg, qc = process_image(refl)
        except ValueError as e:
            qc_rows.append(_qc_row(r, None, str(e)))
            continue
        if x is None:
            x = np.zeros((len(egg_ids), len(days), *frame.shape), dtype=np.float32)
            spec = np.zeros((len(egg_ids), len(days), len(s)), dtype=np.float32)
            bg = np.full_like(spec, np.nan)
        if frame.shape != x.shape[2:]:
            raise ValueError(f"{_where(r)}的波段数和前面的采集不一致")
        i, t = egg_index[r["egg_id"]], day_index[_day(r)]
        x[i, t], spec[i, t], mask[i, t], bg[i, t] = frame, s, True, _background(refl, egg)
        qc_rows.append(_qc_row(r, qc))
        if egg is not None:
            masks[f"{r['egg_id']}_d{_day(r)}"] = egg
        if k % LOG_EVERY == 0:
            log.info(f"  已预处理 {k}/{len(rows)} 次采集")
    if any(q["失败原因"] for q in qc_rows):
        return None, qc_rows, masks
    frames = EggFrames(egg_ids=egg_ids, days=days, x=x, spec=spec, mask=mask,
                       wavelengths=_wavelength_axis(csv_file.parent, rows), bg_spec=bg)
    if frames.wavelengths is not None and len(frames.wavelengths) != x.shape[2]:
        raise ValueError(f"波长文件有 {len(frames.wavelengths)} 个数，图像有 {x.shape[2]} 个波段")
    return frames, qc_rows, masks


# ---------- 预处理缓存 ----------
def _code_version() -> bytes:
    """预处理代码本身：calibration.py、roi.py、envi.py 里的任何阈值、步骤改了，缓存都会作废。"""
    return b"".join(Path(m).read_bytes() for m in (calibration.__file__, roi.__file__, envi.__file__, __file__))


def _fingerprint(csv_file: Path, rows: list[Row]) -> str:
    """缓存的身份：缓存格式、预处理代码、每个图像和波长文件的路径、大小、修改时间。标签列不影响缓存。"""
    h = hashlib.sha1(f"{CACHE_VERSION}|{MAT_VARIABLE}|".encode() + _code_version())
    for r in rows:
        h.update(f"|{r['egg_id']}|{_day(r)}".encode())
        for name in (_cell(r, "cube"), *_ref_names(r), _cell(r, "wavelengths")):
            if not name:
                continue
            path = csv_file.parent / name
            if not path.exists():
                raise FileNotFoundError(f"{_where(r)}：找不到文件 {path}")
            for part in _source_files(path):
                st = part.stat()
                h.update(f"|{part.name}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def _read_cache(path: Path) -> EggFrames:
    with np.load(path, allow_pickle=False) as z:
        wavelengths = z["wavelengths"]
        return EggFrames(egg_ids=tuple(z["egg_ids"].tolist()), days=z["days"], x=z["x"], spec=z["spec"],
                         mask=z["mask"], wavelengths=wavelengths if wavelengths.size else None,
                         bg_spec=z["bg_spec"] if z["bg_spec"].size else None)


def _write_qc(path: Path, qc_rows: list[dict]) -> None:
    """找蛋质量记录（含失败的图）。写不了只提醒，不影响运行。"""
    try:
        save_csv(path.with_name(f"{path.stem}_roi_qc.csv"), qc_rows)
    except OSError as e:
        log.warning(f"找蛋质量记录写不进 {path.parent}（{e}）")


def _write_cache(path: Path, frames: EggFrames, masks: dict[str, np.ndarray]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("wb") as f:
            np.savez(f, egg_ids=np.array(frames.egg_ids), days=frames.days, x=frames.x, spec=frames.spec,
                     mask=frames.mask, wavelengths=np.array([]) if frames.wavelengths is None else frames.wavelengths,
                     bg_spec=np.array([]) if frames.bg_spec is None else frames.bg_spec)
        os.replace(tmp, path)       # 写完再改名，中途出错不会留下坏缓存
        if masks:
            np.savez_compressed(path.with_name(f"{path.stem}_roi_masks.npz"), **masks)
        log.info(f"预处理结果已缓存到 {path}（找蛋质量记录：{path.stem}_roi_qc.csv）")
    except OSError as e:
        log.warning(f"缓存写不进 {path.parent}（{e}），不影响这次运行，下次还会重新预处理")


def load_frames(csv_file: Path, rows: list[Row]) -> EggFrames:
    """图像文件和预处理代码都没变就读缓存，否则重新预处理并写缓存。"""
    cache = csv_file.parent / CACHE_DIRNAME / f"{_fingerprint(csv_file, rows)}.npz"
    if cache.exists():
        try:
            frames = _read_cache(cache)
            log.info(f"读取预处理缓存：{cache}")
            return frames
        except (OSError, ValueError, KeyError) as e:
            log.warning(f"缓存 {cache} 读不了（{e}），重新预处理")
    log.info(f"预处理 {len(rows)} 次采集（只在第一次、图像或预处理代码有变动时做）……")
    frames, qc_rows, masks = _process_rows(csv_file, rows)
    _write_qc(cache, qc_rows)
    failed = [q for q in qc_rows if q["失败原因"]]
    if failed:
        first = failed[0]
        raise ValueError(f"{len(rows)} 张图像里有 {len(failed)} 张处理失败，没有生成缓存。"
                         f"每张图的失败原因见 {cache.with_name(f'{cache.stem}_roi_qc.csv')} 的“失败原因”列。"
                         f"例如蛋 {first['egg_id']} 第 {first['day']} 天（{first['cube']}）：{first['失败原因']}")
    _log_retention(qc_rows)
    _write_cache(cache, frames, masks)
    return frames


def _log_retention(qc_rows: list[dict]) -> None:
    """汇报蛋保留率：裁剪只取蛋里面的内接矩形，斜放的长蛋丢得多；各张图差得多，说明摆放角度影响了输入。"""
    values = np.array([q["蛋保留率"] for q in qc_rows if q["蛋保留率"] != ""], dtype=float)
    if not values.size:
        return
    log.info(f"蛋保留率（裁剪用到的蛋区域比例）：中位数 {np.median(values):.2f}，最低 {values.min():.2f}，"
             f"最高 {values.max():.2f}")
    low = int((values < LOW_EGG_RETENTION).sum())
    if low:
        log.warning(f"  有 {low} 张图的蛋保留率低于 {LOW_EGG_RETENTION}，裁剪丢掉了大半个蛋（多半是蛋斜放），"
                    "建议摆正蛋或用固定蛋位托架")
    if values.max() - values.min() > RETENTION_SPREAD:
        log.warning(f"  各张图的蛋保留率相差 {values.max() - values.min():.2f}：摆放角度不同，模型看到的蛋面积不一样，"
                    "建议统一蛋的朝向")


# ---------- 标签 ----------
def _read_sex(row: Row) -> int:
    value = _cell(row, "sex")
    if value not in ("0", "1"):
        raise ValueError(f"{_where(row)}的 sex 是 {value!r}，只能填 0（雌）或 1（雄）")
    return int(value)


def _solo_batch(egg_id: str) -> str:
    """没有批次信息的蛋用的占位批次名：每颗蛋自成一组。"""
    return f"蛋{egg_id}"


def _read_batch(row: Row) -> str:
    return _cell(row, "batch") or _solo_batch(row["egg_id"])


def eggs_without_batch(data: EggDataset) -> list[str]:
    """meta.csv 里没填 batch 的蛋（它们的批次是占位名）。训练前据此报错或提醒，见 train_demo.check_batches。"""
    return [str(e) for e, b in zip(data.egg_ids, data.batch) if str(b) == _solo_batch(e)]


def _read_dev(row: Row) -> float:
    """实测发育指标：必须是 0~1 之间的有限数（nan、inf、超出范围都拒收）。"""
    value = _cell(row, "dev")
    try:
        dev = float(value)
    except ValueError:
        raise ValueError(f"{_where(row)}的 dev {value!r} 不是数字") from None
    if not np.isfinite(dev) or not 0 <= dev <= 1:
        raise ValueError(f"{_where(row)}的 dev 是 {value!r}，只能填 0~1 之间的实测值，没有就留空")
    return dev


def _per_egg(rows: list[Row], egg_ids: tuple[str, ...], read: Callable[[Row], object],
             what: str) -> np.ndarray:
    """每颗蛋一个值（性别、批次），同一颗蛋的各行必须一致。"""
    by_egg: dict[str, object] = {}
    for r in rows:
        value = read(r)
        if by_egg.setdefault(r["egg_id"], value) != value:
            raise ValueError(f"蛋 {r['egg_id']} 的{what}前后不一致")
    return np.array([by_egg[e] for e in egg_ids])


def _dev_labels(rows: list[Row], frames: EggFrames) -> tuple[np.ndarray, np.ndarray]:
    """dev 留空 = 没有实测值（dev_mask 为 False），不用孵化日顶替。"""
    egg_index = {e: i for i, e in enumerate(frames.egg_ids)}
    day_index = {int(d): t for t, d in enumerate(frames.days)}
    dev = np.zeros(frames.mask.shape, dtype=np.float32)
    dev_mask = np.zeros(frames.mask.shape, dtype=bool)
    for r in rows:
        if _cell(r, "dev"):
            i, t = egg_index[r["egg_id"]], day_index[_day(r)]
            dev[i, t], dev_mask[i, t] = _read_dev(r), True
    return dev, dev_mask


# ---------- 对外接口 ----------
def load_real_dataset(csv_path: str) -> EggDataset:
    """训练用：需要 egg_id、day、sex、cube 四列，其余列可选。"""
    csv_file = Path(csv_path)
    rows = read_meta(csv_file, TRAIN_COLUMNS)
    frames = load_frames(csv_file, rows)
    dev, dev_mask = _dev_labels(rows, frames)
    return EggDataset(egg_ids=frames.egg_ids, x=frames.x, spec=frames.spec, mask=frames.mask,
                      sex=_per_egg(rows, frames.egg_ids, _read_sex, "性别"),
                      dev=dev, dev_mask=dev_mask,
                      batch=_per_egg(rows, frames.egg_ids, _read_batch, "批次"),
                      days=frames.days, wavelengths=frames.wavelengths, bg_spec=frames.bg_spec)


def load_unlabeled(csv_path: str) -> EggFrames:
    """判断新蛋用：只需要 egg_id、day、cube 三列，其余列可选。"""
    csv_file = Path(csv_path)
    return load_frames(csv_file, read_meta(csv_file, PREDICT_COLUMNS))
