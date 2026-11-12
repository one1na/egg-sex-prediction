"""读取 ENVI 标准格式的高光谱文件：一个文本头文件 .hdr + 一个二进制数据文件（.raw / .img / .dat / .bin 或没有扩展名）。

线扫高光谱相机大多导出这种格式。只做读取，返回 (行, 列, 波段) 的 float32 数组；波长从头文件的 wavelength 字段读。
头文件里缺关键字段、数据文件大小对不上、格式不支持，都直接报错。
"""
import re
from pathlib import Path

import numpy as np

DATA_SUFFIXES = (".raw", ".img", ".dat", ".bin", "")   # 没写 data file 时，按这些扩展名找数据文件
DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}
INTERLEAVES = ("bsq", "bil", "bip")
MICROMETER_UNITS = ("micrometers", "micrometer", "microns", "um", "µm")
UM_TO_NM = 1000.0
REQUIRED = ("samples", "lines", "bands", "data type", "interleave")


def read_header(hdr: Path) -> dict[str, str]:
    """把 .hdr 读成 {小写字段名: 原文}，花括号里的多行内容合成一行。"""
    if not hdr.exists():
        raise FileNotFoundError(f"找不到 ENVI 头文件：{hdr}")
    text = hdr.read_text(encoding="utf-8", errors="replace")
    if not text.lstrip().upper().startswith("ENVI"):
        raise ValueError(f"{hdr} 不是 ENVI 头文件（第一行应该是 ENVI）")
    fields = {}
    for match in re.finditer(r"^\s*([^=\n]+?)\s*=\s*(\{[^}]*\}|[^\n]*)", text, flags=re.MULTILINE):
        fields[match.group(1).strip().lower()] = " ".join(match.group(2).strip().strip("{}").split())
    missing = [k for k in REQUIRED if k not in fields]
    if missing:
        raise ValueError(f"{hdr} 缺少字段 {missing}")
    return fields


def _int(fields: dict[str, str], key: str, hdr: Path, default: int | None = None) -> int:
    value = fields.get(key)
    if value is None and default is not None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{hdr} 的 {key} = {value!r} 不是整数") from None


def data_file(hdr: Path, fields: dict[str, str] | None = None) -> Path:
    """头文件对应的数据文件：优先用 data file 字段，否则在同一文件夹按同名不同扩展名找。"""
    fields = read_header(hdr) if fields is None else fields
    named = fields.get("data file")
    candidates = [hdr.parent / Path(named).name] if named else [hdr.with_suffix(s) for s in DATA_SUFFIXES]
    for path in candidates:
        if path.exists() and path != hdr:
            return path
    raise FileNotFoundError(f"找不到 {hdr.name} 对应的数据文件（找过 {[p.name for p in candidates]}）")


def _dtype(fields: dict[str, str], hdr: Path) -> np.dtype:
    code = _int(fields, "data type", hdr)
    if code not in DTYPES:
        raise ValueError(f"{hdr} 的 data type = {code} 不支持（支持 {sorted(DTYPES)}）")
    order = ">" if _int(fields, "byte order", hdr, default=0) == 1 else "<"
    return np.dtype(order + DTYPES[code])


def load_envi(hdr: Path) -> np.ndarray:
    """读出 (行, 列, 波段) 的 float32 数组。"""
    fields = read_header(hdr)
    lines, samples, bands = (_int(fields, k, hdr) for k in ("lines", "samples", "bands"))
    interleave = fields["interleave"].lower()
    if interleave not in INTERLEAVES:
        raise ValueError(f"{hdr} 的 interleave = {interleave!r} 不支持（支持 {INTERLEAVES}）")
    dtype, offset = _dtype(fields, hdr), _int(fields, "header offset", hdr, default=0)
    path = data_file(hdr, fields)
    expected = offset + lines * samples * bands * dtype.itemsize
    if path.stat().st_size != expected:
        raise ValueError(f"{path.name} 有 {path.stat().st_size} 字节，按头文件（{lines} 行 × {samples} 列 × "
                         f"{bands} 波段，{dtype}，偏移 {offset}）应该是 {expected} 字节")
    flat = np.fromfile(path, dtype=dtype, offset=offset)
    shapes = {"bsq": ((bands, lines, samples), (1, 2, 0)), "bil": ((lines, bands, samples), (0, 2, 1)),
              "bip": ((lines, samples, bands), (0, 1, 2))}
    shape, axes = shapes[interleave]
    return np.ascontiguousarray(flat.reshape(shape).transpose(axes), dtype=np.float32)


def envi_wavelengths(hdr: Path) -> np.ndarray:
    """头文件里的波长（统一成 nm）。没有 wavelength 字段或个数和波段数不一致就报错。"""
    fields = read_header(hdr)
    if "wavelength" not in fields:
        raise ValueError(f"{hdr} 里没有 wavelength 字段，请另外提供波长文件")
    values = np.array([float(v) for v in fields["wavelength"].replace(",", " ").split()])
    if len(values) != _int(fields, "bands", hdr):
        raise ValueError(f"{hdr} 的 wavelength 有 {len(values)} 个数，bands = {fields['bands']}")
    units = fields.get("wavelength units", "nanometers").lower()
    return values * UM_TO_NM if units in MICROMETER_UNITS else values
