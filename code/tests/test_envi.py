"""ENVI 格式（.hdr + 数据文件）的读取：三种排列方式、字节序、头部偏移、波长单位、坏文件报错、整条读取链。"""
import logging
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)

import envi  # noqa: E402
import real_data  # noqa: E402
from helpers import ellipse  # noqa: E402

LAYOUT = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}   # (行, 列, 波段) 存盘前的轴顺序


def write_envi(path: Path, cube: np.ndarray, interleave: str = "bil", dtype: str = "<u2", offset: int = 0,
               wavelengths: list[float] | None = None, units: str = "Nanometers", data_suffix: str = ".raw") -> Path:
    """写一个 ENVI 文件对，返回 .hdr 路径。cube 是 (行, 列, 波段)。"""
    codes = {v: k for k, v in envi.DTYPES.items()}
    dt = np.dtype(dtype)
    lines, samples, bands = cube.shape
    header = [f"ENVI", f"samples = {samples}", f"lines = {lines}", f"bands = {bands}",
              f"header offset = {offset}", f"data type = {codes[dt.str[1:]]}", f"interleave = {interleave}",
              f"byte order = {1 if dt.byteorder == '>' else 0}"]
    if wavelengths is not None:
        header += ["wavelength = {\n " + ",\n ".join(map(str, wavelengths)) + "}", f"wavelength units = {units}"]
    hdr = path.with_suffix(".hdr")
    hdr.write_text("\n".join(header) + "\n", encoding="utf-8")
    body = np.ascontiguousarray(cube.transpose(LAYOUT[interleave])).astype(dt).tobytes()
    path.with_suffix(data_suffix).write_bytes(b"\0" * offset + body)
    return hdr


class TempDir(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class TestEnviRead(TempDir):
    cube = np.arange(5 * 4 * 3).reshape(5, 4, 3).astype(np.float32)

    def test_all_interleaves_round_trip(self) -> None:
        for interleave in envi.INTERLEAVES:
            with self.subTest(interleave=interleave):
                hdr = write_envi(self.dir / interleave, self.cube, interleave)
                np.testing.assert_array_equal(envi.load_envi(hdr), self.cube)

    def test_big_endian_float_with_offset_and_img_suffix(self) -> None:
        hdr = write_envi(self.dir / "c", self.cube / 7, "bip", ">f4", offset=128, data_suffix=".img")
        np.testing.assert_allclose(envi.load_envi(hdr), self.cube / 7, rtol=1e-6)

    def test_wavelengths_in_micrometers_become_nm(self) -> None:
        hdr = write_envi(self.dir / "c", self.cube, wavelengths=[0.4, 0.7, 1.0], units="Micrometers")
        np.testing.assert_allclose(envi.envi_wavelengths(hdr), [400, 700, 1000])

    def test_size_mismatch_is_rejected(self) -> None:
        hdr = write_envi(self.dir / "c", self.cube)
        with (self.dir / "c.raw").open("ab") as f:
            f.write(b"\0\0")
        with self.assertRaisesRegex(ValueError, "字节"):
            envi.load_envi(hdr)

    def test_missing_data_file_and_bad_header(self) -> None:
        hdr = write_envi(self.dir / "c", self.cube)
        (self.dir / "c.raw").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "数据文件"):
            envi.load_envi(hdr)
        bad = self.dir / "bad.hdr"
        bad.write_text("samples = 3\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "不是 ENVI"):
            envi.load_envi(bad)

    def test_wavelength_count_must_match_bands(self) -> None:
        hdr = write_envi(self.dir / "c", self.cube, wavelengths=[400.0, 500.0])
        with self.assertRaisesRegex(ValueError, "bands"):
            envi.envi_wavelengths(hdr)


class TestEnviThroughLoader(TempDir):
    """meta.csv 里直接填 .hdr：校正、找蛋、波长都从 ENVI 文件来；改了数据文件缓存要作废。"""

    def setUp(self) -> None:
        super().setUp()
        rng = np.random.default_rng(0)
        bands, shape = 4, (30, 30)
        white = 3000 + rng.normal(0, 5, (10, shape[1], bands))
        dark = 100 + rng.normal(0, 5, (10, shape[1], bands))
        refl = 0.05 + rng.normal(0, 0.005, (*shape, bands))
        refl[ellipse(shape, (15, 15), (8, 10))] = 0.5
        raw = refl * (white.mean(0) - dark.mean(0)) + dark.mean(0)
        wl = [450.0, 550.0, 650.0, 750.0]
        write_envi(self.dir / "egg", raw, wavelengths=wl)
        write_envi(self.dir / "white", white, "bsq", wavelengths=wl)
        write_envi(self.dir / "dark", dark, "bip", wavelengths=wl)
        (self.dir / "meta.csv").write_text(
            "egg_id,day,sex,cube,white,dark,wavelengths\ne1,0,1,egg.hdr,white.hdr,dark.hdr,egg.hdr\n",
            encoding="utf-8")

    def test_loads_and_reads_wavelengths_from_header(self) -> None:
        data = real_data.load_real_dataset(str(self.dir / "meta.csv"))
        self.assertEqual(data.x.shape[2], 4)
        np.testing.assert_allclose(data.wavelengths, [450, 550, 650, 750])
        self.assertAlmostEqual(float(data.spec[0, 0].mean()), 0.5, delta=0.02)

    def test_changing_data_file_invalidates_cache(self) -> None:
        rows = real_data.read_meta(self.dir / "meta.csv", real_data.TRAIN_COLUMNS)
        before = real_data._fingerprint(self.dir / "meta.csv", rows)
        raw = self.dir / "egg.raw"
        raw.write_bytes(raw.read_bytes()[::-1])
        stat = raw.stat()       # 指纹看大小和修改时间；紧接着改写时修改时间可能落在同一个时钟刻度里，这里明确推后 1 秒
        os.utime(raw, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        self.assertNotEqual(before, real_data._fingerprint(self.dir / "meta.csv", rows))


if __name__ == "__main__":
    unittest.main()
