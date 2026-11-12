"""每次训练一个文件夹 output/runs/<时间>_<类型>/：配置、分折、逐日结果、预测明细、日志、模型。

复现记录（config.json 的"复现"一项）：命令、代码版本（git 提交号、有没有没提交的改动、代码文件的指纹）、
数据指纹（模型实际看到的数组）、依赖版本是否和 requirements.txt 锁定的一致。代码有没提交的改动时，
改动另存为 code_changes.patch，这样即使没提交，也能还原出跑这次结果的代码。
"""
import csv
import hashlib
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from paths import ROOT, RUNS_DIR

log = logging.getLogger("egg_train")
CODE_DIR = ROOT / "code"
REQUIREMENTS = CODE_DIR / "requirements.txt"
PATCH_NAME = "code_changes.patch"
GIT_TIMEOUT = 20                # 秒

EXIT_SELF_CHECK_FAILED = 2      # 退出码：流程自检或背景负对照没通过，模型没有保存（1 留给程序出错）
EXIT_HOLDOUT_USED = 3           # 退出码：独立测试集已经被查看过，这次的正式评估被拦下，什么都没训练
MAX_SAME_SECOND_RUNS = 100      # 同一秒内最多开多少个运行文件夹（加 _2、_3… 后缀区分）


def new_run_dir(kind: str) -> Path:
    """新建运行文件夹。同一秒启动了两次时，后一个加 _2、_3… 后缀，绝不覆盖已有的记录。"""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    base = f"{datetime.now():%Y-%m-%d_%H%M%S}_{kind}"
    for n in range(1, MAX_SAME_SECOND_RUNS + 1):
        path = RUNS_DIR / (base if n == 1 else f"{base}_{n}")
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError(f"{RUNS_DIR} 里以 {base} 开头的文件夹已经有 {MAX_SAME_SECOND_RUNS} 个，请检查是不是有程序在反复启动")


def attach_log(run_dir: Path) -> None:
    """之后打印的内容同时写进 run_dir/log.txt。"""
    handler = logging.FileHandler(run_dir / "log.txt", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger().addHandler(handler)


def environment() -> dict[str, str]:
    """软件版本：要复现结果，得用同样的版本。"""
    import numpy
    import scipy
    import sklearn
    import torch
    return {"python": sys.version.split()[0], "系统": platform.platform(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__,
            "torch线程数": str(torch.get_num_threads()),
            "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无（只用 CPU）"}


# ---------- 复现 ----------
def set_determinism(seed: int) -> None:
    """固定所有随机源，并让 PyTorch 尽量只用结果确定的算法（没有确定性实现的算子只警告、不报错）。
    同一台机器、同样版本时结果应该逐位相同；换 CPU 型号、线程数或 GPU，浮点累加顺序可能不同，结果会有极小差别。"""
    import torch
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")     # GPU 上矩阵乘法确定性需要的设置
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _git(*args: str) -> str | None:
    """在项目根目录跑 git；没装 git、不是仓库或出错时返回 None。"""
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=false", "-C", str(ROOT), *args],
                             capture_output=True, timeout=GIT_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.decode("utf-8", errors="replace") if out.returncode == 0 else None


def code_hash(code_dir: Path = CODE_DIR) -> str:
    """code 文件夹里所有 .py（含子文件夹）的内容指纹：不依赖 git，代码哪怕改一个字符都会变。"""
    h = hashlib.sha256()
    for path in sorted(code_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        h.update(path.relative_to(code_dir).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()[:16]


def code_version(run_dir: Path | None = None) -> dict:
    """git 提交号和有没有没提交的改动；有改动且给了 run_dir 时，把改动另存为 code_changes.patch。"""
    info: dict = {"代码指纹": code_hash()}
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        info["git"] = "不可用（没装 git 或不是 git 仓库），只能靠代码指纹核对"
        return info
    status = _git("status", "--porcelain", "--untracked-files=all") or ""
    changed = [line[3:] for line in status.splitlines() if line.strip()]
    info.update({"git提交": commit.strip(), "有未提交的改动": bool(changed)})
    if changed:
        info["改动的文件"] = changed
        diff = _git("diff", "HEAD", "--binary")
        if run_dir is not None and diff:
            (run_dir / PATCH_NAME).write_text(diff, encoding="utf-8")
            info["改动另存为"] = PATCH_NAME
        untracked = [line[3:] for line in status.splitlines() if line.startswith("??")]
        if untracked:
            info["说明"] = "新建、还没加进 git 的文件不在 patch 里，靠代码指纹核对"
    return info


def _update(h: "hashlib._Hash", name: str, value: object) -> None:
    """把一个字段（数组、字符串序列或 None）连同名字、类型、形状写进指纹。"""
    if value is None:
        h.update(f"{name}:None;".encode())
        return
    if isinstance(value, (tuple, list)) or (isinstance(value, np.ndarray) and value.dtype.kind in "OUS"):
        items = [str(v) for v in np.asarray(value, dtype=object).ravel()]
        h.update(f"{name}:str{len(items)};".encode() + "\x1f".join(items).encode() + b";")
        return
    arr = np.ascontiguousarray(value)
    h.update(f"{name}:{arr.dtype.str}{arr.shape};".encode() + arr.tobytes() + b";")


IMAGE_FIELDS = ("egg_ids", "days", "x", "spec", "mask", "wavelengths", "bg_spec")
LABEL_FIELDS = ("egg_ids", "sex", "dev", "dev_mask", "batch")


def data_fingerprint(data: object, csv_path: str | None = None) -> dict:
    """模型实际看到的数据的指纹：图像（预处理之后）和标签分开算，只改了标签或只换了图像都看得出来。
    给了 meta.csv 时另记它本身的指纹。"""
    out = {}
    for label, fields in (("图像", IMAGE_FIELDS), ("标签", LABEL_FIELDS)):
        h = hashlib.sha256()
        for name in fields:
            _update(h, name, getattr(data, name, None))
        out[label] = h.hexdigest()[:16]
    out["整体"] = hashlib.sha256((out["图像"] + out["标签"]).encode()).hexdigest()[:16]
    if csv_path:
        out["数据表"] = hashlib.sha256(Path(csv_path).read_bytes()).hexdigest()[:16]
    return out


def pinned_versions(path: Path = REQUIREMENTS) -> dict[str, str]:
    """requirements.txt 里用 == 锁定的版本。"""
    pins = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def dependency_check(path: Path = REQUIREMENTS) -> dict:
    """装的版本和锁定的对不对得上（torch 的 +cpu 这类后缀不算差别）。对不上会在日志里提醒，但不拦。"""
    if not path.exists():
        return {"一致": None, "说明": f"找不到 {path.name}"}
    mismatches = []
    for name, want in pinned_versions(path).items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = "没装"
        if have.split("+")[0] != want.split("+")[0]:
            mismatches.append(f"{name}：锁定 {want}，实际 {have}")
    if mismatches:
        log.warning("提醒：软件版本和 requirements.txt 锁定的不一样（" + "；".join(mismatches)
                    + "），结果可能和别人复现的有出入")
    return {"一致": not mismatches, "不一致": mismatches}


def reproducibility(run_dir: Path, data: object = None, csv_path: str | None = None) -> dict:
    """写进 config.json 的"复现"一项。"""
    info = {"命令": [Path(sys.argv[0]).name, *sys.argv[1:]], "代码版本": code_version(run_dir),
            "依赖版本": dependency_check(), "确定性": "已固定随机种子，PyTorch 只用确定性算法（没有的只警告）"}
    if data is not None:
        info["数据指纹"] = data_fingerprint(data, csv_path)
    return info


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    """utf-8-sig 编码，Excel 直接打开不乱码。"""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
