"""每次训练一个文件夹 output/runs/<时间>_<类型>/：配置、分折、逐日结果、预测明细、日志、模型。"""
import csv
import json
import logging
import platform
import sys
from datetime import datetime
from pathlib import Path

from paths import RUNS_DIR

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
            "scikit-learn": sklearn.__version__, "torch": torch.__version__}


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
