"""真实数据基线：用伊利诺伊大学公开的平均光谱，判断蛋是否受精。

运行：双击 3-真实数据基线.bat，或 py -3.11 baseline_fertility.py
- 输入：每颗蛋 300 个波段的平均光谱（374~1015 nm）
- 标签：受精 / 未受精
- 验证：按采集日期分组的 5 折交叉验证（测试的蛋来自训练时没见过的日子）
- 结果：窗口里显示，并保存到 output/baseline_fertility.csv
"""
import csv
import logging

import numpy as np
from scipy.signal import savgol_filter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from baselines import PLSDA, snv
from paths import BASELINE_RESULT_CSV, FERTILITY_CSV
from run_record import save_csv
from synth_data import EggDataset

N_FOLDS = 5
SEED = 42
SG_WINDOW = 11          # Savitzky-Golay 平滑窗口（波段数）
SG_ORDER = 2
TOP_BANDS = 10          # 显示最重要的前几个波段

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("baseline")


def _read_fertility() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 样本编号、光谱 X、标签 y（1=受精）、分组 groups（采集日期）、波长。"""
    if not FERTILITY_CSV.exists():
        raise FileNotFoundError(f"找不到 {FERTILITY_CSV}，请先下载伊利诺伊受精数据集的 CSV")
    with FERTILITY_CSV.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    band_cols = [i for i, c in enumerate(header) if c.replace(".", "", 1).isdigit()]
    status_col = header.index("Fertility status")
    id_col = header.index("Sample ID")

    ids = [r[id_col].strip("'") for r in body]
    X = np.array([[float(r[i]) for i in band_cols] for r in body], dtype=np.float64)
    y = np.array([1 if r[status_col] == "Fertile" else 0 for r in body])
    # 编号形如 Egg-F-03Dec-0096：第 3 段是采集日期。编号里的 F/IF 就是答案，绝不能当输入
    groups = np.array([i.split("-")[2] for i in ids])
    wavelengths = np.array([float(header[i]) for i in band_cols])
    return ids, X, y, groups, wavelengths


def load_fertility() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回 光谱 X、标签 y（1=受精）、分组 groups（采集日期）、波长。"""
    return _read_fertility()[1:]


def as_egg_dataset() -> EggDataset:
    """把受精数据装成训练流程用的格式，好让整条流水线原样跑在真实光谱上（train_demo.py --fertility-check）。

    注意三点：标签放在 sex 一栏里，但它是受精状态（1=受精），不是性别；每颗蛋只有一条平均光谱，
    所以"图像"是 1×1 像素、只有一个时间点（记作第 0 天）；批次 = 采集日期。光谱是原始像素强度，不是反射率。
    """
    ids, X, y, groups, wavelengths = _read_fertility()
    if len(set(ids)) != len(ids):
        raise ValueError(f"{FERTILITY_CSV} 里有重复的 Sample ID")
    n = len(ids)
    spec = X.astype(np.float32)[:, None, :]
    return EggDataset(egg_ids=tuple(ids), x=spec[:, :, :, None, None], spec=spec, mask=np.ones((n, 1), dtype=bool),
                      sex=y, dev=np.zeros((n, 1), dtype=np.float32), dev_mask=np.zeros((n, 1), dtype=bool),
                      batch=groups, days=np.array([0.0], dtype=np.float32),
                      wavelengths=wavelengths.astype(np.float32))


def preprocess(X: np.ndarray) -> np.ndarray:
    """SNV（消除整体亮度差异）+ Savitzky-Golay 平滑。每条光谱单独处理，不会泄漏。"""
    return savgol_filter(snv(X), SG_WINDOW, SG_ORDER, axis=1)


def build_models() -> dict[str, object]:
    return {
        "PLS-DA": make_pipeline(StandardScaler(), PLSDA(n_components=10)),
        "逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1)),
        "SVM": make_pipeline(StandardScaler(), SVC(C=1.0)),
        "随机森林": RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
    }


def evaluate(model: object, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> list[float]:
    scores = []
    for train, test in GroupKFold(n_splits=N_FOLDS).split(X, y, groups):
        model.fit(X[train], y[train])
        scores.append(float((model.predict(X[test]) == y[test]).mean()))
    return scores


def top_bands(X: np.ndarray, y: np.ndarray, wavelengths: np.ndarray) -> list[tuple[float, float]]:
    forest = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)
    order = np.argsort(forest.feature_importances_)[::-1][:TOP_BANDS]
    return [(float(wavelengths[i]), float(forest.feature_importances_[i])) for i in order]


def main() -> None:
    X_raw, y, groups, wavelengths = load_fertility()
    X = preprocess(X_raw)
    log.info(f"数据：{len(y)} 颗蛋（受精 {y.sum()}，未受精 {len(y) - y.sum()}），"
             f"{X.shape[1]} 个波段，{len(set(groups))} 个采集日期")
    log.info(f"验证方式：按采集日期分 {N_FOLDS} 折，测试的蛋来自训练时没见过的日子\n")

    log.info("平均准确率  波动(标准差)   各折准确率                            方法")
    results = []
    for name, model in build_models().items():
        scores = evaluate(model, X, y, groups)
        mean, std = float(np.mean(scores)), float(np.std(scores))
        folds = " ".join(f"{s:.3f}" for s in scores)
        log.info(f"{mean:>8.3f}    {std:>8.3f}       {folds}   {name}")
        results.append({"方法": name, "平均准确率": f"{mean:.4f}", "标准差": f"{std:.4f}",
                        "各折准确率": folds})

    log.info(f"\n随机森林认为最重要的 {TOP_BANDS} 个波段：")
    for wl, imp in top_bands(X, y, wavelengths):
        log.info(f"  {wl:7.1f} nm   重要性 {imp:.4f}")

    save_csv(BASELINE_RESULT_CSV, results)
    log.info(f"\n结果已保存到 {BASELINE_RESULT_CSV}")
    log.info("提示：瞎猜的准确率约 0.5。")


if __name__ == "__main__":
    main()
