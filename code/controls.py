"""比"打乱标签"更直接的两个负对照，只报告、不参与选方法。

1. 批次可分性：只看决策日的平均光谱，能不能认出蛋来自哪个批次。认得很准说明批次之间差异（光照、种蛋、
   设备状态）很大。交叉验证按批次分组已经防住了"学批次冒充学性别"，但差异大时更要盯紧跨批次的结果。
2. 背景对照：只给模型看蛋外面一圈背景的光谱（不看蛋），用同样的分折预测性别。背景里不该有性别信息；
   判得准就说明有东西跟性别一起变（比如按性别摆放位置、拍摄顺序、光照漂移），正常结果先别信。
"""
import logging

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from baselines import LOGREG_C, snv
from evaluation import CHANCE_ALARM, CHANCE_PASS, chance_status
from synth_data import EggDataset

PROBE_FOLDS = 5                 # 批次可分性：按蛋分层的折数（每个批次至少要有这么多蛋，不够就减少）
MIN_PROBE_FOLDS = 2
STRONG_BATCH_RATIO = 2.0        # 认批次的准确率达到瞎猜水平的这么多倍（且至少高出 STRONG_BATCH_GAP），算"批次差异很大"
STRONG_BATCH_GAP = 0.3

log = logging.getLogger("egg_train")


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(C=LOGREG_C, max_iter=5000))


def batch_probe(data: EggDataset, t: int) -> dict | None:
    """决策日平均光谱认批次的准确率和瞎猜水平（最大批次的占比）。批次少于 2 个或每批蛋太少时返回 None。"""
    sel = data.mask[:, t]
    batches = data.batch[sel].astype(str)
    names, counts = np.unique(batches, return_counts=True)
    n_folds = min(PROBE_FOLDS, int(counts.min())) if len(names) >= 2 else 0
    if n_folds < MIN_PROBE_FOLDS:
        return None
    X, correct = snv(data.spec[sel, t]), np.zeros(len(batches), dtype=bool)
    for train, test in StratifiedKFold(n_folds, shuffle=True, random_state=0).split(X, batches):
        correct[test] = _model().fit(X[train], batches[train]).predict(X[test]) == batches[test]
    acc, chance = float(correct.mean()), float(counts.max() / counts.sum())
    strong = acc >= max(STRONG_BATCH_RATIO * chance, chance + STRONG_BATCH_GAP)
    return {"批次数": len(names), "蛋数": int(sel.sum()), "认批次准确率": round(acc, 4),
            "瞎猜水平": round(chance, 4), "批次差异很大": bool(strong)}


def background_control(data: EggDataset, fold_of_egg: np.ndarray, t: int, n_train: float,
                       n_test: float) -> dict | None:
    """只用背景光谱、按交叉验证同样的分折预测性别，每折准确率应接近 0.5。没有背景光谱时返回 None。"""
    if data.bg_spec is None:
        return None
    usable = data.mask[:, t] & np.isfinite(data.bg_spec[:, t]).all(axis=1)
    folds = sorted(set(fold_of_egg[usable]))
    fold_acc = []
    for k in folds:
        train, test = usable & (fold_of_egg != k), usable & (fold_of_egg == k)
        if len(np.unique(data.sex[train])) < 2 or not test.any():
            continue
        model = _model().fit(data.bg_spec[train, t], data.sex[train])     # 不做 SNV：整体亮度差异也算可疑信号
        fold_acc.append(float((model.predict(data.bg_spec[test, t]) == data.sex[test]).mean()))
    if len(fold_acc) < MIN_PROBE_FOLDS:
        return None
    status, test = chance_status(np.array(fold_acc), n_train, n_test)
    return {"各折准确率": [round(a, 4) for a in fold_acc], "平均准确率": round(float(np.mean(fold_acc)), 4),
            "单侧p": None if np.isnan(test.p) else round(float(test.p), 4), "结果": status}


def run_controls(data: EggDataset, fold_of_egg: np.ndarray, t: int, n_train: float, n_test: float) -> dict:
    """跑两个负对照并写日志，返回写进 config.json 的内容。"""
    day = int(data.days[t])
    log.info(f"\n负对照（第 {day} 天，只报告，不参与选方法）")
    probe = batch_probe(data, t)
    if probe is None:
        log.info("  批次可分性：批次少于 2 个或每批蛋太少，没做")
    else:
        verdict = ("批次之间差异很大：模型可能更容易学到批次而不是性别，要重点看按批次分组的结果和独立测试集"
                   if probe["批次差异很大"] else "批次之间差异不算大")
        log.info(f"  批次可分性：只看平均光谱认批次，准确率 {probe['认批次准确率']:.3f}（瞎猜 {probe['瞎猜水平']:.3f}，"
                 f"{probe['批次数']} 个批次）→ {verdict}")
    bg = background_control(data, fold_of_egg, t, n_train, n_test)
    if bg is None:
        log.info("  背景对照：没有背景光谱（模拟数据，或 ROI_MODE=\"none\"），没做")
    elif bg["结果"] == CHANCE_PASS:
        log.info(f"  背景对照：只看蛋外面的背景猜性别，准确率 {bg['平均准确率']:.3f} → 通过（背景里没有性别信息）")
    else:
        level = log.error if bg["结果"] == CHANCE_ALARM else log.warning
        level(f"  背景对照：只看蛋外面的背景猜性别，准确率 {bg['平均准确率']:.3f} → {bg['结果']}。"
              "背景里不该有性别信息，有东西跟性别一起变（摆放位置、拍摄顺序、光照漂移？），正常结果先别信")
    return {"批次可分性": probe, "背景对照": bg}
