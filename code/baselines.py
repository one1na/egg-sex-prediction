"""传统光谱方法，作为深度模型的对照：每个孵化日用蛋区域的平均光谱单独训练一个分类器。

所有方法都输出"判为雄"的分数，大于 0.5 判雄，方便和深度模型放在同一张表里比较。
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

PLS_COMPONENTS = 10
LOGREG_C = 0.1
BASELINE_NAMES = ("PLS-DA", "逻辑回归")
MATCHED_SUFFIX = "·等样本"      # 同样本量对照：基线只用深度模型拟合用的那部分蛋训练（不含早停验证集）


def matched_name(name: str) -> str:
    """基线的"同样本量对照"版本的名字。"""
    return name + MATCHED_SUFFIX


MATCHED_NAMES = tuple(matched_name(name) for name in BASELINE_NAMES)


class PLSDA(BaseEstimator, ClassifierMixin):
    """PLS-DA：光谱领域最常用的分类方法。回归值大于 0.5 判为 1。"""

    def __init__(self, n_components: int = PLS_COMPONENTS) -> None:
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PLSDA":
        n = min(self.n_components, X.shape[1], len(X) - 1)   # 样本少时自动减少成分数
        self.pls_ = PLSRegression(n_components=n).fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.pls_.predict(X).ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 0.5).astype(int)


def snv(spectra: np.ndarray) -> np.ndarray:
    """SNV：每条光谱减去自己的均值、除以自己的标准差，消除整体亮度差异。"""
    centered = spectra - spectra.mean(axis=-1, keepdims=True)
    return centered / (spectra.std(axis=-1, keepdims=True) + 1e-12)


def _models() -> dict[str, Pipeline]:
    return {"PLS-DA": make_pipeline(StandardScaler(), PLSDA()),
            "逻辑回归": make_pipeline(StandardScaler(), LogisticRegression(C=LOGREG_C, max_iter=5000))}


def _male_score(model: Pipeline, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def per_day_baselines(spec: np.ndarray, mask: np.ndarray, sex: np.ndarray,
                      train_idx: np.ndarray, test_idx: np.ndarray, use_snv: bool = True) -> dict[str, np.ndarray]:
    """每个孵化日各训练一次，返回 {方法名: (测试蛋数, T) 判为雄的分数}，没数据的帧是 nan。
    use_snv=False 只给消融实验用（看 SNV 预处理的贡献）；正式训练和部署一律做 SNV。"""
    X = snv(spec) if use_snv else spec
    scores = {name: np.full((len(test_idx), spec.shape[1]), np.nan) for name in BASELINE_NAMES}
    for t in range(spec.shape[1]):
        train_t = train_idx[mask[train_idx, t]]
        test_t = mask[test_idx, t]
        if len(np.unique(sex[train_t])) < 2 or not test_t.any():
            continue
        for name, model in _models().items():
            model.fit(X[train_t, t], sex[train_t])
            scores[name][test_t, t] = _male_score(model, X[test_idx[test_t], t])
    return scores


def fit_final_baselines(spec: np.ndarray, mask: np.ndarray, sex: np.ndarray,
                        name: str) -> dict[int, Pipeline]:
    """部署用：用全部蛋给每个孵化日训练一个 name 方法的模型，返回 {孵化日序号: 模型}。"""
    X = snv(spec)
    models = {}
    for t in range(spec.shape[1]):
        idx = np.flatnonzero(mask[:, t])
        if len(np.unique(sex[idx])) == 2:
            models[t] = _models()[name].fit(X[idx, t], sex[idx])
    return models


def baseline_scores(models: dict[float, Pipeline], spec: np.ndarray, mask: np.ndarray,
                    days: list[float]) -> np.ndarray:
    """用 {孵化日: 模型} 给新蛋打分，返回 (N, T) 判为雄的分数；没模型或没数据的帧是 nan。"""
    X = snv(spec)
    scores = np.full(mask.shape, np.nan)
    for t, day in enumerate(days):
        model, sel = models.get(float(day)), mask[:, t]
        if model is not None and sel.any():
            scores[sel, t] = _male_score(model, X[sel, t])
    return scores


def day_only_dev(days: np.ndarray) -> np.ndarray:
    """发育的对照：只看孵化日，发育 = 孵化日 / 最大孵化日。"""
    return days / days.max()
