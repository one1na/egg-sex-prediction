"""验证方式和指标：独立批次测试集、按批次分组的交叉验证、逐日指标、方法比较、"应该是瞎猜"的检查。

覆盖率：一种方法在某天真正给出了判断的测试蛋占全部测试蛋的比例。
单帧方法（深度-单帧、基线）当天没有图像就判断不了；深度-累计可以用之前几天的图像判断。
准确率、AUC 只在给出判断的蛋上算，所以要和覆盖率一起看。
决策效用：正确判断的蛋数 / 全部蛋数，判断不了的蛋算"没判对"。它不受覆盖率不同的影响，是方法比较的主指标。
"""
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

INNER_VAL_FRACTION = 0.2        # 训练部分里拿多少做早停验证
MIN_GROUPS_FOR_GROUPED_VAL = 4  # 训练部分至少有这么多批次，早停验证才按批次分
HOLDOUT_FRACTION = 0.2          # 留作独立测试的批次比例
MIN_GROUPS_FOR_HOLDOUT = 6      # 至少这么多批次才留独立测试集（否则交叉验证都不够分）
CHANCE_ALPHA = 0.01             # 应该是瞎猜时，"准确率高于 0.5"的单侧 p 值低于它就报警
CHANCE_MARGIN = 0.15            # p 值算不出来时，平均准确率比 0.5 高出这么多也报警（各折全对等情况）
CHANCE_PASS, CHANCE_ALARM, CHANCE_UNKNOWN = "通过", "报警", "无法确认"
ALPHA = 0.05                    # 深度模型 vs 基线的显著性水平（单侧）
METRIC_KEYS = ("acc", "bal_acc", "auc", "n", "coverage", "utility")
PRIMARY_METRIC = "utility"      # 事先规定：深度模型 vs 基线、挑最好的基线，都只看决策日的决策效用
MIN_HOLDOUT_BATCHES = 5         # 独立测试集少于这么多批次，结果只能当参考
WILSON_LABEL = "按鸡蛋计算的描述性区间（批次内不独立，不能当作对新批次的置信区间）"
MIN_BATCHES_FOR_INTERVAL = 3    # 以批次为单位的区间至少要这么多批次，否则只报平均和最低最高
BATCH_INTERVAL_LABEL = ("以批次为单位的 95% t 区间（每个批次算一个值；批次少时区间很宽，"
                        "交叉验证里各批次共用部分训练数据、并不完全独立，区间只会偏窄不会偏宽）")

Split = tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True)
class Paired:
    """a 比 b 各折平均高多少、95% 置信区间、单侧 p 值。无法估计时为 nan。"""
    diff: float
    ci_low: float
    ci_high: float
    p: float


# ---------- 划分 ----------
def holdout_split(sex: np.ndarray, groups: np.ndarray, seed: int) -> Split | None:
    """留出约 HOLDOUT_FRACTION 的批次作独立测试集，开发阶段完全不碰。批次太少时返回 None。"""
    n_groups = len(np.unique(groups))
    if n_groups < MIN_GROUPS_FOR_HOLDOUT:
        return None
    cv = StratifiedGroupKFold(n_splits=round(1 / HOLDOUT_FRACTION), shuffle=True, random_state=seed)
    return next(cv.split(np.zeros(len(sex)), sex, groups))


def outer_folds(sex: np.ndarray, groups: np.ndarray, n_folds: int, seed: int) -> list[Split]:
    """外层交叉验证：同一批次（没有批次信息就是同一颗蛋）只落在训练或测试的一边，各折雌雄比例尽量接近。"""
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("至少要有 2 个批次（或 2 颗蛋）才能做交叉验证")
    cv = StratifiedGroupKFold(n_splits=min(n_folds, n_groups), shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(sex)), sex, groups))


def inner_split(train_idx: np.ndarray, sex: np.ndarray, groups: np.ndarray,
                seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """从训练部分分出早停验证集。批次够多就按批次分，否则按蛋随机分。返回 (拟合, 验证, 是否按批次)。"""
    g = groups[train_idx]
    n_groups = len(np.unique(g))
    if n_groups >= MIN_GROUPS_FOR_GROUPED_VAL:
        n_splits = min(round(1 / INNER_VAL_FRACTION), n_groups)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fit, val = next(cv.split(train_idx, sex[train_idx], g))
        return train_idx[fit], train_idx[val], True
    cv = StratifiedShuffleSplit(n_splits=1, test_size=INNER_VAL_FRACTION, random_state=seed)
    fit, val = next(cv.split(train_idx, sex[train_idx]))
    return train_idx[fit], train_idx[val], False


# ---------- 指标 ----------
def day_metrics(scores_t: np.ndarray, sex: np.ndarray) -> dict[str, float]:
    """一组蛋在某一天的指标。scores_t 是 nan 的蛋表示没给出判断。分数大于 0.5 判雄。"""
    done = ~np.isnan(scores_t)
    res = {"acc": np.nan, "bal_acc": np.nan, "auc": np.nan, "n": float(done.sum()),
           "coverage": float(done.mean()) if len(done) else np.nan,
           "utility": float((done & ((scores_t > 0.5).astype(int) == sex)).mean()) if len(done) else np.nan}
    if not done.any():
        return res
    y, s = sex[done], scores_t[done]
    pred = (s > 0.5).astype(int)
    res["acc"] = float((pred == y).mean())
    if len(np.unique(y)) == 2:
        res["bal_acc"] = float(balanced_accuracy_score(y, pred))
        res["auc"] = float(roc_auc_score(y, s))
    return res


def fold_day_metrics(scores: np.ndarray, sex: np.ndarray, fold_of_egg: np.ndarray,
                     n_folds: int) -> dict[str, np.ndarray]:
    """每折、每个孵化日的各项指标，形状 (折数, T)。覆盖率的分母是这一折的全部测试蛋。"""
    res = {key: np.full((n_folds, scores.shape[1]), np.nan) for key in METRIC_KEYS}
    for k in range(n_folds):
        in_fold = fold_of_egg == k
        for t in range(scores.shape[1]):
            for key, value in day_metrics(scores[in_fold, t], sex[in_fold]).items():
                res[key][k, t] = value
    return res


def pooled_utility(scores_t: np.ndarray, sex: np.ndarray) -> float:
    """把所有折的折外预测合在一起、按蛋算的决策效用。各折蛋数不同时，它和"各折平均"会有出入，两个都报告。"""
    return day_metrics(scores_t, sex)["utility"]


@dataclass(frozen=True)
class CommonAccuracy:
    """两种方法都给出了判断的蛋上的准确率，每折一个值（没有共同可判断的蛋时为 nan）。"""
    n_common: np.ndarray
    acc_a: np.ndarray
    acc_b: np.ndarray


def common_accuracy(a: np.ndarray, b: np.ndarray, sex: np.ndarray, fold_of_egg: np.ndarray,
                    n_folds: int) -> CommonAccuracy:
    """a、b 是两种方法对每颗蛋的分数 (N,)，nan = 判断不了。只在两者都能判断的蛋上比准确率。"""
    n = np.zeros(n_folds)
    acc = {key: np.full(n_folds, np.nan) for key in "ab"}
    for k in range(n_folds):
        both = (fold_of_egg == k) & ~np.isnan(a) & ~np.isnan(b)
        n[k] = both.sum()
        if both.any():
            acc["a"][k] = float(((a[both] > 0.5).astype(int) == sex[both]).mean())
            acc["b"][k] = float(((b[both] > 0.5).astype(int) == sex[both]).mean())
    return CommonAccuracy(n, acc["a"], acc["b"])


@dataclass(frozen=True)
class ScopeStats:
    """独立测试集某个范围（全部 / 某个批次）在决策日的成绩。"""
    scope: str
    n_eggs: int
    judged: int
    correct: int
    bal_acc: float
    auc: float

    @property
    def coverage(self) -> float:
        return self.judged / self.n_eggs if self.n_eggs else float("nan")

    @property
    def accuracy(self) -> float:
        return self.correct / self.judged if self.judged else float("nan")

    @property
    def utility(self) -> float:
        return self.correct / self.n_eggs if self.n_eggs else float("nan")


def _scope(name: str, scores_t: np.ndarray, sex: np.ndarray) -> ScopeStats:
    m = day_metrics(scores_t, sex)
    return ScopeStats(name, len(sex), int(m["n"]), round(m["acc"] * m["n"]) if m["n"] else 0, m["bal_acc"], m["auc"])


def holdout_table(scores_t: np.ndarray, sex: np.ndarray, batch: np.ndarray) -> list[ScopeStats]:
    """第一行是全部，后面每个测试批次一行。没有批次信息（每颗蛋自成一批）时只有第一行。"""
    rows = [_scope("全部", scores_t, sex)]
    batches = sorted(map(str, np.unique(batch)))
    if len(batches) < len(sex):
        rows += [_scope(b, scores_t[batch.astype(str) == b], sex[batch.astype(str) == b]) for b in batches]
    return rows


def wilson_ci(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """准确率的 95% 置信区间（Wilson 方法，样本少时比"均值 ± 1.96 标准误"可靠）。"""
    if n == 0:
        return float("nan"), float("nan")
    p = correct / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(center - half), float(center + half)


# ---------- 以批次为单位的区间 ----------
@dataclass(frozen=True)
class GroupInterval:
    """一组"每个批次一个值"的汇总：批次数、平均、95% t 区间、最低、最高。算不出来的是 nan。"""
    n: int
    mean: float
    ci_low: float
    ci_high: float
    low: float
    high: float


def group_interval(values: np.ndarray) -> GroupInterval:
    """以批次为单位的 95% t 区间。批次少于 MIN_BATCHES_FOR_INTERVAL 个时不给区间，只给平均和最低最高。"""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    nan = float("nan")
    if len(v) == 0:
        return GroupInterval(0, nan, nan, nan, nan, nan)
    mean, low, high = float(v.mean()), float(v.min()), float(v.max())
    if len(v) < MIN_BATCHES_FOR_INTERVAL:
        return GroupInterval(len(v), mean, nan, nan, low, high)
    half = float(stats.t.ppf(0.975, df=len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    return GroupInterval(len(v), mean, mean - half, mean + half, low, high)


def batch_utilities(scores_t: np.ndarray, sex: np.ndarray, batch: np.ndarray, names: np.ndarray) -> np.ndarray:
    """每个批次在某一天的决策效用（正确判断的蛋 / 这个批次的全部蛋），顺序同 names。"""
    b = batch.astype(str)
    return np.array([day_metrics(scores_t[b == name], sex[b == name])["utility"] for name in names])


# ---------- 比较 ----------
def compare_paired(a: np.ndarray, b: np.ndarray, n_train: float, n_test: float) -> Paired:
    """a 比 b 各折平均高多少。

    配对 t 检验，方差按 Nadeau-Bengio 方法放大：交叉验证各折的训练集互相重叠，
    直接用普通 t 检验会把差异夸大。
    折数不到 2，或各折差值完全相同（方差为 0，常见于数据太少），p 值和区间无法可靠估计，记为 nan。
    """
    ok = ~(np.isnan(a) | np.isnan(b))
    diff = a[ok] - b[ok]
    nan = float("nan")
    if len(diff) < 2:
        return Paired(float(diff.mean()) if len(diff) else nan, nan, nan, nan)
    mean, var = float(diff.mean()), float(diff.var(ddof=1))
    if var == 0:
        return Paired(mean, nan, nan, nan)
    se = np.sqrt((1 / len(diff) + n_test / n_train) * var)
    half = stats.t.ppf(0.975, df=len(diff) - 1) * se
    return Paired(mean, mean - half, mean + half, float(stats.t.sf(mean / se, df=len(diff) - 1)))


def above_chance(fold_acc: np.ndarray, n_train: float, n_test: float) -> Paired:
    """各折准确率比 0.5 平均高多少。

    以"折"为单位检验：同一批次的蛋判断结果会一起偏，按颗蛋算误差范围会偏小、容易误报。
    """
    return compare_paired(fold_acc, np.full(len(fold_acc), 0.5), n_train, n_test)


def chance_status(fold_acc: np.ndarray, n_train: float, n_test: float) -> tuple[str, Paired]:
    """“应该是瞎猜”的检查结果：通过 / 报警 / 无法确认，以及检验本身。

    p < CHANCE_ALPHA 报警；p 算不出来（折数太少或各折完全一样）时绝不算通过：
    平均准确率比 0.5 高出 CHANCE_MARGIN 以上报警，否则无法确认。
    """
    test = above_chance(fold_acc, n_train, n_test)
    if not np.isnan(test.p):
        return (CHANCE_ALARM if test.p < CHANCE_ALPHA else CHANCE_PASS), test
    mean = float(np.mean(fold_acc)) if len(fold_acc) and not np.any(np.isnan(fold_acc)) else float("nan")
    return (CHANCE_ALARM if mean > 0.5 + CHANCE_MARGIN else CHANCE_UNKNOWN), test
