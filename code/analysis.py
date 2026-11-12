"""统计评估的补充分析，全部只报告、不参与选方法（选方法的规则只有决策日主比较一条，事先写死）：

- 概率校准：分数说"80% 是雄"时，这类蛋里是不是真有约 80% 是雄。用 Brier 分数和 ECE（期望校准误差）衡量，
  另给分箱的可靠性表。准确率高不代表概率可信；以后要按分数设"拿不准就不判"的门槛，得先看这个。
- 逐日比较（多重比较校正）：决策日的比较是事先规定的唯一主检验，不校正；其他每一天的深度模型 vs 基线
  都是探索性的，天数一多总有碰巧"显著"的，所以用 Holm 方法按天数校正 p 值。
- 失败案例：决策日每颗蛋判对、判错还是没判；按批次、性别、把握程度、当天有没有图像分组看错误率，
  并列出"很有把握却判错"的蛋和两种方法都判错的蛋，方便回头核对标签和原始图像。

所有分析都用折外预测（每颗蛋只用它当测试集那一折的模型给的分数），不碰独立测试集。
"""
from dataclasses import dataclass

import numpy as np

from evaluation import ALPHA, Paired, compare_paired

N_BINS = 10                    # 可靠性表和 ECE 的分箱数（等宽）
CONFIDENT = 0.4                # |分数 - 0.5| 超过它（分数 > 0.9 或 < 0.1）算"很有把握"
CONFIDENCE_BINS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5 + 1e-9)    # 按 |分数 - 0.5| 分组看错误率
NOT_PROBABILITY = ("PLS-DA",)  # 这些方法的分数是回归值，不是概率；先截到 0~1 再算，只能当参考


def _is_probability(name: str) -> bool:
    return not any(name.startswith(p) for p in NOT_PROBABILITY)


# ---------- 概率校准 ----------
@dataclass(frozen=True)
class Calibration:
    n: int
    brier: float                # 越小越好；全猜 0.5 是 0.25
    ece: float                  # 各箱 |平均分数 - 实际雄的比例| 按蛋数加权平均；越小越好
    bins: list[dict]            # 每箱：范围、蛋数、平均分数、实际雄的比例


def calibration(scores_t: np.ndarray, sex: np.ndarray) -> Calibration:
    """一天里给出判断的蛋（分数不是 nan）。分数先截到 0~1。"""
    done = ~np.isnan(scores_t)
    s, y = np.clip(scores_t[done], 0.0, 1.0), sex[done].astype(float)
    if not len(s):
        return Calibration(0, float("nan"), float("nan"), [])
    edges = np.linspace(0, 1, N_BINS + 1)
    which = np.clip(np.digitize(s, edges[1:-1]), 0, N_BINS - 1)
    bins, ece = [], 0.0
    for b in range(N_BINS):
        sel = which == b
        if not sel.any():
            continue
        mean_s, frac = float(s[sel].mean()), float(y[sel].mean())
        ece += sel.sum() / len(s) * abs(mean_s - frac)
        bins.append({"范围": f"{edges[b]:.1f}~{edges[b + 1]:.1f}", "蛋数": int(sel.sum()),
                     "平均分数": round(mean_s, 4), "实际雄的比例": round(frac, 4)})
    return Calibration(len(s), float(np.mean((s - y) ** 2)), float(ece), bins)


def calibration_report(scores: dict[str, np.ndarray], sex: np.ndarray, t: int) -> tuple[dict, list[dict]]:
    """决策日各方法的校准：返回 (写进 config 的汇总, 可靠性表的行)。"""
    summary, rows = {}, []
    for name, s in scores.items():
        c = calibration(s[:, t], sex)
        summary[name] = {"判断的蛋数": c.n, "Brier": _r(c.brier), "ECE": _r(c.ece),
                         "分数是概率": _is_probability(name)}
        rows += [{"方法": name, **b} for b in c.bins]
    return summary, rows


# ---------- 逐日比较 ----------
def holm(p: np.ndarray) -> np.ndarray:
    """Holm 校正后的 p 值（控制族错误率，比 Bonferroni 检验力高）。nan 不参与，原样保留。"""
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    if not len(ok):
        return out
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out


def per_day_comparison(days: np.ndarray, deep: np.ndarray, base: np.ndarray, decision_t: int,
                       n_train: float, n_test: float) -> list[dict]:
    """deep、base 是两种方法各折各天的决策效用 (折数, T)。决策日是主检验，不校正；其余天一起做 Holm 校正。"""
    cmps: list[Paired] = [compare_paired(deep[:, t], base[:, t], n_train, n_test) for t in range(len(days))]
    others = [t for t in range(len(days)) if t != decision_t]
    adjusted = dict(zip(others, holm(np.array([cmps[t].p for t in others]))))
    rows = []
    for t, c in enumerate(cmps):
        primary = t == decision_t
        p_adj = c.p if primary else adjusted[t]
        rows.append({"孵化日": int(days[t]), "性质": "主检验（事先规定，不校正）" if primary else "探索性（Holm 校正）",
                     "深度决策效用": _r(np.nanmean(deep[:, t]) if not np.all(np.isnan(deep[:, t])) else np.nan),
                     "基线决策效用": _r(np.nanmean(base[:, t]) if not np.all(np.isnan(base[:, t])) else np.nan),
                     "差值": _r(c.diff), "差值95%区间下限": _r(c.ci_low), "差值95%区间上限": _r(c.ci_high),
                     "单侧p": _r(c.p), "校正后p": _r(p_adj),
                     "深度明显更好": bool(not np.isnan(p_adj) and p_adj < ALPHA)})
    return rows


# ---------- 失败案例 ----------
def _outcome(score: float, sex: int) -> str:
    if np.isnan(score):
        return "没判断"
    return "判对" if int(score > 0.5) == sex else "判错"


def failure_rows(egg_ids: tuple[str, ...], batch: np.ndarray, sex: np.ndarray, fold_of_egg: np.ndarray,
                 mask_t: np.ndarray, seen_before: np.ndarray, scores: dict[str, np.ndarray]) -> list[dict]:
    """决策日每颗蛋一行：各方法的分数和结果。排序：很有把握却判错的在最前，然后是其他判错的。"""
    names = list(scores)
    rows = []
    for i, egg in enumerate(egg_ids):
        row = {"egg_id": egg, "批次": str(batch[i]), "性别标签": int(sex[i]), "测试折": int(fold_of_egg[i]) + 1,
               "决策日有图像": int(mask_t[i]), "到决策日为止的图像数": int(seen_before[i])}
        wrong_conf = 0.0
        for name in names:
            s = scores[name][i]
            out = _outcome(s, int(sex[i]))
            row[f"{name}分数"] = _r(s)
            row[f"{name}结果"] = out
            if out == "判错":
                wrong_conf = max(wrong_conf, abs(float(np.clip(s, 0, 1)) - 0.5))
        row["都判错"] = int(all(row[f"{n}结果"] == "判错" for n in names))
        row["判错时的把握"] = round(wrong_conf, 4)
        rows.append(row)
    rows.sort(key=lambda r: (-r["判错时的把握"], r["egg_id"]))
    return rows


def _rate(outcomes: list[str]) -> dict:
    n = len(outcomes)
    wrong, none = outcomes.count("判错"), outcomes.count("没判断")
    judged = n - none
    return {"蛋数": n, "判错": wrong, "没判断": none,
            "错误率（判断了的蛋）": _r(wrong / judged if judged else np.nan),
            "决策效用": _r(outcomes.count("判对") / n if n else np.nan)}


def failure_summary(rows: list[dict], names: list[str]) -> list[dict]:
    """按批次、性别、当天有没有图像、把握程度分组，每种方法每组一行。"""
    out = []
    for name in names:
        groups: dict[tuple[str, str], list[str]] = {}

        def add(kind: str, value: str, outcome: str) -> None:
            groups.setdefault((kind, value), []).append(outcome)

        for r in rows:
            o = r[f"{name}结果"]
            add("全部", "全部", o)
            add("批次", r["批次"], o)
            add("性别标签", "雄" if r["性别标签"] == 1 else "雌", o)
            add("决策日有图像", "有" if r["决策日有图像"] else "没有", o)
            s = r[f"{name}分数"]
            if s is not None:
                conf = abs(float(np.clip(s, 0, 1)) - 0.5)
                k = int(np.digitize(conf, CONFIDENCE_BINS[1:-1]))
                add("把握 |分数-0.5|", f"{CONFIDENCE_BINS[k]:.1f}~{min(CONFIDENCE_BINS[k + 1], 0.5):.1f}", o)
        for (kind, value), outcomes in sorted(groups.items(), key=lambda kv: (kv[0][0] != "全部", kv[0])):
            out.append({"方法": name, "分组": kind, "组": value, **_rate(outcomes)})
    return out


def failure_overview(rows: list[dict], summary: list[dict], names: list[str]) -> dict:
    """写进 config 的要点：每种方法很有把握却判错几颗、错误率最高的批次、雌雄错误率；两种方法都判错几颗。"""
    info = {}
    for name in names:
        confident = sum(1 for r in rows if r[f"{name}结果"] == "判错"
                        and abs(float(np.clip(r[f"{name}分数"], 0, 1)) - 0.5) > CONFIDENT)
        mine = [s for s in summary if s["方法"] == name]
        batches = [s for s in mine if s["分组"] == "批次" and s["错误率（判断了的蛋）"] is not None]
        worst = max(batches, key=lambda s: s["错误率（判断了的蛋）"]) if batches else None
        info[name] = {"很有把握却判错": confident,
                      "错误率最高的批次": None if worst is None else {"批次": worst["组"],
                                                                      "错误率": worst["错误率（判断了的蛋）"],
                                                                      "蛋数": worst["蛋数"]},
                      "按性别的错误率": {s["组"]: s["错误率（判断了的蛋）"] for s in mine if s["分组"] == "性别标签"}}
    info["都判错的蛋数"] = sum(r["都判错"] for r in rows)
    return info


def _r(value: float) -> float | None:
    """写进表和 JSON：nan 变成 None（CSV 里是空格子）。"""
    return None if value is None or np.isnan(value) else round(float(value), 4)
