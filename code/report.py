"""把结果整理成表：打印到窗口和日志，并生成保存成 CSV 的行。"""
import logging
import unicodedata

import numpy as np

from baselines import day_only_dev
from evaluation import (ALPHA, BATCH_INTERVAL_LABEL, CHANCE_ALARM, CHANCE_PASS, CHANCE_UNKNOWN, MIN_BATCHES_FOR_INTERVAL,
                        WILSON_LABEL, CommonAccuracy, GroupInterval, Paired, ScopeStats, chance_status, group_interval,
                        wilson_ci)
from synth_data import EggDataset

NAME_WIDTH = 16
CELL_WIDTH = 13
Metrics = dict[str, dict[str, np.ndarray]]   # 方法名 -> fold_day_metrics 的结果
METRIC_NAMES = {"utility": "决策效用"}

log = logging.getLogger("egg_train")


# ---------- 数字格式 ----------
def nanmean(values: np.ndarray) -> float:
    """全是 nan 时返回 nan，不报警告。"""
    return float("nan") if np.all(np.isnan(values)) else float(np.nanmean(values))


def _mean_std(values: np.ndarray) -> str:
    mean = nanmean(values)
    return "-" if np.isnan(mean) else f"{mean:.3f}±{np.nanstd(values):.3f}"


def _mean(values: np.ndarray) -> str:
    mean = nanmean(values)
    return "-" if np.isnan(mean) else f"{mean:.3f}"


def _num(value: float) -> float | str:
    """保存用：nan 写成空格子。"""
    return "" if np.isnan(value) else round(float(value), 4)


def num_or_none(value: float) -> float | None:
    """写进 JSON 用：nan 变成 null。"""
    return None if np.isnan(value) else round(float(value), 4)


def _p(p: float) -> str:
    return "无法估计（折数太少或各折差值完全相同）" if np.isnan(p) else f"{p:.3f}"


def _width(text: str) -> int:
    """中文在等宽字体里占两格。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int = NAME_WIDTH) -> str:
    return text + " " * max(width - _width(text), 0)


def _header(days: np.ndarray) -> str:
    return _pad("孵化日") + "".join(f"{int(d):>{CELL_WIDTH}}" for d in days)


# ---------- 打印 ----------
def log_per_day(days: np.ndarray, metrics: Metrics, key: str, title: str) -> None:
    """每行一种方法，每列一个孵化日，格子里是 各折平均±标准差。"""
    log.info(f"\n{title}")
    log.info(_header(days))
    for name, m in metrics.items():
        log.info(_pad(name) + "".join(f"{_mean_std(m[key][:, t]):>{CELL_WIDTH}}" for t in range(len(days))))


def log_decision_day(day: int, t: int, metrics: Metrics, deep: str, base: str, cmp: Paired,
                     metric: str, common: dict[str, CommonAccuracy], pooled: dict[str, float]) -> bool:
    """决策日：各方法的决策效用（主指标）、覆盖率、只在判断了的蛋上算的准确率，以及两两在共同可判断蛋上的准确率。
    决策效用同时给"各折平均"和"合并全部折外预测按蛋算"两种；检验和选择用各折平均（事先规定）。

    返回深度模型的主指标有没有明显好过最好的基线。
    """
    name = METRIC_NAMES.get(metric, metric)
    log.info(f"\n★ 决策日（第 {day} 天）交叉验证结果，各折平均±标准差")
    log.info(f"  事先规定的主指标：{name}（正确判断的蛋 / 全部蛋，判断不了算没判对），"
             "不受各方法覆盖率不同的影响；其余指标供参考")
    for method, m in metrics.items():
        log.info(f"  {_pad(method)}{name} {_mean_std(m[metric][:, t])}   覆盖率 {_mean(m['coverage'][:, t])}"
                 f"   判断了的蛋上的准确率 {_mean_std(m['acc'][:, t])}"
                 f"   平衡准确率 {_mean(m['bal_acc'][:, t])}   AUC {_mean(m['auc'][:, t])}")
    log.info(f"  {name}的另一种算法：把各折的折外预测合在一起按蛋算（各折蛋数不同时和各折平均会有出入，仅供对照）")
    log.info("    " + "   ".join(f"{method} {_mean(np.array([value]))}" for method, value in pooled.items()))
    log.info(f"  共同可判断的蛋上的准确率（{deep} vs 各基线，只算两者都给出判断的蛋，各折平均）")
    for method, c in common.items():
        log.info(f"    vs {_pad(method)}{deep} {_mean(c.acc_a)}   {method} {_mean(c.acc_b)}   "
                 f"平均每折 {np.mean(c.n_common):.1f} 颗")
    better = bool(cmp.p < ALPHA)
    ci = "无法估计" if np.isnan(cmp.ci_low) else f"[{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}]"
    log.info(f"  → {deep}比最好的基线（{base}）{name}平均高 {cmp.diff:+.3f}，95% 置信区间 {ci}，单侧 p = {_p(cmp.p)}"
             f"（配对 t 检验，已按交叉验证校正）：{'明显好过' if better else '没有明显好过'}基线")
    if not better:
        log.info(f"    按流程规定：深度模型没有明显好过基线时，判断新蛋改用 {base}")
    return better


def log_matched(deep: str, base: str, cmp: Paired, metric: str, formal_better: bool) -> bool:
    """同样本量对照：基线只用深度模型拟合用的那部分蛋训练。只报告，不参与选择。返回这个对照里深度模型是否明显更好。"""
    name = METRIC_NAMES.get(metric, metric)
    better = bool(cmp.p < ALPHA)
    ci = "无法估计" if np.isnan(cmp.ci_low) else f"[{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}]"
    log.info("  同样本量对照（只报告，不参与选择）：上面的基线用了这一折的全部训练蛋，深度模型却要留出一部分做早停验证；"
             f"这里让基线也只用深度模型拟合用的那部分蛋（{base}）")
    log.info(f"    {deep}比 {base} {name}平均高 {cmp.diff:+.3f}，95% 置信区间 {ci}，单侧 p = {_p(cmp.p)}："
             f"{'明显好过' if better else '没有明显好过'}")
    if better != formal_better:
        log.warning("    注意：两种比较的结论不一致，谁更好取决于给基线多少训练样本，"
                    "不能据此下“深度模型更好”或“基线更好”的结论")
    return better


def _interval_text(g: GroupInterval, signed: bool = False) -> str:
    if g.n == 0:
        return "-"
    fmt = "+.3f" if signed else ".3f"
    ci = (f"无法估计（批次少于 {MIN_BATCHES_FOR_INTERVAL} 个）" if np.isnan(g.ci_low)
          else f"[{g.ci_low:{fmt}}, {g.ci_high:{fmt}}]")
    return f"平均 {g.mean:{fmt}}，95% 区间 {ci}，最低 {g.low:{fmt}}，最高 {g.high:{fmt}}"


def _interval_dict(g: GroupInterval) -> dict:
    return {"批次数": g.n, "平均": num_or_none(g.mean), "95%区间": [num_or_none(g.ci_low), num_or_none(g.ci_high)],
            "最低": num_or_none(g.low), "最高": num_or_none(g.high)}


def _wins(diff: np.ndarray) -> tuple[int, int, int]:
    """差值里 (更高, 持平, 更低) 的批次数。"""
    return int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())


def log_per_batch(day: int, utility: dict[str, np.ndarray], diffs: dict[str, np.ndarray], deep: str) -> None:
    """决策日按批次看：每种方法各批次的决策效用，以及深度模型减基线的差值，都以批次为单位给区间。"""
    n = len(next(iter(utility.values())))
    log.info(f"\n★ 决策日（第 {day} 天）按批次看的决策效用：{n} 个批次，每个批次算一个值（来自它当测试集的那一折）")
    log.info(f"  区间是{BATCH_INTERVAL_LABEL}")
    for name, values in utility.items():
        log.info(f"  {_pad(name)}{_interval_text(group_interval(values))}")
    for base, diff in diffs.items():
        up, same, down = _wins(diff)
        log.info(f"  {deep} 减 {base}：{_interval_text(group_interval(diff), True)}；"
                 f"{n} 个批次里 {up} 个更高、{same} 个持平、{down} 个更低")


def per_batch_rows(names: np.ndarray, n_eggs: np.ndarray, utility: dict[str, np.ndarray],
                   diffs: dict[str, np.ndarray], deep: str) -> list[dict]:
    """每个批次一行：蛋数、各方法的决策效用、深度模型减各基线的差值。"""
    rows = []
    for i, name in enumerate(names):
        row = {"批次": str(name), "蛋数": int(n_eggs[i])}
        row.update({f"{method}决策效用": _num(values[i]) for method, values in utility.items()})
        row.update({f"{deep}减{base}": _num(diff[i]) for base, diff in diffs.items()})
        rows.append(row)
    return rows


def per_batch_summary(utility: dict[str, np.ndarray], diffs: dict[str, np.ndarray], deep: str) -> dict:
    """写进 config.json 的按批次汇总。"""
    summary = {"区间说明": BATCH_INTERVAL_LABEL,
               "各方法决策效用": {name: _interval_dict(group_interval(values)) for name, values in utility.items()},
               "差值": {}}
    for base, diff in diffs.items():
        up, same, down = _wins(diff)
        summary["差值"][f"{deep}减{base}"] = {**_interval_dict(group_interval(diff)),
                                             "更高批次数": up, "持平批次数": same, "更低批次数": down}
    return summary


def holdout_batch_interval(rows: list[ScopeStats]) -> dict | None:
    """独立测试集各批次决策效用的按批次汇总（rows 第一行是全部，后面每批一行）；不到 2 个批次返回 None。"""
    if len(rows) < 3:
        return None
    return _interval_dict(group_interval(np.array([r.utility for r in rows[1:]])))


def _scope_line(r: ScopeStats, indent: str, interval: bool) -> str:
    text = (f"{indent}蛋 {r.n_eggs} 颗，判断 {r.judged} 颗，判对 {r.correct} 颗   覆盖率 {r.coverage:.3f}"
            f"   条件准确率 {r.accuracy:.3f}   决策效用 {r.utility:.3f}")
    if interval:
        low, high = wilson_ci(r.correct, r.judged)
        text += f"\n      条件准确率的{WILSON_LABEL}：{low:.3f}~{high:.3f}"
    return text


FORMAL_RESULT = "正式结果（第一次查看）"
NOT_INDEPENDENT = "调试查看，非独立验证"


def log_holdout(day: int, tables: dict[str, list[ScopeStats]], chosen: str, views: int, formal: bool = True) -> None:
    """独立测试集上选定方法在决策日的结果（总体和每个批次）。没选中的方法不评估。
    formal=False（调试模式再次查看）时不标"正式结果"，改标"非独立验证"。"""
    if formal:
        log.info(f"\n★★ 独立测试集（开发阶段完全没碰过的批次），决策日第 {day} 天，这是第 {views} 次查看")
    else:
        log.info(f"\n★★ 【{NOT_INDEPENDENT}】已经看过的测试集，决策日第 {day} 天，这是第 {views} 次查看；"
                 "下面的数字不是独立验证结果")
    log.info("  只评估交叉验证阶段事先选定的方法；没选中的方法不在这里打开成绩，免得看到分数后回头改选择")
    mark = "正式结果" if formal else NOT_INDEPENDENT
    for name, rows in tables.items():
        log.info(f"  {name}" + (f"  ← {mark}" if name == chosen else ""))
        total = rows[0]
        log.info(_scope_line(total, "    全部：", True) + f"   平衡准确率 {_mean(np.array([total.bal_acc]))}"
                 f"   AUC {_mean(np.array([total.auc]))}")
        for r in rows[1:]:
            log.info(_scope_line(r, f"    批次 {r.scope}：", False))
        if len(rows) >= 3:
            batches = group_interval(np.array([r.utility for r in rows[1:]]))
            log.info(f"    按批次看决策效用：{_interval_text(batches)}\n      区间是{BATCH_INTERVAL_LABEL}")


def log_dev(data: EggDataset, dev_pred: np.ndarray) -> list[dict]:
    """发育预测的平均误差（所有折合在一起，只算有实测值的帧），对照是只看孵化日。返回保存用的行。"""
    if not data.dev_mask.any():
        log.info("\n发育指标全部留空：已关掉发育这一项，不报告发育预测")
        return []
    naive = day_only_dev(data.days)
    log.info("\n发育预测的平均误差 MAE（所有折合在一起，只算有实测发育指标的帧；对照 = 只看孵化日）")
    rows = []
    for t, day in enumerate(data.days):
        sel = data.dev_mask[:, t] & data.mask[:, t] & ~np.isnan(dev_pred[:, t])
        if not sel.any():
            continue
        deep_mae = float(np.abs(dev_pred[sel, t] - data.dev[sel, t]).mean())
        naive_mae = float(np.abs(naive[t] - data.dev[sel, t]).mean())
        log.info(f"  第 {int(day):>2} 天   深度模型 {deep_mae:.3f}   只看孵化日 {naive_mae:.3f}"
                 f"   （{int(sel.sum())} 帧）")
        rows.append({"孵化日": int(day), "深度模型MAE": round(deep_mae, 4),
                     "只看孵化日MAE": round(naive_mae, 4), "帧数": int(sel.sum())})
    return rows


CHANCE_NOTES = {CHANCE_PASS: "通过", CHANCE_ALARM: "报警：明显偏高，可能有信息泄漏",
                CHANCE_UNKNOWN: "无法确认：统计量算不出来（折数太少或各折完全一样），不能当作通过"}


def log_chance(metrics: Metrics, day_indices: list[int], n_train: float, n_test: float,
               title: str) -> dict[str, str]:
    """应该是瞎猜的情况：每种方法在这些孵化日的准确率都应接近 0.5。返回 {方法: 通过/报警/无法确认}。"""
    log.info(f"\n{title}")
    statuses = {}
    for name, m in metrics.items():
        fold_acc = np.array([nanmean(m["acc"][k, day_indices]) for k in range(m["acc"].shape[0])])
        status, test = chance_status(fold_acc, n_train, n_test)
        statuses[name] = status
        log.info(f"  {_pad(name)}准确率 {_mean_std(fold_acc)}   高于 0.5 的单侧 p = {_p(test.p)}   "
                 f"{CHANCE_NOTES[status]}")
    return statuses


def chance_passed(statuses: dict[str, str]) -> bool:
    """自检整体通过 = 每种方法都是"通过"。"""
    return all(s == CHANCE_PASS for s in statuses.values())


# ---------- 保存 ----------
def fold_rows(data: EggDataset, fold_of_egg: np.ndarray) -> list[dict]:
    return [{"egg_id": e, "批次": str(b), "性别标签": int(s), "测试折": int(k) + 1}
            for e, b, s, k in zip(data.egg_ids, data.batch, data.sex, fold_of_egg)]


def per_day_rows(days: np.ndarray, metrics: Metrics) -> list[dict]:
    rows = []
    for name, m in metrics.items():
        for t, day in enumerate(days):
            if np.all(np.isnan(m["acc"][:, t])):
                continue
            rows.append({"方法": name, "孵化日": int(day), "覆盖率均值": _num(nanmean(m["coverage"][:, t])),
                         "准确率均值": _num(nanmean(m["acc"][:, t])),
                         "准确率标准差": _num(np.nanstd(m["acc"][:, t])),
                         "平衡准确率均值": _num(nanmean(m["bal_acc"][:, t])),
                         "AUC均值": _num(nanmean(m["auc"][:, t])), "决策效用均值": _num(nanmean(m["utility"][:, t])),
                         "判断的蛋数": int(np.nansum(m["n"][:, t]))})
    return rows


def per_fold_rows(days: np.ndarray, metrics: Metrics) -> list[dict]:
    rows = []
    for name, m in metrics.items():
        for k in range(m["acc"].shape[0]):
            for t, day in enumerate(days):
                if np.isnan(m["acc"][k, t]):
                    continue
                rows.append({"方法": name, "折": k + 1, "孵化日": int(day), "覆盖率": _num(m["coverage"][k, t]),
                             "准确率": _num(m["acc"][k, t]), "平衡准确率": _num(m["bal_acc"][k, t]),
                             "AUC": _num(m["auc"][k, t]), "决策效用": _num(m["utility"][k, t]),
                             "判断的蛋数": int(m["n"][k, t])})
    return rows


def prediction_rows(data: EggDataset, scores: dict[str, np.ndarray], dev_pred: np.ndarray,
                    fold_of_egg: np.ndarray) -> list[dict]:
    """每颗蛋每天一行（只要有一种方法给出判断）：在哪一折当测试集、性别标签、各方法判为雄的分数、发育预测。"""
    rows = []
    for i, egg in enumerate(data.egg_ids):
        for t, day in enumerate(data.days):
            if all(np.isnan(s[i, t]) for s in scores.values()):
                continue
            row = {"egg_id": egg, "批次": str(data.batch[i]), "测试折": int(fold_of_egg[i]) + 1,
                   "孵化日": int(day), "当天有图像": int(data.mask[i, t]), "性别标签": int(data.sex[i])}
            row.update({f"{name}分数": _num(s[i, t]) for name, s in scores.items()})
            row["发育实测"] = _num(data.dev[i, t]) if data.dev_mask[i, t] else ""
            row["发育预测"] = _num(dev_pred[i, t]) if data.dev_mask.any() else ""
            rows.append(row)
    return rows


def holdout_rows(day: int, tables: dict[str, list[ScopeStats]], chosen: str, views: int,
                 formal: bool = True) -> list[dict]:
    """每种方法：一行总体 + 每个批次一行。区间只给总体，并注明它是按鸡蛋算的描述性区间。
    formal=False（调试模式再次查看）时"正式结果"一律为 0，"性质"列写明非独立验证。"""
    rows = []
    for name, table in tables.items():
        for r in table:
            low, high = wilson_ci(r.correct, r.judged) if r.scope == "全部" else (np.nan, np.nan)
            rows.append({"方法": name, "范围": r.scope, "决策日": day, "正式结果": int(formal and name == chosen),
                         "性质": FORMAL_RESULT if formal else NOT_INDEPENDENT,
                         "蛋数": r.n_eggs, "判断的蛋数": r.judged, "判对的蛋数": r.correct,
                         "覆盖率": _num(r.coverage), "条件准确率": _num(r.accuracy), "决策效用": _num(r.utility),
                         f"条件准确率区间下限，{WILSON_LABEL}": _num(low),
                         f"条件准确率区间上限，{WILSON_LABEL}": _num(high),
                         "平衡准确率": _num(r.bal_acc), "AUC": _num(r.auc), "测试集被查看次数": views})
    return rows


def comparison_rows(metrics: Metrics, t: int, deep: str, common: dict[str, CommonAccuracy]) -> list[dict]:
    """决策日：深度模型对每个基线，每折的决策效用、覆盖率和共同可判断蛋上的准确率。"""
    rows = []
    for base, c in common.items():
        for k in range(len(c.n_common)):
            rows.append({"折": k + 1, "对比基线": base, "共同可判断蛋数": int(c.n_common[k]),
                         "深度共同准确率": _num(c.acc_a[k]), "基线共同准确率": _num(c.acc_b[k]),
                         "深度覆盖率": _num(metrics[deep]["coverage"][k, t]),
                         "基线覆盖率": _num(metrics[base]["coverage"][k, t]),
                         "深度决策效用": _num(metrics[deep]["utility"][k, t]),
                         "基线决策效用": _num(metrics[base]["utility"][k, t])})
    return rows
