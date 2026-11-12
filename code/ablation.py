"""消融实验：每次从完整方法里去掉一个部件，看决策日的决策效用掉多少，回答"这个部件有没有用"。

运行：双击 7-消融实验.bat，或 py -3.11 ablation.py [--data meta.csv] [--decision-day 4]

规矩（和正式训练一致，写在代码里而不是靠自觉）：
- 只在开发集上做交叉验证，独立测试集完全不碰、查看次数不变；也不保存任何模型。
- 所有变体用同样的分折、同样的早停验证划分、同样的随机种子，和完整方法配对比较（Nadeau-Bengio 校正）。
- 要比哪几项在下面写死，全部报告；显著性门槛按比较的项数做 Bonferroni 校正，免得项数一多总有碰巧"显著"的。

变体：
- 深度模型：去掉时序（改用单帧输出，不另外训练）、去掉空间信息（每帧只给蛋区域的平均光谱）、
  去掉发育任务（发育损失权重设 0）、依次去掉一段波段。
- 基线 PLS-DA：去掉 SNV 预处理、依次去掉一段波段。

注意：在模拟数据上跑出来的结论只说明这套代码按设计工作（模拟数据的性别信号是人为放进去的），
不说明真实种蛋上哪个部件有用。真实结论要等带性别标签的真实数据。
"""
import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

import deep_train as dt
from baselines import per_day_baselines
from evaluation import (ALPHA, PRIMARY_METRIC, Paired, batch_utilities, compare_paired, fold_day_metrics,
                        group_interval, inner_split, outer_folds)
from run_record import attach_log, environment, new_run_dir, save_csv, save_json
from synth_data import EggDataset
from train_demo import DECISION_DAY, N_FOLDS, SEED, check_batches, decision_index, load_data, split_holdout

BAND_GROUPS = 4                 # 波段消融：按波长顺序平分成几段，每次去掉一段
MIN_BANDS_PER_GROUP = 2         # 每段至少这么多个波段，不够就不做波段消融
BASELINE = "PLS-DA"             # 基线消融用哪种基线（事先定好，不看结果挑）
FULL_DEEP = "深度模型（完整）"
FULL_BASE = f"{BASELINE}（完整）"
SYNTH_NOTE = ("★ 这是模拟数据：性别信号是生成数据时人为放进去的（放在哪几个波段、有没有空间分布都是设定好的），"
              "下面的结论只说明消融代码按设计工作，不说明真实种蛋上哪个部件有用。")

log = logging.getLogger("egg_train")

Transform = Callable[[EggDataset], EggDataset]


@dataclass(frozen=True)
class Variant:
    """一个消融变体。family 决定和谁比（深度模型的变体和完整深度模型比，基线的变体和完整基线比）。"""
    name: str
    family: str                         # "深度" 或 "基线"
    question: str                       # 这个变体回答什么问题
    transform: Transform | None = None  # 对数据做什么改动；None = 不改
    dev_weight: float | None = None     # 深度模型的发育损失权重；None = 和完整模型一样
    output: str = "cum"                 # 深度模型用哪个输出：cum 累计，frame 单帧
    reuse_full: bool = False            # True = 不另外训练，直接用完整模型的另一个输出
    use_snv: bool = True                # 基线是否做 SNV


@dataclass(frozen=True)
class AblationResult:
    variants: tuple[Variant, ...]
    scores: dict[str, np.ndarray]       # 变体名 -> (N, T) 折外判为雄的分数
    fold_of_egg: np.ndarray
    n_folds: int
    n_train: float
    n_test: float


# ---------- 数据改动 ----------
def no_spatial(data: EggDataset) -> EggDataset:
    """去掉空间信息：每帧只留蛋区域的平均光谱（和基线看到的输入一样），图像变成 1×1。"""
    return replace(data, x=np.ascontiguousarray(data.spec[:, :, :, None, None].astype(np.float32)))


def band_groups(n_bands: int) -> list[np.ndarray]:
    """把波段按顺序平分成 BAND_GROUPS 段；波段太少（每段不到 MIN_BANDS_PER_GROUP 个）时返回空。"""
    if n_bands < BAND_GROUPS * MIN_BANDS_PER_GROUP:
        return []
    return [g for g in np.array_split(np.arange(n_bands), BAND_GROUPS)]


def drop_bands(drop: np.ndarray) -> Transform:
    """去掉 drop 这些波段（直接删掉，不是置零：置零会留下"这里被改过"的痕迹）。"""
    def transform(data: EggDataset) -> EggDataset:
        keep = np.setdiff1d(np.arange(data.x.shape[2]), drop)
        return replace(data, x=np.ascontiguousarray(data.x[:, :, keep]), spec=data.spec[:, :, keep],
                       wavelengths=None if data.wavelengths is None else data.wavelengths[keep],
                       bg_spec=None if data.bg_spec is None else data.bg_spec[:, :, keep])
    return transform


def group_label(group: np.ndarray, wavelengths: np.ndarray | None) -> str:
    if wavelengths is None:
        return f"第 {group[0] + 1}–{group[-1] + 1} 波段"
    return f"{wavelengths[group[0]]:.0f}–{wavelengths[group[-1]]:.0f} nm"


def build_variants(data: EggDataset, dev_weight: float) -> tuple[Variant, ...]:
    """这次要比的全部变体，事先定好。数据不支持的项不做（并在日志里说明），不会悄悄少一项。"""
    variants = [Variant(FULL_DEEP, "深度", "对照：什么都不去掉"),
                Variant("深度：去掉时序（单帧输出）", "深度", "把多天串起来比只看当天强多少", output="frame", reuse_full=True)]
    if data.x.shape[3] * data.x.shape[4] > 1:
        variants.append(Variant("深度：去掉空间信息（只用平均光谱）", "深度", "蛋内的空间分布有没有用", transform=no_spatial))
    else:
        log.info("  数据每帧只有一条光谱，没有空间信息，不做“去掉空间信息”")
    if dev_weight > 0:
        variants.append(Variant("深度：去掉发育任务", "深度", "同时学发育对判性别有没有帮助", dev_weight=0.0))
    else:
        log.info("  数据没有发育指标，发育任务本来就是关的，不做“去掉发育任务”")
    groups = band_groups(data.x.shape[2])
    if not groups:
        log.info(f"  只有 {data.x.shape[2]} 个波段，不够分 {BAND_GROUPS} 段，不做波段消融")
    for g in groups:
        variants.append(Variant(f"深度：去掉 {group_label(g, data.wavelengths)}", "深度", "性别信号在不在这段波长",
                                transform=drop_bands(g)))
    variants.append(Variant(FULL_BASE, "基线", "对照：什么都不去掉"))
    variants.append(Variant(f"{BASELINE}：去掉 SNV", "基线", "SNV 预处理有没有用", use_snv=False))
    for g in groups:
        variants.append(Variant(f"{BASELINE}：去掉 {group_label(g, data.wavelengths)}", "基线", "性别信号在不在这段波长",
                                transform=drop_bands(g)))
    return tuple(variants)


# ---------- 交叉验证 ----------
def _deep_fold(data: EggDataset, fit_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
               dev_weight: float, seed: int) -> dict[str, np.ndarray]:
    trained = dt.train_early_stop(data, fit_idx, val_idx, dev_weight, seed)
    return dt.predict_scores(trained, data.x[test_idx], data.mask[test_idx], data.days)


def run_ablation(data: EggDataset, dev_weight: float, variants: tuple[Variant, ...]) -> AblationResult:
    """所有变体共用同一套外层分折、早停验证划分和随机种子（和正式训练的完全一样）。"""
    folds = outer_folds(data.sex, data.batch, N_FOLDS, SEED)
    n_eggs, n_days = data.mask.shape
    scores = {v.name: np.full((n_eggs, n_days), np.nan) for v in variants}
    fold_of_egg = np.full(n_eggs, -1)
    views = {v.name: (data if v.transform is None else v.transform(data)) for v in variants}
    log.info(f"\n开发集 {n_eggs} 颗蛋，{len(folds)} 折交叉验证 × {len(variants)} 个变体")
    for k, (train_idx, test_idx) in enumerate(folds):
        start = time.time()
        seed = SEED + k
        fit_idx, val_idx, _ = inner_split(train_idx, data.sex, data.batch, seed)
        fold_of_egg[test_idx] = k
        full = None
        for v in variants:
            view = views[v.name]
            if v.family == "基线":
                out = per_day_baselines(view.spec, view.mask, view.sex, train_idx, test_idx, use_snv=v.use_snv)[BASELINE]
            elif v.reuse_full:
                out = full[v.output]        # 完整模型排在最前面，这里一定已经训练过
            else:
                weight = dev_weight if v.dev_weight is None else v.dev_weight
                deep = _deep_fold(view, fit_idx, val_idx, test_idx, weight, seed)
                if v.name == FULL_DEEP:
                    full = deep
                out = deep[v.output]
            scores[v.name][test_idx] = out
        log.info(f"  第 {k + 1}/{len(folds)} 折完成，用时 {time.time() - start:.0f} 秒")
    return AblationResult(variants, scores, fold_of_egg, len(folds),
                          float(np.mean([len(tr) for tr, _ in folds])), float(np.mean([len(te) for _, te in folds])))


# ---------- 汇总 ----------
def _num(value: float) -> float | None:
    return None if np.isnan(value) else round(float(value), 4)


def reference_of(v: Variant) -> str:
    return FULL_DEEP if v.family == "深度" else FULL_BASE


def n_comparisons(variants: tuple[Variant, ...]) -> int:
    return sum(v.name not in (FULL_DEEP, FULL_BASE) for v in variants)


def summarize(data: EggDataset, res: AblationResult, day_idx: int) -> list[dict]:
    """每个变体一行：决策日的决策效用，以及"去掉它以后掉了多少"（完整 − 变体）和配对检验。"""
    metric = {name: fold_day_metrics(s, data.sex, res.fold_of_egg, res.n_folds)[PRIMARY_METRIC][:, day_idx]
              for name, s in res.scores.items()}
    batches = np.unique(data.batch.astype(str))
    has_batches = len(batches) < len(data.sex)
    threshold = ALPHA / max(1, n_comparisons(res.variants))
    rows = []
    for v in res.variants:
        mine = metric[v.name]
        is_ref = v.name == reference_of(v)
        cmp = Paired(*(float("nan"),) * 4) if is_ref else compare_paired(metric[reference_of(v)], mine,
                                                                         res.n_train, res.n_test)
        per_batch = group_interval(batch_utilities(res.scores[v.name][:, day_idx], data.sex, data.batch, batches)
                                   if has_batches else np.array([np.nan]))
        rows.append({"变体": v.name, "回答的问题": v.question, "对照": "" if is_ref else reference_of(v),
                     "决策效用": _num(np.nanmean(mine)), "各折标准差": _num(np.nanstd(mine, ddof=1)),
                     "按批次区间下限": _num(per_batch.ci_low), "按批次区间上限": _num(per_batch.ci_high),
                     "去掉后下降": _num(cmp.diff), "下降95%下限": _num(cmp.ci_low), "下降95%上限": _num(cmp.ci_high),
                     "单侧p": _num(cmp.p), "校正后门槛": "" if is_ref else round(threshold, 4),
                     "结论": "" if is_ref else verdict(cmp, threshold)})
    return rows


def verdict(cmp: Paired, threshold: float) -> str:
    """只有两种说法：有明确贡献（校正后仍显著）/ 看不出贡献。看不出不等于没有：数据少时检验力很低。"""
    if np.isnan(cmp.p):
        return "无法判断（折数太少或各折完全一样）"
    return "有明确贡献" if cmp.p < threshold else "看不出贡献"


def per_day_rows(data: EggDataset, res: AblationResult) -> list[dict]:
    """每个变体在每个孵化日的决策效用（各折平均），看贡献随孵化日怎么变。"""
    rows = []
    for name, s in res.scores.items():
        utility = fold_day_metrics(s, data.sex, res.fold_of_egg, res.n_folds)[PRIMARY_METRIC]
        rows.append({"变体": name, **{f"第{int(d)}天": _num(np.nanmean(utility[:, t])) for t, d in enumerate(data.days)}})
    return rows


def _fmt(value: float | None, signed: bool = False) -> str:
    if value is None:
        return "   —  "
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def log_summary(rows: list[dict], day: int, n_cmp: int) -> None:
    width = max(len(r["变体"]) for r in rows) + 2
    log.info(f"\n===== 消融结果：第 {day} 天的决策效用（各折平均）=====")
    log.info(f"“去掉后下降”= 完整方法 − 这个变体，正数说明这个部件有贡献。共 {n_cmp} 项比较，"
             f"单侧 p 低于 {ALPHA}/{n_cmp} = {ALPHA / max(1, n_cmp):.4f} 才算“有明确贡献”。")
    log.info(f"{'变体'.ljust(width)}  效用    去掉后下降  [95% 区间]          单侧p   结论")
    for r in rows:
        if not r["对照"]:
            log.info(f"{r['变体'].ljust(width)}  {_fmt(r['决策效用'])}   （对照）")
            continue
        interval = f"[{_fmt(r['下降95%下限'], True)}, {_fmt(r['下降95%上限'], True)}]"
        log.info(f"{r['变体'].ljust(width)}  {_fmt(r['决策效用'])}   {_fmt(r['去掉后下降'], True)}     {interval}   "
                 f"{_fmt(r['单侧p'])}   {r['结论']}")


def log_per_day(rows: list[dict]) -> None:
    width = max(len(r["变体"]) for r in rows) + 2
    days = [k for k in rows[0] if k != "变体"]
    log.info("\n逐日决策效用（各折平均）")
    log.info("变体".ljust(width) + "  " + "  ".join(d.rjust(6) for d in days))
    for r in rows:
        log.info(r["变体"].ljust(width) + "  " + "  ".join(_fmt(r[d]).rjust(7) for d in days))


# ---------- 入口 ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="消融实验：只做交叉验证，不碰独立测试集，不保存模型")
    parser.add_argument("--data", default=None, help="真实数据表 meta.csv 的路径；不填则用模拟数据")
    parser.add_argument("--decision-day", type=int, default=DECISION_DAY, help="看哪个孵化日的结果")
    parser.add_argument("--allow-no-batch", action="store_true", help="允许数据没有批次信息（结果不能当作对新批次的性能）")
    args = parser.parse_args()
    # split_holdout / load_data 还会读这几项：消融不做标签打乱，也不新指定测试批次
    args.shuffle_labels, args.fertility_check, args.holdout_batches = False, False, None
    return args


def main() -> int:
    args = parse_args()
    run_dir = new_run_dir("消融实验")
    attach_log(run_dir)
    if not args.data:
        log.warning(SYNTH_NOTE)
    data_all = load_data(args)
    check_batches(data_all, args)
    day_idx = decision_index(data_all.days, args.decision_day)
    dev_weight = dt.DEV_LOSS_WEIGHT if data_all.dev_mask.any() else 0.0
    data, hold = split_holdout(data_all, args)      # 测试批次只是被排除在外，下面不会读它
    log.info("消融只用开发集做交叉验证：" + ("独立测试集不参加、不评估，查看次数不变" if hold is not None else "这份数据没有留独立测试集"))
    del hold

    variants = build_variants(data, dev_weight)
    res = run_ablation(data, dev_weight, variants)
    rows, days = summarize(data, res, day_idx), per_day_rows(data, res)
    log_summary(rows, args.decision_day, n_comparisons(variants))
    log_per_day(days)
    log.info("\n读结果时注意：“看不出贡献”不等于“没有贡献”——蛋少、折少时检验力很低，下降的区间很宽就属于这种情况。")
    if not args.data:
        log.warning(f"\n{SYNTH_NOTE}")
    save_csv(run_dir / "ablation.csv", rows)
    save_csv(run_dir / "ablation_per_day.csv", days)
    save_json(run_dir / "config.json", {
        "时间": datetime.now().isoformat(timespec="seconds"), "类型": "消融实验", "数据": args.data or "模拟数据",
        "说明": SYNTH_NOTE if not args.data else "只做交叉验证，不碰独立测试集，不保存模型",
        "决策日": args.decision_day, "主指标": PRIMARY_METRIC, "折数": res.n_folds, "随机种子": SEED,
        "开发集蛋数": len(data.sex), "发育损失权重": dev_weight, "基线": BASELINE, "波段分段数": BAND_GROUPS,
        "比较项数": n_comparisons(variants), "显著性水平": ALPHA,
        "校正后门槛": ALPHA / max(1, n_comparisons(variants)),
        "变体": [{"名字": v.name, "回答的问题": v.question} for v in variants], "环境": environment()})
    log.info(f"\n这次消融的全部记录在：{run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
