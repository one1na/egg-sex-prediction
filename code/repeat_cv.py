"""重复交叉验证：换几种分折方式（不同的随机种子）把交叉验证重跑几遍，看决策日的结论稳不稳。

运行：双击 8-重复交叉验证.bat，或 py -3.11 repeat_cv.py [--data meta.csv] [--decision-day 4] [--repeats 5]

为什么要做：一次交叉验证的结果取决于"哪些批次恰好分在同一折"。换一种分法，决策效用可能变，
"深度模型是否明显更好"的结论也可能翻转。重复几遍，能看出结论对分折方式有多敏感。

规矩：
- 只在开发集上做，独立测试集完全不碰、查看次数不变，不保存任何模型；每一遍都先过数据隔离检查。
- 第 1 遍用正式训练的分折（种子和正式训练相同），结果应该和正式训练的交叉验证一致；之后每遍换一个种子。
- 每遍都按正式规则走一次：挑决策效用最高的基线，深度模型 vs 它做配对检验。
- 各遍之间共用同一批蛋，并不独立：各遍结果的离散程度说明"换个分法会差多少"，不是置信区间；
  也不能挑最好的一遍报告——正式结果仍然是第 1 遍（即正式训练）的。
"""
import argparse
import logging
import sys
from datetime import datetime

import numpy as np

import deep_train as dt
import report
from baselines import BASELINE_NAMES
from deep_train import PRIMARY_DEEP
from evaluation import ALPHA, PRIMARY_METRIC, compare_paired, fold_day_metrics
from isolation import audit, plan_folds
from run_record import attach_log, environment, new_run_dir, reproducibility, save_csv, save_json, set_determinism
from train_demo import (DECISION_DAY, N_FOLDS, SEED, check_batches, cross_validate, decision_index, load_data,
                        split_holdout)

REPEATS = 5
SEED_STEP = 1000                # 第 r 遍（从 0 数）的分折种子 = SEED + r * SEED_STEP
SYNTH_NOTE = ("★ 这是模拟数据：性别信号是人为放进去的。这里的数字只说明重复交叉验证的代码按设计工作，"
              "不代表真实种蛋上的效果。")

log = logging.getLogger("egg_train")


def one_repeat(data, dev_weight: float, day_idx: int, seed: int) -> dict:
    """用 seed 分折跑一遍交叉验证，按正式规则比较，返回这一遍的结果。"""
    plans = plan_folds(data, N_FOLDS, seed)
    audit(data, None, plans)
    cv = cross_validate(data, dev_weight, plans)
    utility = {name: fold_day_metrics(s, data.sex, cv.fold_of_egg, cv.n_folds)[PRIMARY_METRIC][:, day_idx]
               for name, s in cv.scores.items() if name in (PRIMARY_DEEP, *BASELINE_NAMES)}
    best = max(BASELINE_NAMES, key=lambda n: np.nan_to_num(report.nanmean(utility[n]), nan=-1.0))
    cmp = compare_paired(utility[PRIMARY_DEEP], utility[best], cv.n_train, cv.n_test)
    better = bool(cmp.p < ALPHA)                    # 和正式训练同一条规则（p 算不出来按"没有明显好过"）
    return {"分折种子": seed, "折数": cv.n_folds, "深度决策效用": report.num_or_none(report.nanmean(utility[PRIMARY_DEEP])),
            "最好的基线": best, "基线决策效用": report.num_or_none(report.nanmean(utility[best])),
            "差值": report.num_or_none(cmp.diff), "单侧p": report.num_or_none(cmp.p), "深度明显更好": better,
            "按规则部署": PRIMARY_DEEP if better else best}


def spread(values: list[float | None]) -> dict:
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    v = v[~np.isnan(v)]
    if not len(v):
        return {"平均": None, "标准差": None, "最低": None, "最高": None}
    return {"平均": round(float(v.mean()), 4), "标准差": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None,
            "最低": round(float(v.min()), 4), "最高": round(float(v.max()), 4)}


def summarize(rows: list[dict]) -> dict:
    decisions = [r["按规则部署"] for r in rows]
    return {"遍数": len(rows), "深度决策效用": spread([r["深度决策效用"] for r in rows]),
            "基线决策效用": spread([r["基线决策效用"] for r in rows]), "差值": spread([r["差值"] for r in rows]),
            "深度明显更好的遍数": sum(r["深度明显更好"] for r in rows),
            "最好的基线是否每遍相同": len({r["最好的基线"] for r in rows}) == 1,
            "部署决定是否每遍相同": len(set(decisions)) == 1,
            "正式结果": "第 1 遍（和正式训练同样的分折）"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重复交叉验证：换分折方式重跑，看结论稳不稳；不碰独立测试集，不保存模型")
    parser.add_argument("--data", default=None, help="真实数据表 meta.csv 的路径；不填则用模拟数据")
    parser.add_argument("--decision-day", type=int, default=DECISION_DAY, help="看哪个孵化日的结果")
    parser.add_argument("--repeats", type=int, default=REPEATS, help=f"重复几遍（默认 {REPEATS}，至少 2）")
    parser.add_argument("--allow-no-batch", action="store_true", help="允许数据没有批次信息（结果不能当作对新批次的性能）")
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats 至少是 2")
    args.shuffle_labels, args.fertility_check, args.holdout_batches = False, False, None
    return args


def main() -> int:
    args = parse_args()
    run_dir = new_run_dir("重复交叉验证")
    attach_log(run_dir)
    set_determinism(SEED)
    if not args.data:
        log.warning(SYNTH_NOTE)
    data_all = load_data(args)
    repro = reproducibility(run_dir, data_all, args.data)
    check_batches(data_all, args)
    day_idx = decision_index(data_all.days, args.decision_day)
    dev_weight = dt.DEV_LOSS_WEIGHT if data_all.dev_mask.any() else 0.0
    data, hold = split_holdout(data_all, args)
    audit(data, hold, plan_folds(data, N_FOLDS, SEED))
    log.info("重复交叉验证只用开发集：" + ("独立测试集不参加、不评估，查看次数不变" if hold is not None else "这份数据没有留独立测试集"))
    del hold

    rows = []
    for r in range(args.repeats):
        seed = SEED + r * SEED_STEP
        log.info(f"\n===== 第 {r + 1}/{args.repeats} 遍（分折种子 {seed}{'，和正式训练相同' if r == 0 else ''}）=====")
        rows.append({"遍": r + 1, **one_repeat(data, dev_weight, day_idx, seed)})
        x = rows[-1]
        log.info(f"第 {r + 1} 遍：深度 {x['深度决策效用']}，最好的基线 {x['最好的基线']} {x['基线决策效用']}，"
                 f"差值 {x['差值']}，p {x['单侧p']}，按规则部署 {x['按规则部署']}")
    summary = summarize(rows)
    log.info(f"\n决策日（第 {args.decision_day} 天）{args.repeats} 遍汇总：")
    for key in ("深度决策效用", "基线决策效用", "差值"):
        s = summary[key]
        log.info(f"  {key}：平均 {s['平均']}，标准差 {s['标准差']}，最低 {s['最低']}，最高 {s['最高']}")
    log.info(f"  深度模型明显更好：{summary['深度明显更好的遍数']}/{args.repeats} 遍；"
             f"部署决定{'每遍相同' if summary['部署决定是否每遍相同'] else '随分折方式变化——结论不稳，别下定论'}")
    log.info("各遍共用同一批蛋、并不独立：离散程度说明结论对分折方式有多敏感，不是置信区间；正式结果仍是第 1 遍。")
    if not args.data:
        log.warning(f"\n{SYNTH_NOTE}")
    save_csv(run_dir / "repeat_cv.csv", rows)
    save_json(run_dir / "config.json", {
        "时间": datetime.now().isoformat(timespec="seconds"), "类型": "重复交叉验证", "数据": args.data or "模拟数据",
        "说明": SYNTH_NOTE if not args.data else "只做交叉验证，不碰独立测试集，不保存模型",
        "决策日": args.decision_day, "主指标": PRIMARY_METRIC, "每遍折数": N_FOLDS, "遍数": args.repeats,
        "分折种子": [r["分折种子"] for r in rows], "开发集蛋数": len(data.sex), "汇总": summary,
        "环境": environment(), "复现": repro})
    log.info(f"\n这次的全部记录在：{run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
