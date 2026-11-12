"""训练流程：读数据 → 留出独立测试批次 → 开发集上按批次分组交叉验证（深度模型和基线比）
→ 决策日规则选定部署方法 → 用开发集训练最终模型 → 独立测试集只测一次。

运行：双击 1-训练模型.bat，或 py -3.11 train_demo.py [--data meta.csv] [--decision-day 4] [--holdout-batches B10,B11]
独立测试批次第一次留出后就记在锁文件里，以后新增的批次都进开发集；--holdout-batches 用来明确指定一组新批次当测试集。
独立测试集只能正式评估一次：已经看过的测试集再训练会在开头被拦下（退出码 3）。--skip-holdout 只重新训练、不看测试集；
          --debug-holdout 允许调试查看，结果一律标注"非独立验证"。
流程自检：双击 4-流程自检.bat，或加 --shuffle-labels。性别标签随机打乱后整套跑一遍，
          准确率应该都在 0.5 左右；明显偏高说明测试数据的信息漏进了训练。自检不保存模型，没通过时退出码非 0。
模型发布门控：正常训练时自检或背景负对照没通过，就不训练最终模型、不看独立测试集、不写模型文件（退出码 2），
          output 里上一次通过检查的模型原样保留。
流水线检验：双击 6-真实光谱检验.bat，或加 --fertility-check。用伊利诺伊的真实平均光谱（标签是受精状态，不是性别）
          把整条流程跑一遍；不保存判断新蛋用的模型，锁文件放在那份数据旁边。
真实数据必须有 batch 列；确实没有时加 --allow-no-batch（每颗蛋自成一组，结果不能当作对新批次的性能）。
每次运行的配置、分折、逐日结果、独立测试结果和日志存到 output/runs/<时间>_<类型>/；
最终模型存到 output/egg_model.pt（选了基线时另有 output/baseline_model.pkl），运行文件夹里也留一份。
"""
import argparse
import logging
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

import numpy as np

import baseline_fertility
import deep_train as dt
import deploy
import holdout_lock
import report
from baselines import BASELINE_NAMES, LOGREG_C, MATCHED_NAMES, PLS_COMPONENTS, matched_name, per_day_baselines
from controls import run_controls
from deep_train import PRIMARY_DEEP
from evaluation import (CHANCE_PASS, CHANCE_UNKNOWN, MIN_HOLDOUT_BATCHES, PRIMARY_METRIC, WILSON_LABEL, batch_utilities,
                        common_accuracy, compare_paired,
                        fold_day_metrics, holdout_split, pooled_utility)
from isolation import FoldPlan, IsolationError, audit, plan_folds
from preprocess_binding import preprocess_settings
from real_data import eggs_without_batch, load_real_dataset
from run_record import (EXIT_HOLDOUT_USED, EXIT_SELF_CHECK_FAILED, attach_log, environment, new_run_dir,
                        reproducibility, save_csv, save_json, set_determinism)
from synth_data import EggDataset, make_dataset

# 数据来源：None = 用模拟数据；填 meta.csv 的路径 = 用真实数据
DATA_CSV: str | None = None
DECISION_DAY = 4            # 必须出结果的孵化日，主指标只看这一天
N_FOLDS = 5
SEED = 42
DEEP_NAMES = ("深度-单帧", PRIMARY_DEEP)
FERTILITY_NOTE = ("★ 流水线检验（伊利诺伊公开数据）：这里的标签是【受精状态】，不是性别。日志和结果表里的“雄”= 受精、"
                  "“雌”= 未受精。每颗蛋只有一条平均光谱、一个时间点（记作第 0 天），所以校正、找蛋裁剪没有被检验，"
                  "深度模型的空间卷积和时序部分也只是空转。结果只能说明这套流程在真实光谱上跑得通，"
                  "不能说明种蛋性别可分，也不是性别鉴定的准确率。")
HOLDOUT_CSV = "holdout.csv"                            # 独立测试集的正式结果（第一次查看）
HOLDOUT_DEBUG_CSV = "holdout_调试_非独立验证.csv"       # 调试模式再次查看的结果，文件名就写明不是独立验证

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("egg_train")


@dataclass(frozen=True)
class CVResult:
    scores: dict[str, np.ndarray]   # 方法名 -> (N, T) 判为雄的分数，来自这颗蛋当测试集的那一折；判断不了是 nan
    dev_pred: np.ndarray            # (N, T) 深度模型的发育预测
    fold_of_egg: np.ndarray         # (N,) 每颗蛋在第几折当测试集（从 0 数）
    best_epochs: tuple[int, ...]    # 每折深度模型最好的轮数
    n_train: float                  # 平均每折训练蛋数
    n_test: float                   # 平均每折测试蛋数
    n_fit: float                    # 平均每折深度模型真正用来拟合的蛋数（训练蛋里去掉早停验证的）

    @property
    def n_folds(self) -> int:
        return len(self.best_epochs)


# ---------- 准备 ----------
def parse_batches(text: str) -> tuple[str, ...]:
    batches = tuple(b.strip() for b in text.split(",") if b.strip())
    if not batches:
        raise argparse.ArgumentTypeError("--holdout-batches 至少要写一个批次，如 B10,B11")
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练种蛋模型：独立测试集 + 分组交叉验证 + 最终模型")
    parser.add_argument("--data", default=DATA_CSV, help="真实数据表 meta.csv 的路径；不填则用模拟数据")
    parser.add_argument("--decision-day", type=int, default=DECISION_DAY, help="必须出结果的孵化日")
    parser.add_argument("--shuffle-labels", action="store_true", help="流程自检：打乱性别标签后整套跑一遍")
    parser.add_argument("--holdout-batches", type=parse_batches, default=None,
                        help="明确指定新的独立测试批次，逗号分隔，如 B10,B11；必须是从没参加过开发的批次")
    parser.add_argument("--skip-holdout", action="store_true",
                        help="只重新训练，不评估独立测试集（测试批次仍然不参加训练，查看次数不变）")
    parser.add_argument("--debug-holdout", action="store_true",
                        help="调试模式：允许再次查看已经看过的独立测试集，结果一律标注“非独立验证”")
    parser.add_argument("--allow-no-batch", action="store_true",
                        help="允许数据没有批次信息（每颗蛋自成一组）；结果不能当作对新批次的性能")
    parser.add_argument("--fertility-check", action="store_true",
                        help="流水线检验：用伊利诺伊公开的真实平均光谱（标签是受精状态，不是性别）把整条流程跑一遍；"
                             "不保存判断新蛋用的模型")
    args = parser.parse_args()
    if args.fertility_check:
        if args.data:
            parser.error("--fertility-check 用的是固定的伊利诺伊数据，不能同时指定 --data")
        # 每颗蛋只有一条光谱、一个时间点（记作第 0 天）；锁文件放在数据文件旁边，和性别数据的互不相干
        args.data, args.decision_day = str(baseline_fertility.FERTILITY_CSV), 0
    if args.shuffle_labels and args.holdout_batches:
        parser.error("流程自检不使用独立测试集，不能同时指定 --holdout-batches")
    if args.skip_holdout and args.debug_holdout:
        parser.error("--skip-holdout（不看测试集）和 --debug-holdout（调试查看测试集）不能同时用")
    return args


def load_data(args: argparse.Namespace) -> EggDataset:
    if args.fertility_check:
        data = baseline_fertility.as_egg_dataset()
    else:
        data = load_real_dataset(args.data) if args.data else make_dataset(seed=SEED)
    if args.shuffle_labels:      # 按蛋打乱：每颗蛋拿到另一颗蛋的性别
        return replace(data, sex=np.random.default_rng(SEED).permutation(data.sex))
    return data


def subset(data: EggDataset, idx: np.ndarray) -> EggDataset:
    return replace(data, egg_ids=tuple(data.egg_ids[i] for i in idx), x=data.x[idx], spec=data.spec[idx],
                   mask=data.mask[idx], sex=data.sex[idx], dev=data.dev[idx], dev_mask=data.dev_mask[idx],
                   batch=data.batch[idx], bg_spec=None if data.bg_spec is None else data.bg_spec[idx])


def _fresh_holdout_batches(data: EggDataset) -> tuple[str, ...]:
    """第一次划分：按批次分层抽约 20% 当测试批次；批次太少返回空。"""
    split = holdout_split(data.sex, data.batch, SEED)
    return () if split is None else tuple(np.unique(data.batch[split[1]].astype(str)))


def _recorded_split(data: EggDataset, path: Path, batches: tuple[str, ...]) -> holdout_lock.SplitRecord | None:
    """锁文件里的批次划分；旧版本的锁只有查看次数、没有划分时，先补上划分（见 holdout_lock.recover_split）。"""
    record = holdout_lock.read_split(path)
    if record is not None or not holdout_lock.has_legacy_views(path):
        return record
    candidate = _fresh_holdout_batches(data)
    eggs = tuple(e for e, b in zip(data.egg_ids, data.batch.astype(str)) if b in candidate)
    record = holdout_lock.recover_split(path, batches, candidate, holdout_lock.holdout_hash(eggs))
    if record.holdout_batches:
        log.info(f"旧版本的查看记录没有存批次划分，已认出当年的独立测试批次：{'、'.join(record.holdout_batches)}")
    else:
        log.warning("旧版本的查看记录没有存批次划分，数据也变过，认不出当年的独立测试批次。为了不让训练过的蛋混进测试集，"
                    "现有批次全部记为开发批次；要独立测试，请用 --holdout-batches 指定从没用过的新批次")
    return record


def split_holdout(data: EggDataset, args: argparse.Namespace) -> tuple[EggDataset, EggDataset | None]:
    """返回 (开发集, 独立测试集)。划分记在锁文件里，以后沿用；流程自检不碰锁文件，只临时划一份。"""
    if args.shuffle_labels:
        split = holdout_split(data.sex, data.batch, SEED)
        return (data, None) if split is None else (subset(data, split[0]), subset(data, split[1]))
    path = holdout_lock.ensure_lock(args.data)
    batches = tuple(np.unique(data.batch.astype(str)))
    plan = holdout_lock.plan_split(_recorded_split(data, path, batches), batches, args.holdout_batches,
                                   lambda: _fresh_holdout_batches(data))
    holdout_lock.write_split(path, plan.record)
    log.info(f"批次划分（{plan.source}，记录在 {path}）：开发 {len(plan.dev)} 个批次"
             + (f"，其中新加入 {len(plan.new_dev)} 个（{'、'.join(plan.new_dev)}）" if plan.new_dev else ""))
    if not plan.holdout:
        log.warning("没有留独立测试集（批次少于 6 个，或以前的批次已用于开发、不能再当测试集）：性能只来自交叉验证，"
                    "而选方法也用了同一批结果，估计会偏乐观。有新批次时用 --holdout-batches 指定它们当测试集")
        return data, None
    in_hold = np.isin(data.batch.astype(str), plan.holdout)
    hold = subset(data, np.flatnonzero(in_hold))
    log.info(f"独立测试集：{len(plan.holdout)} 个批次（{'、'.join(plan.holdout)}）共 {len(hold.sex)} 颗蛋，"
             "开发阶段完全不碰，最后只测一次")
    return subset(data, np.flatnonzero(~in_hold)), hold


class HoldoutUsedError(Exception):
    """独立测试集已经被正式查看过，这次又要做正式评估：拦下，不训练也不评估。"""


def prior_holdout_views(hold: EggDataset, args: argparse.Namespace) -> int:
    """这个独立测试集以前被评估过几次（旧版本按蛋编号指纹存的记录先并进批次键）。"""
    path = holdout_lock.lock_path(args.data)
    key = holdout_lock.view_key(hold.batch)
    holdout_lock.merge_legacy(path, key, holdout_lock.holdout_hash(hold.egg_ids))
    return holdout_lock.prior_views(path, key)


def holdout_used_message(hold: EggDataset, prior: int, path: Path) -> str:
    batches = "、".join(sorted(map(str, np.unique(hold.batch)))[:holdout_lock.MAX_NAMES_IN_KEY])
    return (f"独立测试集（批次 {batches}）已经被查看过 {prior} 次（记录在 {path}），不能再做正式评估。"
            "这次运行已停止：没有训练，没有查看测试集，没有改动任何模型。可以这样继续：\n"
            "  ① 有从没参加过训练的新批次：加 --holdout-batches B10,B11，指定它们当新的独立测试集；\n"
            "  ② 只想重新训练、不看测试集：加 --skip-holdout（测试批次仍然不参加训练，查看次数不变）；\n"
            f"  ③ 只为调试要再看一眼：加 --debug-holdout，结果一律标注“{report.NOT_INDEPENDENT}”，不能写进报告或论文")


def guard_holdout(hold: EggDataset | None, args: argparse.Namespace) -> EggDataset | None:
    """训练开始前就决定这次能不能碰独立测试集（不等几分钟的交叉验证跑完才拦）。

    返回这次要评估的测试集；--skip-holdout 返回 None；已经看过又不是调试模式则抛 HoldoutUsedError。
    """
    if hold is None:
        return None
    if args.skip_holdout:
        log.info("这次不评估独立测试集（--skip-holdout）：测试批次仍然不参加训练，查看次数不变")
        return None
    prior = prior_holdout_views(hold, args)
    if prior and not args.debug_holdout:
        raise HoldoutUsedError(holdout_used_message(hold, prior, holdout_lock.lock_path(args.data)))
    return hold


def check_batches(data: EggDataset, args: argparse.Namespace) -> None:
    """真实数据必须有批次信息：缺了就没法按批次分组，"独立测试集"也只是随机抽的蛋，结果会偏乐观。
    默认直接报错；--allow-no-batch 才退回"每颗蛋自成一组"，并醒目提醒。"""
    if not args.data:
        return
    missing = eggs_without_batch(data)
    if not missing:
        return
    some = "、".join(missing[:5]) + ("……" if len(missing) > 5 else "")
    if not args.allow_no_batch:
        raise ValueError(f"{args.data} 里有 {len(missing)} 颗蛋没有填 batch（入孵批次），例如 {some}。没有批次信息，"
                         "就不能保证同一批次的蛋不同时出现在训练和测试两边，交叉验证和独立测试的结果都会偏乐观。"
                         "请在 meta.csv 里给每颗蛋补上 batch 列；确实没有批次信息、只想把流程跑通时，"
                         "加 --allow-no-batch（每颗蛋自成一组，结果不能当作对新批次的性能）")
    log.warning(f"!!! {len(missing)} 颗蛋没有批次信息（--allow-no-batch）：这些蛋每颗自成一组，同一批次的蛋可能同时出现在"
                "训练和测试两边，下面的结果不能当作对新批次的性能 !!!")


def decision_index(days: np.ndarray, decision_day: int) -> int:
    hits = np.flatnonzero(days == decision_day)
    if len(hits) == 0:
        raise ValueError(f"数据里没有第 {decision_day} 天，决策日只能选这些孵化日：{[int(d) for d in days]}")
    return int(hits[0])


def log_overview(data: EggDataset, args: argparse.Namespace, dev_weight: float) -> None:
    n_eggs, n_groups = len(data.sex), len(np.unique(data.batch))
    n_male = int(data.sex.sum())
    groups = "没有批次信息，每颗蛋自成一组" if n_groups == n_eggs else f"{n_groups} 个批次"
    log.info(f"数据：{args.data or '模拟数据'}")
    log.info(f"  {n_eggs} 颗蛋（雄 {n_male}，雌 {n_eggs - n_male}），{groups}，"
             f"孵化日 {[int(d) for d in data.days]}，{data.x.shape[2]} 个波段")
    log.info(f"决策日：第 {args.decision_day} 天（主指标只看这一天；当天没拍的蛋，"
             "深度-累计用之前几天的图像判断，单帧方法记为无法判断）")
    log.info("发育指标：" + ("有实测值，参与训练" if dev_weight > 0 else "全部留空，已关掉发育这一项"))
    if args.shuffle_labels:
        log.info("★ 流程自检：性别标签已随机打乱，下面所有准确率都应该在 0.5 左右")


# ---------- 交叉验证 ----------
def run_fold(data: EggDataset, plan: FoldPlan, dev_weight: float,
             seed: int) -> tuple[dict[str, np.ndarray], np.ndarray, int, int]:
    """一折：深度模型和基线在同一批测试蛋上打分。深度模型从训练部分里再分出早停验证，只用剩下的拟合；
    基线训练两遍：用这一折的全部训练蛋（正式比较，和部署时一致），以及只用深度模型拟合的那部分（同样本量对照）。
    返回 (各方法分数, 发育预测, 深度模型最好轮数, 拟合用的蛋数)。"""
    start = time.time()
    train_idx, test_idx, fit_idx, val_idx, grouped = plan.train, plan.test, plan.fit, plan.val, plan.grouped
    trained = dt.train_early_stop(data, fit_idx, val_idx, dev_weight, seed)
    deep = dt.predict_scores(trained, data.x[test_idx], data.mask[test_idx], data.days)
    base = per_day_baselines(data.spec, data.mask, data.sex, train_idx, test_idx)
    matched = per_day_baselines(data.spec, data.mask, data.sex, fit_idx, test_idx)
    log.info(f"  训练 {len(fit_idx)} 颗 + 早停验证 {len(val_idx)} 颗（{'按批次' if grouped else '按蛋随机'}分），"
             f"测试 {len(test_idx)} 颗；深度模型第 {trained.epochs} 轮最好；用时 {time.time() - start:.0f} 秒")
    scores = {"深度-单帧": deep["frame"], PRIMARY_DEEP: deep["cum"], **base,
              **{matched_name(name): s for name, s in matched.items()}}
    return scores, deep["dev"], trained.epochs, len(fit_idx)


def cross_validate(data: EggDataset, dev_weight: float, plans: list[FoldPlan]) -> CVResult:
    """plans 来自 isolation.plan_folds，训练前已经过隔离检查。"""
    folds = [(p.train, p.test) for p in plans]
    log.info(f"\n开发集 {len(data.sex)} 颗蛋，按批次分组做 {len(folds)} 折交叉验证")
    n_eggs, n_days = data.mask.shape
    scores = {name: np.full((n_eggs, n_days), np.nan) for name in (*DEEP_NAMES, *BASELINE_NAMES, *MATCHED_NAMES)}
    dev_pred = np.full((n_eggs, n_days), np.nan)
    fold_of_egg = np.full(n_eggs, -1)
    epochs, n_fit = [], []
    for k, plan in enumerate(plans):
        train_idx, test_idx = plan.train, plan.test
        log.info(f"第 {k + 1}/{len(folds)} 折")
        fold_scores, fold_dev, best_epoch, fit_size = run_fold(data, plan, dev_weight, SEED + k)
        for name, s in fold_scores.items():
            scores[name][test_idx] = s
        dev_pred[test_idx] = fold_dev
        fold_of_egg[test_idx] = k
        epochs.append(best_epoch)
        n_fit.append(fit_size)
    return CVResult(scores=scores, dev_pred=dev_pred, fold_of_egg=fold_of_egg, best_epochs=tuple(epochs),
                    n_fit=float(np.mean(n_fit)),
                    n_train=float(np.mean([len(tr) for tr, _ in folds])),
                    n_test=float(np.mean([len(te) for _, te in folds])))


# ---------- 结果 ----------
def evaluate(data: EggDataset, cv: CVResult, run_dir: Path) -> report.Metrics:
    """逐日指标：打印，并保存分折、逐日结果和每颗蛋的预测明细。"""
    metrics = {name: fold_day_metrics(s, data.sex, cv.fold_of_egg, cv.n_folds) for name, s in cv.scores.items()}
    report.log_per_day(data.days, metrics, "acc", "逐日准确率（各折平均±标准差；每颗蛋只在它当测试集的那一折被判断）")
    report.log_per_day(data.days, metrics, "coverage", "逐日覆盖率（给出判断的测试蛋占比）")
    report.log_per_day(data.days, metrics, "utility", "逐日决策效用（正确判断的蛋 / 全部测试蛋，判断不了算没判对）")
    report.log_per_day(data.days, metrics, "auc", "逐日 AUC（各折平均±标准差）")
    save_csv(run_dir / "folds.csv", report.fold_rows(data, cv.fold_of_egg))
    save_csv(run_dir / "results_per_day.csv", report.per_day_rows(data.days, metrics))
    save_csv(run_dir / "results_per_fold.csv", report.per_fold_rows(data.days, metrics))
    save_csv(run_dir / "predictions_oof.csv",
             report.prediction_rows(data, cv.scores, cv.dev_pred, cv.fold_of_egg))
    return metrics


def batch_report(data: EggDataset, cv: CVResult, day_idx: int, best_base: str, run_dir: Path) -> dict | None:
    """决策日按批次看：深度模型、最好的基线和它的同样本量版本，每个批次一个决策效用，以批次为单位给区间。
    只报告，不参与选择。没有批次信息（每颗蛋自成一组）时不做，返回 None。"""
    names, n_eggs = np.unique(data.batch.astype(str), return_counts=True)
    if len(names) == len(data.sex):
        log.info("\n按批次看的决策效用：没有批次信息（每颗蛋自成一组），没做")
        return None
    methods = (PRIMARY_DEEP, best_base, matched_name(best_base))
    utility = {m: batch_utilities(cv.scores[m][:, day_idx], data.sex, data.batch, names) for m in methods}
    diffs = {m: utility[PRIMARY_DEEP] - utility[m] for m in methods[1:]}
    report.log_per_batch(int(data.days[day_idx]), utility, diffs, PRIMARY_DEEP)
    save_csv(run_dir / "decision_day_per_batch.csv",
             report.per_batch_rows(names, n_eggs, utility, diffs, PRIMARY_DEEP))
    return report.per_batch_summary(utility, diffs, PRIMARY_DEEP)


def decide(data: EggDataset, cv: CVResult, metrics: report.Metrics, day_idx: int, run_dir: Path) -> dict:
    """决策日：深度模型（累计输出）和最好的基线比，只看事先规定的主指标 PRIMARY_METRIC（决策效用）。

    最好的基线 = 主指标平均最高的基线（用全部训练蛋训练的版本，和部署时一致）。共同可判断蛋上的准确率和覆盖率、
    同样本量对照、按批次的区间都只报告，不参与决定。
    """
    primary = {name: m[PRIMARY_METRIC][:, day_idx] for name, m in metrics.items()}
    best_base = max(BASELINE_NAMES, key=lambda name: np.nan_to_num(report.nanmean(primary[name]), nan=-1.0))
    cmp = compare_paired(primary[PRIMARY_DEEP], primary[best_base], cv.n_train, cv.n_test)
    common = {b: common_accuracy(cv.scores[PRIMARY_DEEP][:, day_idx], cv.scores[b][:, day_idx], data.sex,
                                 cv.fold_of_egg, cv.n_folds) for b in (*BASELINE_NAMES, *MATCHED_NAMES)}
    pooled = {name: pooled_utility(s[:, day_idx], data.sex) for name, s in cv.scores.items()}
    better = report.log_decision_day(int(data.days[day_idx]), day_idx, metrics, PRIMARY_DEEP, best_base, cmp,
                                     PRIMARY_METRIC, common, pooled)
    matched = matched_name(best_base)       # 同一种基线方法，只是训练样本量和深度模型拉平
    cmp_matched = compare_paired(primary[PRIMARY_DEEP], primary[matched], cv.n_fit, cv.n_test)
    better_matched = report.log_matched(PRIMARY_DEEP, matched, cmp_matched, PRIMARY_METRIC, better)
    save_csv(run_dir / "decision_day_comparison.csv", report.comparison_rows(metrics, day_idx, PRIMARY_DEEP, common))
    best = common[best_base]
    return {"主指标": PRIMARY_METRIC, "深度模型": PRIMARY_DEEP, "最好的基线": best_base,
            "同样本量对照": {"说明": "基线只用深度模型拟合用的那部分蛋训练（不含早停验证集）；只报告，不参与选择",
                             "基线": matched, "平均每折拟合蛋数": cv.n_fit, "平均每折训练蛋数": cv.n_train,
                             "基线决策效用": report.num_or_none(report.nanmean(primary[matched])),
                             "差值": report.num_or_none(cmp_matched.diff),
                             "差值95%区间": [report.num_or_none(cmp_matched.ci_low),
                                             report.num_or_none(cmp_matched.ci_high)],
                             "单侧p": report.num_or_none(cmp_matched.p), "深度模型明显更好": better_matched,
                             "与正式比较结论一致": better == better_matched},
            "按批次": batch_report(data, cv, day_idx, best_base, run_dir),
            "深度模型决策效用": report.nanmean(primary[PRIMARY_DEEP]), "基线决策效用": report.nanmean(primary[best_base]),
            "深度模型决策效用（合并按蛋）": pooled[PRIMARY_DEEP], "基线决策效用（合并按蛋）": pooled[best_base],
            "深度模型覆盖率": report.nanmean(metrics[PRIMARY_DEEP]["coverage"][:, day_idx]),
            "基线覆盖率": report.nanmean(metrics[best_base]["coverage"][:, day_idx]),
            "共同可判断蛋上深度模型准确率": report.nanmean(best.acc_a), "共同可判断蛋上基线准确率": report.nanmean(best.acc_b),
            "差值": cmp.diff, "差值95%区间": [cmp.ci_low, cmp.ci_high], "单侧p": cmp.p, "深度模型明显更好": better}


SELF_CHECK_PASS, SELF_CHECK_FAIL = "通过", "未通过"


def _log_self_check(statuses: dict[str, str], leak_consequence: str) -> None:
    if report.chance_passed(statuses):
        log.info("\n自检通过：每种方法都是“通过”，没有发现信息泄漏")
        return
    bad = "；".join(f"{name}：{status}" for name, status in statuses.items() if status != CHANCE_PASS)
    log.error(f"\n自检没通过（{bad}）。“报警”= 本该瞎猜的地方仍然判得准，说明测试数据的信息漏进了训练，{leak_consequence}；"
              f"“{CHANCE_UNKNOWN}”= 统计量算不出来（折数太少或各折结果完全一样），不能当作通过，请加大数据量后重做")


def self_check(data: EggDataset, cv: CVResult, metrics: report.Metrics,
               args: argparse.Namespace) -> dict[str, str] | None:
    """打乱标签时检查所有天；模拟数据正常训练时检查没有性别信号的第 0 天；真实数据正常训练不检查。

    返回 {方法: 通过/报警/无法确认}，没做检查返回 None。
    同样本量对照的基线只是报告用的对照，不进自检：免得多出来的方法凭空增加误报，把发布门控卡住。
    """
    metrics = {name: m for name, m in metrics.items() if name not in MATCHED_NAMES}
    if args.shuffle_labels:
        statuses = report.log_chance(metrics, list(range(len(data.days))), cv.n_train, cv.n_test,
                                     "流程自检：打乱标签后，每种方法所有孵化日合起来的准确率"
                                     "（各折平均±标准差，应接近 0.5）")
        _log_self_check(statuses, "正常训练的结果先别信")
        return statuses
    if not args.data and data.days[0] == 0:
        statuses = report.log_chance(metrics, [0], cv.n_train, cv.n_test,
                                     "流程自检：模拟数据第 0 天没有性别信号，每种方法这一天的准确率应接近 0.5")
        _log_self_check(statuses, "这次训练的结果先别信")
        return statuses
    return None


def check_record(statuses: dict[str, str] | None) -> dict | None:
    """写进 config.json 的自检结果。"""
    if statuses is None:
        return None
    return {"各方法": statuses, "总体": SELF_CHECK_PASS if report.chance_passed(statuses) else SELF_CHECK_FAIL}


def release_blockers(check: dict | None, controls: dict) -> list[str]:
    """模型发布门控：列出不能保存模型、不能做独立测试的原因；空列表 = 可以发布。

    自检和背景负对照都只认"通过"：报警和无法确认一律拦下（没做的检查不拦）。
    """
    reasons = []
    if check is not None and check["总体"] != SELF_CHECK_PASS:
        bad = "，".join(f"{name}：{status}" for name, status in check["各方法"].items() if status != CHANCE_PASS)
        reasons.append(f"流程自检未通过（{bad}）")
    background = controls.get("背景对照")
    if background is not None and background["结果"] != CHANCE_PASS:
        reasons.append(f"背景负对照{background['结果']}（只看背景猜性别的准确率 {background['平均准确率']}）")
    return reasons


def run_config(args: argparse.Namespace, data: EggDataset, dev_weight: float, cv: CVResult) -> dict:
    return {
        "时间": datetime.now().isoformat(timespec="seconds"),
        "数据": args.data or "模拟数据", "流程自检（打乱标签）": args.shuffle_labels,
        "开发集蛋数": len(data.sex), "开发集批次数": int(len(np.unique(data.batch))),
        "孵化日": [int(d) for d in data.days], "波段数": int(data.x.shape[2]),
        "决策日": args.decision_day, "交叉验证折数": cv.n_folds, "随机种子": SEED,
        "各折最好轮数": list(cv.best_epochs),
        "深度模型": {"BATCH_SIZE": dt.BATCH_SIZE, "MAX_EPOCHS": dt.MAX_EPOCHS, "PATIENCE": dt.PATIENCE,
                     "LR": dt.LR, "WEIGHT_DECAY": dt.WEIGHT_DECAY, "发育损失权重": dev_weight},
        "基线": {"方法": list(BASELINE_NAMES), "PLS成分数": PLS_COMPONENTS, "逻辑回归C": LOGREG_C,
                 "光谱预处理": "SNV + 标准化"},
        "图像预处理": preprocess_settings(),
        "主指标": PRIMARY_METRIC,
        "环境": environment(),
    }


def run_holdout(final: deploy.FinalModels, hold: EggDataset, args: argparse.Namespace, day_idx: int,
                chosen: str, run_dir: Path) -> dict:
    """独立测试集评估：只有第一次查看是正式结果。已经看过的测试集默认拦下（HoldoutUsedError）；
    调试模式（--debug-holdout）可以再看，但所有输出都标注"非独立验证"，仍然记一次查看。返回写进配置的内容。"""
    path = holdout_lock.lock_path(args.data)
    egg_hash = holdout_lock.holdout_hash(hold.egg_ids)
    ids_hash = holdout_lock.view_key(hold.batch)
    prior = prior_holdout_views(hold, args)
    if prior and not args.debug_holdout:
        raise HoldoutUsedError(holdout_used_message(hold, prior, path))
    changed = holdout_lock.egg_change_note(path, ids_hash, egg_hash, len(hold.egg_ids))
    if prior and changed:
        log.warning(f"\n!!! 独立测试批次里的蛋和上次查看时不一样（{changed}）：测试集被改过，查看次数仍然累计 !!!")
    n_batches = len(np.unique(hold.batch))
    formal = prior == 0
    if not formal:
        log.warning(f"\n!!! 调试模式：这个独立测试集已经被查看过 {prior} 次（记录在 {path}）。它不再独立：之前的结果可能已经"
                    f"影响了后来的改动和选择。这次的结果一律标注“{report.NOT_INDEPENDENT}”，不能写进报告或论文；"
                    "正式检验请用新的批次 !!!")
    if n_batches < MIN_HOLDOUT_BATCHES:
        log.warning(f"独立测试集只有 {n_batches} 个批次（少于 {MIN_HOLDOUT_BATCHES} 个），批次之间的差异没法估计，"
                    "结果只能当参考")
    views = prior + 1
    rows, tables = deploy.evaluate_holdout(final, hold, day_idx, chosen, views, formal)
    save_csv(run_dir / (HOLDOUT_CSV if formal else HOLDOUT_DEBUG_CSV), rows)
    total = tables[chosen][0]
    holdout_lock.record_view(path, ids_hash, run_dir.name, chosen,
                             {"蛋数": total.n_eggs, "判断数": total.judged, "判对数": total.correct,
                              "覆盖率": report.num_or_none(total.coverage), "决策效用": report.num_or_none(total.utility)},
                             n_eggs=len(hold.egg_ids), egg_hash=egg_hash, formal=formal)
    return {"蛋数": len(hold.sex), "批次": sorted(map(str, np.unique(hold.batch))), "holdout_views": views,
            "正式结果": formal, "性质": report.FORMAL_RESULT if formal else report.NOT_INDEPENDENT,
            "区间说明": WILSON_LABEL, "按批次决策效用": report.holdout_batch_interval(tables[chosen])}


def finish(data: EggDataset, hold: EggDataset | None, cv: CVResult, metrics: report.Metrics,
           args: argparse.Namespace, dev_weight: float, run_dir: Path) -> dict:
    """正常训练的收尾：决策日选方法、发育报告、自检、最终模型、独立测试。返回写进配置的内容。"""
    day_idx = decision_index(data.days, args.decision_day)
    result = decide(data, cv, metrics, day_idx, run_dir)
    chosen = PRIMARY_DEEP if result["深度模型明显更好"] else result["最好的基线"]
    save_csv(run_dir / "dev_mae.csv", report.log_dev(data, cv.dev_pred))
    extra = {"决策日结果": result, "部署方法": chosen, "自检": check_record(self_check(data, cv, metrics, args)),
             "负对照": run_controls(data, cv.fold_of_egg, day_idx, cv.n_train, cv.n_test)}
    blockers = release_blockers(extra["自检"], extra["负对照"])
    extra["模型发布"] = {"已发布": not blockers, "未发布原因": blockers}
    if blockers:        # 发布门控：检查没通过就到此为止，不训练最终模型、不碰独立测试集、不写任何模型文件
        log.error(f"\n!!! 质量检查没通过，这次不保存模型，也不做独立测试（{'；'.join(blockers)}）。"
                  "output 文件夹里上一次通过检查的模型原样保留；独立测试集没有被查看，查看次数不变。"
                  "这次的运行文件夹里只有诊断记录 !!!")
        return extra
    final = deploy.train_final(data, cv.best_epochs, dev_weight, result["最好的基线"], SEED)
    if hold is not None:
        extra["独立测试集"] = run_holdout(final, hold, args, day_idx, chosen, run_dir)
        extra["holdout_views"] = extra["独立测试集"]["holdout_views"]
    deploy.save(final, data, chosen, args.decision_day, run_dir, publish=not args.fertility_check)
    return extra


def exit_code(check: dict | None) -> int:
    """没做自检或通过为 0；没通过（报警或无法确认）为 EXIT_SELF_CHECK_FAILED（程序出错另有退出码 1）。"""
    return EXIT_SELF_CHECK_FAILED if check is not None and check["总体"] != SELF_CHECK_PASS else 0


def main() -> int:
    """返回退出码：0 正常；自检或背景负对照没通过为 2（不保存模型，.bat 和窗口界面据此显示"自检未通过"）；
    独立测试集已经看过、被拦下为 3。"""
    args = parse_args()
    kind = "流程自检" if args.shuffle_labels else ("真实数据" if args.data else "模拟数据")
    if args.fertility_check:
        kind = "真实光谱自检" if args.shuffle_labels else "真实光谱检验"
    run_dir = new_run_dir(kind)
    attach_log(run_dir)
    set_determinism(SEED)
    if args.fertility_check:
        log.warning(FERTILITY_NOTE)
    data_all = load_data(args)
    repro = reproducibility(run_dir, data_all, args.data)
    check_batches(data_all, args)
    decision_index(data_all.days, args.decision_day)
    dev_weight = dt.DEV_LOSS_WEIGHT if data_all.dev_mask.any() else 0.0
    log_overview(data_all, args, dev_weight)
    data, hold = split_holdout(data_all, args)
    if not args.shuffle_labels:
        try:
            hold = guard_holdout(hold, args)
        except HoldoutUsedError as e:
            log.error(f"\n!!! 已拦截：{e}")
            save_json(run_dir / "config.json", {"时间": datetime.now().isoformat(timespec="seconds"),
                                                "数据": args.data or "模拟数据", "已拦截": str(e),
                                                "复现": repro})
            return EXIT_HOLDOUT_USED

    plans = plan_folds(data, N_FOLDS, SEED)
    try:
        isolation = audit(data, hold, plans)
    except IsolationError as e:
        log.error(f"\n!!! {e}")
        save_json(run_dir / "config.json", {"时间": datetime.now().isoformat(timespec="seconds"),
                                            "数据": args.data or "模拟数据", "隔离检查": str(e), "复现": repro})
        raise
    log.info(f"数据隔离检查通过：独立测试集、{len(plans)} 折交叉验证、每折的早停验证之间没有共同的蛋或批次")
    for note in isolation["提醒"]:
        log.warning(f"  提醒：{note}")
    cv = cross_validate(data, dev_weight, plans)
    metrics = evaluate(data, cv, run_dir)
    config = run_config(args, data, dev_weight, cv)
    config["复现"] = repro
    config["隔离检查"] = isolation
    if args.shuffle_labels:
        config["自检"] = check_record(self_check(data, cv, metrics, args))
    else:
        config.update(finish(data, hold, cv, metrics, args, dev_weight, run_dir))
    if args.fertility_check:
        config["标签含义"] = FERTILITY_NOTE
        log.warning(f"\n{FERTILITY_NOTE}")
    save_json(run_dir / "config.json", config)
    log.info(f"\n这次运行的全部记录在：{run_dir}")
    if config.get("模型发布", {}).get("已发布") is False:      # 背景负对照没通过时自检可能是通过的，同样算没通过
        return EXIT_SELF_CHECK_FAILED
    return exit_code(config.get("自检"))


if __name__ == "__main__":
    sys.exit(main())
