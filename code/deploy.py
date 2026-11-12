"""定稿：用开发数据训练最终模型，在独立测试集上只测一次，保存部署用的文件。

两种候选方法都训练：深度模型（累计输出）和事先指定的主比较基线（baselines.PRIMARY_BASELINE）。
部署用哪一种由交叉验证阶段的决策规则事先定好（chosen），独立测试集只评估它一种、只报告它的成绩，不再参与挑选。
"""
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.pipeline import Pipeline

import deep_train as dt
import report
from baselines import baseline_scores, fit_final_baselines
from deep_train import PRIMARY_DEEP
from evaluation import ScopeStats, holdout_table
from paths import BASELINE_MODEL_PATH, MODEL_PATH
from preprocess_binding import preprocess_settings
from synth_data import EggDataset

FEW_EPOCHS = 5              # 最好的轮数少于它，说明模型几乎没学到东西，要提醒

log = logging.getLogger("egg_train")


@dataclass(frozen=True)
class FinalModels:
    deep: dt.TrainedModel
    baseline: dict[float, Pipeline]     # 孵化日 -> 基线模型
    base_name: str


def train_final(data: EggDataset, best_epochs: tuple[int, ...], dev_weight: float,
                base_name: str, seed: int) -> FinalModels:
    """深度模型训练固定轮数（各折最好轮数的中位数）；基线每个孵化日训练一个。"""
    epochs = max(1, int(np.median(best_epochs)))
    log.info(f"\n最终模型：用开发集全部 {len(data.sex)} 颗蛋训练，深度模型 {epochs} 轮（各折最好轮数的中位数），"
             f"基线 {base_name} 每个孵化日一个……")
    if epochs < FEW_EPOCHS:
        log.warning(f"  注意：各折最好的轮数只有 {list(best_epochs)}，深度模型几乎刚开始学就停了，"
                    "多半是数据太少或信号太弱，这个深度模型不可靠")
    deep = dt.train_fixed_epochs(data, np.arange(len(data.sex)), dev_weight, seed, epochs)
    models = fit_final_baselines(data.spec, data.mask, data.sex, base_name)
    return FinalModels(deep, {float(data.days[t]): m for t, m in models.items()}, base_name)


def chosen_scores(final: FinalModels, data: EggDataset, chosen: str) -> np.ndarray:
    """选定方法按部署规则给出的判为雄分数 (N, T)，判断不了的位置是 nan。"""
    if chosen == PRIMARY_DEEP:
        return dt.predict_scores(final.deep, data.x, data.mask, data.days)["cum"]
    if chosen == final.base_name:
        return baseline_scores(final.baseline, data.spec, data.mask, list(data.days))
    raise ValueError(f"选定的方法 {chosen!r} 既不是 {PRIMARY_DEEP} 也不是 {final.base_name}")


def evaluate_holdout(final: FinalModels, hold: EggDataset, day_idx: int, chosen: str,
                     views: int, formal: bool = True) -> tuple[list[dict], dict[str, list[ScopeStats]]]:
    """独立测试集上只测一次决策日（总体和每个批次），而且只评估事先选定的方法：
    没选中的方法不打开成绩，免得看到它分数更高后回头改选择。返回保存用的行和成绩表。
    formal=False：调试模式再次查看已经看过的测试集，所有输出标注"非独立验证"。"""
    tables = {chosen: holdout_table(chosen_scores(final, hold, chosen)[:, day_idx], hold.sex, hold.batch)}
    day = int(hold.days[day_idx])
    report.log_holdout(day, tables, chosen, views, formal)
    return report.holdout_rows(day, tables, chosen, views, formal), tables


def save(final: FinalModels, data: EggDataset, chosen: str, decision_day: int, run_dir: Path,
         publish: bool = True) -> None:
    """深度模型连同标准化参数、孵化日、波长轴、预处理设置、部署方法一起存；选了基线时另存基线模型。
    publish=False：只在运行文件夹里留一份，不碰 output 里判断新蛋用的模型（真实光谱流水线检验用，它的标签不是性别）。"""
    trained = final.deep
    ckpt = {"state_dict": trained.model.state_dict(), "n_bands": int(data.x.shape[2]),
            "mean": torch.from_numpy(trained.norm.mean), "std": torch.from_numpy(trained.norm.std),
            "days": data.days.tolist(), "has_dev": bool(data.dev_mask.any()),
            "decision_day": decision_day, "preprocess": preprocess_settings(), "epochs": trained.epochs, "run": run_dir.name, "chosen": chosen,
            "wavelengths": None if data.wavelengths is None else data.wavelengths.tolist()}
    if publish:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_paths = [run_dir / "egg_model.pt"] + ([MODEL_PATH] if publish else [])
    for path in model_paths:
        torch.save(ckpt, path)
    log.info(f"最终深度模型已保存到 {model_paths[-1]}" + ("" if publish else "（只留在运行文件夹里，不用于判断新蛋）"))
    if chosen == PRIMARY_DEEP:
        return
    bundle = {"name": final.base_name, "run": run_dir.name, "models": final.baseline,
              "preprocess": ckpt["preprocess"]}
    base_paths = [run_dir / "baseline_model.pkl"] + ([BASELINE_MODEL_PATH] if publish else [])
    for path in base_paths:
        with path.open("wb") as f:
            pickle.dump(bundle, f)
    if publish:
        log.info(f"判断新蛋将使用 {final.base_name}（每个孵化日一个模型），已保存到 {BASELINE_MODEL_PATH}")
    else:
        log.info(f"选定的方法是 {final.base_name}，模型只留在运行文件夹里：{base_paths[0]}")
