"""用训练好的模型判断新的蛋。

先训练得到 output/egg_model.pt，再双击 2-判断新蛋.bat（或 py -3.11 predict.py --data 新蛋的meta.csv）。
新蛋的数据表只需要 egg_id、day、cube 三列，white / dark 按需填写，不需要性别；
wavelengths：训练数据有波长轴，新数据就必须有（反之亦然），两边都没有才可以不填。
图像的处理（校正、找蛋、裁剪缩放）和判断规则都和训练时完全一样（预处理设置存在模型里，不一致会直接报错）：
- "判断"用训练时决策规则选定的方法（深度模型没有明显好过基线时，用基线）；
- 深度-累计在当天没拍时用之前几天的图像判断；单帧方法和基线当天没拍就"无法判断"。
分类分数大于 0.5 判雄。深度模型的分数是模型给的概率；PLS-DA 的是回归分数，可能小于 0 或大于 1，不是概率。
结果：output/predictions_decision_day.csv（每颗蛋一行，决策日的判断）和 output/predictions.csv（每天一行）。
"""
import argparse
import logging
import pickle

import numpy as np
import torch

import deep_train as dt
from baselines import baseline_scores
from model import EggSeqModel
from paths import BASELINE_MODEL_PATH, DECISION_CSV, MODEL_PATH, PREDICTIONS_CSV, SAMPLE_CSV
from preprocess_binding import check_preprocess
from real_data import WAVELENGTH_TOLERANCE, EggFrames, load_unlabeled
from run_record import save_csv

PREDICT_CSV = SAMPLE_CSV   # 换成新蛋数据表的路径，例如 r"D:\新蛋\meta.csv"
SEX_NAMES = ("雌", "雄")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("predict")


def load_model() -> tuple[dt.TrainedModel, dict]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 {MODEL_PATH}，请先双击 1-训练模型.bat 训练模型")
    ckpt = torch.load(MODEL_PATH, weights_only=True)
    model = EggSeqModel(n_bands=ckpt["n_bands"])
    try:
        model.load_state_dict(ckpt["state_dict"])
    except RuntimeError:
        raise RuntimeError(f"{MODEL_PATH} 是旧版本的模型（网络结构已更新），请先重新训练") from None
    check_preprocess(ckpt.get("preprocess"), str(MODEL_PATH))
    norm = dt.Normalizer(ckpt["mean"].numpy(), ckpt["std"].numpy())
    return dt.TrainedModel(model, norm, int(ckpt.get("epochs", 0))), ckpt


def load_baseline(ckpt: dict) -> dict:
    """读训练时保存的基线模型，并确认它和深度模型来自同一次训练。"""
    if not BASELINE_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到 {BASELINE_MODEL_PATH}，请重新训练")
    with BASELINE_MODEL_PATH.open("rb") as f:
        bundle = pickle.load(f)     # 只读本程序训练时自己保存的文件
    if bundle["run"] != ckpt["run"]:
        raise ValueError("基线模型和深度模型不是同一次训练保存的，请重新训练")
    check_preprocess(bundle.get("preprocess"), str(BASELINE_MODEL_PATH))
    return bundle


def check_wavelengths(frames: EggFrames, ckpt: dict) -> None:
    """波长轴：两边都有就必须一致；只有一边有就报错（没法确认波段对得上）；两边都没有（模拟数据）只提示。"""
    trained, new = ckpt.get("wavelengths"), frames.wavelengths
    if trained is None and new is None:
        log.info("训练数据和新数据都没有波长轴（模拟数据），无法核对波段是否对得上")
        return
    if trained is None or new is None:
        have = "新数据有、训练数据没有" if trained is None else "训练数据有、新数据没有"
        raise ValueError(f"波长轴只有一边提供（{have}），无法确认波段对得上。"
                         "训练数据和新数据必须都提供波长文件（meta.csv 的 wavelengths 列），或者都不提供")
    trained = np.asarray(trained)
    if trained.shape != new.shape or np.abs(trained - new).max() > WAVELENGTH_TOLERANCE:
        raise ValueError("新数据的波长轴和训练时不一致（波段数相同不代表波长相同），不能用这个模型判断")


def align_days(frames: EggFrames, model_days: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把新数据的孵化日对齐到训练时的孵化日；没训练过的孵化日直接报错。返回 (x, spec, mask)。"""
    unknown = sorted(set(frames.days.tolist()) - set(model_days))
    if unknown:
        raise ValueError(f"孵化日 {unknown} 在训练数据里没有，模型无法判断。训练时的孵化日：{model_days}")
    n, _, c, h, w = frames.x.shape
    x = np.zeros((n, len(model_days), c, h, w), dtype=np.float32)
    spec = np.zeros((n, len(model_days), c), dtype=np.float32)
    mask = np.zeros((n, len(model_days)), dtype=bool)
    for t_new, day in enumerate(frames.days.tolist()):
        t = model_days.index(day)
        x[:, t], spec[:, t], mask[:, t] = frames.x[:, t_new], frames.spec[:, t_new], frames.mask[:, t_new]
    return x, spec, mask


def score(frames: EggFrames, trained: dt.TrainedModel, ckpt: dict) -> tuple[str, np.ndarray, dict, np.ndarray]:
    """返回 (采用方法, 采用方法的分数 (N, T), 深度模型各输出, mask)。"""
    check_wavelengths(frames, ckpt)
    x, spec, mask = align_days(frames, ckpt["days"])
    if x.shape[2] != ckpt["n_bands"]:
        raise ValueError(f"波段数 {x.shape[2]} 和训练时的 {ckpt['n_bands']} 不一致")
    out = dt.predict_scores(trained, x, mask, np.array(ckpt["days"], dtype=np.float32))
    chosen = ckpt.get("chosen", dt.PRIMARY_DEEP)  # 旧版模型没有这一项，当作用深度模型
    main = out["cum"] if chosen == dt.PRIMARY_DEEP else \
        baseline_scores(load_baseline(ckpt)["models"], spec, mask, ckpt["days"])
    return chosen, main, out, mask


def _judge(value: float) -> str:
    return "无法判断" if np.isnan(value) else SEX_NAMES[int(value > 0.5)]


def _fmt(value: float) -> str:
    return "" if np.isnan(value) else f"{value:.2f}"


def daily_rows(frames: EggFrames, ckpt: dict, chosen: str, main: np.ndarray, out: dict,
               mask: np.ndarray) -> list[dict[str, str]]:
    """每颗蛋每天一行（拍过的天，以及深度-累计能用之前图像判断的天）。"""
    has_dev = bool(ckpt.get("has_dev", True))
    rows = []
    for i, egg in enumerate(frames.egg_ids):
        for t, day in enumerate(ckpt["days"]):
            if not mask[i, t] and np.isnan(main[i, t]) and np.isnan(out["cum"][i, t]):
                continue
            rows.append({
                "egg_id": egg, "孵化日": str(int(day)), "当天有图像": "是" if mask[i, t] else "否",
                "采用方法": chosen, "判断": _judge(main[i, t]), "分类分数": _fmt(main[i, t]),
                "深度-单帧判断": _judge(out["frame"][i, t]), "深度-单帧雄概率": _fmt(out["frame"][i, t]),
                "深度-累计判断": _judge(out["cum"][i, t]), "深度-累计雄概率": _fmt(out["cum"][i, t]),
                "发育预测": _fmt(out["dev"][i, t]) if has_dev else "未训练",
            })
    return rows


def decision_rows(frames: EggFrames, ckpt: dict, chosen: str, main: np.ndarray,
                  mask: np.ndarray) -> tuple[int, list[dict[str, str]]]:
    """每颗蛋一行：决策日的判断（和验证时同一规则）。旧版模型没记决策日，就用最后一个孵化日。"""
    day = float(ckpt.get("decision_day", ckpt["days"][-1]))
    t = ckpt["days"].index(day)
    rows = []
    for i, egg in enumerate(frames.egg_ids):
        reason = ""
        if np.isnan(main[i, t]):
            reason = ("决策日及之前都没有图像" if chosen == dt.PRIMARY_DEEP
                      else "决策日没有图像（基线只能用当天的图像）")
        rows.append({"egg_id": egg, "决策日": str(int(day)), "决策日有图像": "是" if mask[i, t] else "否",
                     "采用方法": chosen, "判断": _judge(main[i, t]), "分类分数": _fmt(main[i, t]),
                     "无法判断的原因": reason})
    return int(day), rows


def main() -> None:
    parser = argparse.ArgumentParser(description="用训练好的模型判断新蛋")
    parser.add_argument("--data", default=str(PREDICT_CSV), help="新蛋数据表 meta.csv 的路径")
    data_csv = parser.parse_args().data

    trained, ckpt = load_model()
    log.info(f"判断的数据：{data_csv}\n")
    frames = load_unlabeled(data_csv)
    chosen, main_scores, out, mask = score(frames, trained, ckpt)
    save_csv(PREDICTIONS_CSV, daily_rows(frames, ckpt, chosen, main_scores, out, mask))
    day, rows = decision_rows(frames, ckpt, chosen, main_scores, mask)
    save_csv(DECISION_CSV, rows)

    judged = sum(r["判断"] != "无法判断" for r in rows)
    log.info(f"决策日（第 {day} 天）的判断，采用方法：{chosen}（分类分数大于 0.5 判雄）")
    log.info("蛋编号    当天有图像  判断      分类分数")
    for r in rows:
        log.info(f"{r['egg_id']:<10}{r['决策日有图像']:>6}      {r['判断']:<8}  {r['分类分数']}")
    log.info(f"\n共 {len(rows)} 颗蛋，决策日给出判断 {judged} 颗（覆盖率 {judged / len(rows):.0%}）")
    log.info(f"决策日结果：{DECISION_CSV}\n逐日详细结果：{PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()
