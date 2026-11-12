"""数据隔离检查：训练开始前，把这次要用的全部划分（独立测试集、外层各折、每折里的早停验证）摆出来逐条核对，
确认同一颗蛋、同一个批次不会跨过隔离边界。有一条不对就报错停下（退出码 1），不会白跑几个小时才发现。

核对的边界：
- 独立测试集 和 开发集：没有共同的蛋，也没有共同的批次。
- 外层每一折的 训练 和 测试：没有共同的蛋、没有共同的批次；每颗开发蛋恰好当一次测试。
- 每一折里的 拟合 和 早停验证：合起来正好是这一折的训练部分，没有共同的蛋，都不含这一折的测试蛋；
  批次够多时按批次分，此时也没有共同的批次。批次太少时只能按蛋随机分，同一批次的蛋可能一边拟合一边早停验证，
  早停轮数的选择会受同批次相关性影响（可能偏乐观），外层测试的独立性不受影响。
  正式模式（真实数据，strict_val=True）下这算问题、直接停下：早停验证不能证明对新批次的泛化，
  批次不够就该补批次，而不是悄悄换成按蛋分；确实要跑就加 --allow-ungrouped-val 明确接受。
  演示模式（模拟数据）只记一条提醒。
蛋是最小单位：同一颗蛋所有孵化日的图像总是一起走，所以"同一颗蛋不跨边界"就包括了"同一颗蛋的不同天数不跨边界"。

预处理参数（标准化的均值和标准差、基线的 SNV + 标准化）、超参数和模型都只在训练那一边拟合，
这一点由 tests/test_isolation.py 用"在测试蛋里放毒"的方式检查。
"""
from dataclasses import dataclass

import numpy as np

from evaluation import MIN_GROUPS_FOR_GROUPED_VAL, inner_split, outer_folds
from synth_data import EggDataset

PASS = "通过"


class IsolationError(Exception):
    """有蛋或批次跨过了隔离边界。"""


@dataclass(frozen=True)
class FoldPlan:
    """外层一折的全部划分（都是开发集里的下标）。"""
    train: np.ndarray       # 这一折的训练部分 = 拟合 + 早停验证
    test: np.ndarray
    fit: np.ndarray         # 深度模型真正用来拟合的
    val: np.ndarray         # 深度模型的早停验证
    grouped: bool           # 早停验证是不是按批次分的


def plan_folds(data: EggDataset, n_folds: int, seed: int) -> list[FoldPlan]:
    """外层分折和每折的早停验证划分。第 k 折用种子 seed + k，训练和消融都从这里拿，保证用的是同一套划分。"""
    plans = []
    for k, (train, test) in enumerate(outer_folds(data.sex, data.batch, n_folds, seed)):
        fit, val, grouped = inner_split(train, data.sex, data.batch, seed + k)
        plans.append(FoldPlan(train, test, fit, val, grouped))
    return plans


def _names(values: object) -> set[str]:
    return {str(v) for v in np.asarray(values, dtype=object).ravel()}


def _shared(a: set[str], b: set[str], what: str, where: str, problems: list[str]) -> None:
    common = sorted(a & b)
    if common:
        shown = "、".join(common[:5]) + ("……" if len(common) > 5 else "")
        problems.append(f"{where}有 {len(common)} 个共同的{what}（{shown}）")


def _check_unique(egg_ids: tuple[str, ...], name: str, problems: list[str]) -> None:
    seen, dup = set(), set()
    for e in egg_ids:
        (dup if e in seen else seen).add(e)
    if dup:
        problems.append(f"{name}里蛋编号重复：{'、'.join(sorted(dup)[:5])}")


def _check_fold(k: int, p: FoldPlan, ids: np.ndarray, batch: np.ndarray, problems: list[str],
                notes: list[str], strict_val: bool = False) -> dict:
    where = f"第 {k + 1} 折"
    _shared(set(ids[p.train]), set(ids[p.test]), "蛋", f"{where}的训练和测试之间", problems)
    _shared(_names(batch[p.train]), _names(batch[p.test]), "批次", f"{where}的训练和测试之间", problems)
    if sorted(np.concatenate([p.fit, p.val]).tolist()) != sorted(p.train.tolist()):
        problems.append(f"{where}的拟合 + 早停验证不等于训练部分")
    _shared(set(ids[p.fit]), set(ids[p.val]), "蛋", f"{where}的拟合和早停验证之间", problems)
    _shared(set(ids[p.val]), set(ids[p.test]), "蛋", f"{where}的早停验证和测试之间", problems)
    if p.grouped:
        _shared(_names(batch[p.fit]), _names(batch[p.val]), "批次", f"{where}的拟合和早停验证之间", problems)
    elif strict_val:
        problems.append(f"{where}的训练部分只有 {len(_names(batch[p.train]))} 个批次（少于 {MIN_GROUPS_FOR_GROUPED_VAL} 个），"
                        "早停验证没法按批次分，只能按蛋随机分。正式模式不允许这样：同批次的蛋一边拟合一边验证，"
                        "早停结果不能当作对新批次的泛化。请增加批次；确实要跑，加 --allow-ungrouped-val 明确接受")
    else:
        notes.append(f"{where}的训练部分批次太少，早停验证按蛋随机分：同一批次的蛋可能一边拟合一边早停验证，"
                     "早停轮数可能偏乐观（不影响测试结果的独立性）")
    return {"折": k + 1, "训练蛋数": len(p.train), "测试蛋数": len(p.test),
            "训练批次数": len(_names(batch[p.train])), "测试批次数": len(_names(batch[p.test])),
            "拟合蛋数": len(p.fit), "早停验证蛋数": len(p.val), "早停验证": "按批次" if p.grouped else "按蛋随机"}


def audit(dev: EggDataset, hold: EggDataset | None, plans: list[FoldPlan], strict_val: bool = False) -> dict:
    """逐条核对；全部通过返回写进 config.json 的记录，否则抛 IsolationError 并列出每一处问题。
    strict_val=True（正式模式）时，早停验证没能按批次分也算问题。"""
    problems: list[str] = []
    notes: list[str] = []
    ids = np.array(dev.egg_ids, dtype=object)
    _check_unique(dev.egg_ids, "开发集", problems)
    record: dict = {}
    if hold is not None:
        _check_unique(hold.egg_ids, "独立测试集", problems)
        _shared(set(dev.egg_ids), set(hold.egg_ids), "蛋", "独立测试集和开发集之间", problems)
        _shared(_names(dev.batch), _names(hold.batch), "批次", "独立测试集和开发集之间", problems)
        record["独立测试集"] = {"蛋数": len(hold.egg_ids), "批次数": len(_names(hold.batch))}
    else:
        record["独立测试集"] = None
    tested = np.concatenate([p.test for p in plans]) if plans else np.array([], dtype=int)
    if sorted(tested.tolist()) != list(range(len(ids))):
        problems.append("外层各折的测试部分合起来不是每颗开发蛋恰好一次")
    folds = [_check_fold(k, p, ids, dev.batch, problems, notes, strict_val) for k, p in enumerate(plans)]
    if problems:
        raise IsolationError("数据隔离检查没通过，训练没有开始：\n  " + "\n  ".join(problems))
    if len(_names(dev.batch)) == len(ids):
        notes.append("数据没有批次信息（每颗蛋自成一组）：只保证同一颗蛋不跨边界，同一批次的蛋可能分在两边")
    return {"结果": PASS, "开发集": {"蛋数": len(ids), "批次数": len(_names(dev.batch))}, **record,
            "各折": folds, "提醒": notes}
