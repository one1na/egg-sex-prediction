"""独立测试集的记录：哪些批次是独立测试集、哪些进过开发集，以及每次被查看的情况。

锁文件（真实数据在 meta.csv 旁，模拟数据在 output 文件夹，都不在可随时删除的 _cache 里）有两部分：
- split：第一次留出测试集时记下的"独立测试批次"和"开发批次"。以后每次训练都沿用这个划分，
  新加进来的批次一律进开发集，以前训练过的批次不会再混进测试集；开发批次的名单只增不减。
- sets：按测试批次名单分别记着每个独立测试集的每次评估（时间、运行文件夹、选定方法、成绩）；
  批次里增减蛋不清零，只提醒。换过测试集再换回来，之前的记录仍然算数。
旧版本的锁只有查看次数、没有 split 时，第一次使用会补上划分（认不出当年的测试批次就全部算开发批次）。
删掉锁文件就等于清零，请不要这样做。
"""
import hashlib
import json
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from paths import CACHE_DIRNAME, HOLDOUT_LOCK_NAME, SYNTH_HOLDOUT_LOCK

SETS_KEY = "sets"
SPLIT_KEY = "split"
SOURCE_FIRST, SOURCE_REUSED, SOURCE_REQUESTED, SOURCE_NONE = "首次划分", "沿用记录", "手工指定", "没有测试集"

log = logging.getLogger("egg_train")


@dataclass(frozen=True)
class SplitRecord:
    holdout_batches: tuple[str, ...]    # 独立测试批次（没留测试集时为空）
    dev_batches: tuple[str, ...]        # 历史上进过开发集的全部批次（只增不减）
    holdout_viewed: bool = False        # 这个测试集是否已经被评估过


@dataclass(frozen=True)
class SplitPlan:
    holdout: tuple[str, ...]
    dev: tuple[str, ...]
    new_dev: tuple[str, ...]            # 这次新加入开发集的批次
    source: str
    record: SplitRecord                 # 划分好之后应该写进锁文件的内容


def lock_path(data_csv: str | None) -> Path:
    if data_csv:
        return Path(data_csv).parent / HOLDOUT_LOCK_NAME
    return SYNTH_HOLDOUT_LOCK


def ensure_lock(data_csv: str | None) -> Path:
    """返回锁文件位置。旧版本放在 _cache 里的锁，在新位置还没有时搬过来（旧文件保留不动）。"""
    path = lock_path(data_csv)
    if not data_csv or path.exists():
        return path
    old = Path(data_csv).parent / CACHE_DIRNAME / HOLDOUT_LOCK_NAME
    if old.exists():
        sets, split = _load(old)
        _save(path, sets, split)
        log.info(f"独立测试集的查看记录已从旧位置 {old} 搬到 {path}（_cache 可能被清理，所以不再放在那里；"
                 "旧文件不再使用，可以删掉）")
    return path


def holdout_hash(egg_ids: tuple[str, ...]) -> str:
    """独立测试集的指纹：排序后的蛋编号。"""
    return hashlib.sha1("\n".join(sorted(map(str, egg_ids))).encode("utf-8")).hexdigest()[:16]


BATCH_KEY_PREFIX = "batches:"
MAX_NAMES_IN_KEY = 20       # 批次名单超过这么多（比如没有批次信息、每颗蛋自成一组）就记名单的指纹


def view_key(batches: Iterable[str]) -> str:
    """查看次数的记录键：只由测试批次名单决定，和每批里有几颗蛋无关，所以增删蛋都不会清零。
    没有批次信息时每颗蛋自成一组（批次名"蛋<编号>"），名单本身也是固定的。"""
    names = sorted({str(b) for b in batches})
    if len(names) > MAX_NAMES_IN_KEY:
        return f"{BATCH_KEY_PREFIX}{len(names)}个#{holdout_hash(tuple(names))}"
    return BATCH_KEY_PREFIX + ",".join(names)


# ---------- 读写 ----------
def _sets_of(lock: dict) -> dict[str, dict]:
    """读出 {指纹: {"egg_ids_n":…, "evaluations":[…]}}；旧格式（只有一个测试集的 hash + evaluations）也认。"""
    if SETS_KEY in lock:
        sets = lock[SETS_KEY]
        for record in sets.values():
            record["evaluations"]
        return sets
    return {lock["hash"]: {"egg_ids_n": None, "evaluations": lock["evaluations"]}}


def _split_of(lock: dict) -> SplitRecord | None:
    raw = lock.get(SPLIT_KEY)
    if raw is None:
        return None
    return SplitRecord(tuple(map(str, raw["holdout_batches"])), tuple(map(str, raw["dev_batches"])),
                       bool(raw.get("holdout_viewed", False)))


def _load(path: Path) -> tuple[dict[str, dict], SplitRecord | None]:
    if not path.exists():
        return {}, None
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
        return _sets_of(lock), _split_of(lock)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(f"独立测试集的查看记录 {path} 读不了（{e}）。请不要手工修改它；"
                         "确认要清零（等于承认测试集已不独立）再删除这个文件") from e


def _save(path: Path, sets: dict[str, dict], split: SplitRecord | None) -> None:
    doc: dict = {SETS_KEY: sets}
    if split is not None:
        doc[SPLIT_KEY] = {"holdout_batches": list(split.holdout_batches), "dev_batches": list(split.dev_batches),
                          "holdout_viewed": split.holdout_viewed}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def read_split(path: Path) -> SplitRecord | None:
    return _load(path)[1]


def has_legacy_views(path: Path) -> bool:
    """旧版本的锁：记过查看次数，却没有记批次划分。"""
    sets, split = _load(path)
    return split is None and any(record.get("evaluations") for record in sets.values())


def recover_split(path: Path, batches: Iterable[str], candidate: tuple[str, ...],
                  candidate_hash: str) -> SplitRecord:
    """给旧锁补上批次划分。旧版本每次都用同一规则重新划分；如果按这条规则划出的测试批次（candidate）
    正好对得上一条旧查看记录，就认定它是当年的测试集（已被查看过）。对不上，说明数据变过、
    当年的测试批次已无从知道，只能保守处理：现有批次全部算开发批次，以后用 --holdout-batches 指定新批次。"""
    present = sorted({str(b) for b in batches})
    if candidate and candidate_hash in _load(path)[0]:
        return SplitRecord(tuple(sorted(candidate)), tuple(b for b in present if b not in candidate), True)
    return SplitRecord((), tuple(present), False)


def write_split(path: Path, split: SplitRecord) -> None:
    _save(path, _load(path)[0], split)


def prior_views(path: Path, ids_hash: str) -> int:
    """这个独立测试集（同样的蛋）以前被评估过几次。"""
    return len(_load(path)[0].get(ids_hash, {}).get("evaluations", []))


def record_view(path: Path, ids_hash: str, run: str, chosen: str, result: dict, n_eggs: int | None = None,
                egg_hash: str | None = None, formal: bool = True) -> int:
    """记一笔评估，返回这是第几次。别的测试集的记录原样保留；有划分记录时，同时标记测试集已被查看。
    formal=False 是调试模式的再次查看（非独立验证），同样算一次查看。"""
    sets, split = _load(path)
    old = sets.get(ids_hash, {})
    evaluations = [*old.get("evaluations", []), {"时间": datetime.now().isoformat(timespec="seconds"), "运行": run,
                                                 "选定方法": chosen, "成绩": result,
                                                 "蛋数": n_eggs, "蛋编号指纹": egg_hash, "正式结果": formal}]
    updated = {**sets, ids_hash: {"egg_ids_n": n_eggs if n_eggs is not None else old.get("egg_ids_n"),
                                  "evaluations": evaluations}}
    _save(path, updated, None if split is None else replace(split, holdout_viewed=True))
    return len(evaluations)


def merge_legacy(path: Path, key: str, legacy_hash: str) -> bool:
    """旧记录按蛋编号指纹存。当前测试集的指纹正好对得上一条旧记录时，把它并进批次键（并补上指纹）；对不上的原样保留。"""
    sets, split = _load(path)
    old = sets.get(legacy_hash)
    if old is None or key == legacy_hash:
        return False
    moved = [{**e, "蛋编号指纹": e.get("蛋编号指纹") or legacy_hash} for e in old["evaluations"]]
    target = sets.get(key, {})
    merged = {"egg_ids_n": target.get("egg_ids_n") or old.get("egg_ids_n"),
              "evaluations": sorted([*moved, *target.get("evaluations", [])], key=lambda e: e.get("时间", ""))}
    _save(path, {**{k: v for k, v in sets.items() if k != legacy_hash}, key: merged}, split)
    return True


def egg_change_note(path: Path, key: str, egg_hash: str, n_eggs: int) -> str | None:
    """上次查看时测试批次里的蛋和现在不一样（加了或少了蛋）时，返回提醒；没看过或没变返回 None。"""
    evaluations = _load(path)[0].get(key, {}).get("evaluations", [])
    last = evaluations[-1] if evaluations else {}
    if not last.get("蛋编号指纹") or last["蛋编号指纹"] == egg_hash:
        return None
    before = last.get("蛋数")
    return f"上次查看时有 {before if before is not None else '?'} 颗蛋，现在 {n_eggs} 颗"


# ---------- 划分 ----------
def _names(batches: Iterable[str]) -> str:
    return "、".join(sorted(batches))


def _check_requested(record: SplitRecord | None, present: set[str], requested: set[str]) -> None:
    """手工指定新的测试批次：必须在数据里，不能是曾经进过开发集的，也不能是已被查看过的测试集的批次。"""
    if requested - present:
        raise ValueError(f"--holdout-batches 里的批次 {_names(requested - present)} 在数据里找不到；"
                         f"数据里有这些批次：{_names(present)}")
    if not present - requested:
        raise ValueError("--holdout-batches 把全部批次都指定成了测试集，开发集会是空的")
    if record is None:
        return
    used = requested & set(record.dev_batches)
    if used:
        raise ValueError(f"批次 {_names(used)} 以前用于开发（已记录为开发批次），不能再当独立测试集。"
                         "请用从没参加过训练的新批次")
    viewed = requested & set(record.holdout_batches) if record.holdout_viewed else set()
    if viewed:
        raise ValueError(f"批次 {_names(viewed)} 属于已经被查看过的独立测试集，不能再当独立测试集。"
                         "请用从没参加过训练的新批次")


def _check_recorded(record: SplitRecord, present: set[str]) -> None:
    missing = set(record.holdout_batches) - present
    if missing:
        raise ValueError(f"记录里的独立测试批次 {_names(missing)} 在数据里找不到（被删掉或改名了）。"
                         "请把它们放回数据；确实不要了，就用 --holdout-batches 指定新的测试批次（旧测试集作废）")
    if set(record.holdout_batches) & set(record.dev_batches):
        raise ValueError("查看记录自相矛盾：有批次同时被记成测试批次和开发批次，请不要手工修改锁文件")


def _record_after(record: SplitRecord | None, holdout: tuple[str, ...], dev: tuple[str, ...]) -> SplitRecord:
    old_holdout = record.holdout_batches if record else ()
    old_dev = record.dev_batches if record else ()
    demoted = set(old_holdout) - set(holdout)      # 被换下来的旧测试批次，以后只能算开发批次
    same = set(old_holdout) == set(holdout)
    return SplitRecord(holdout, tuple(sorted({*old_dev, *dev, *demoted})),
                       bool(record and record.holdout_viewed and same))


def plan_split(record: SplitRecord | None, batches: Iterable[str], requested: tuple[str, ...] | None,
               auto_holdout: Callable[[], tuple[str, ...]]) -> SplitPlan:
    """决定哪些批次是独立测试集。

    - 指定了 requested：作为新的测试集（要通过检查）；
    - 锁文件里已有测试批次：原样沿用，没见过的批次进开发集；
    - 以前没留测试集，但开发批次已记录：仍然不留（旧批次不能当独立测试集），请用 --holdout-batches 指定新批次；
    - 完全没有记录：调用 auto_holdout 首次划分（批次太少时它返回空）。
    """
    present = {str(b) for b in batches}
    if requested:
        _check_requested(record, present, set(requested))
        holdout, source = tuple(sorted(set(requested))), SOURCE_REQUESTED
    elif record is not None and record.holdout_batches:
        _check_recorded(record, present)
        holdout, source = record.holdout_batches, SOURCE_REUSED
    elif record is not None:
        holdout, source = (), SOURCE_NONE
    else:
        holdout = tuple(sorted(auto_holdout()))
        source = SOURCE_FIRST if holdout else SOURCE_NONE
    dev = tuple(sorted(present - set(holdout)))
    seen = set(record.dev_batches) if record else set()
    return SplitPlan(holdout, dev, tuple(b for b in dev if b not in seen), source, _record_after(record, holdout, dev))
