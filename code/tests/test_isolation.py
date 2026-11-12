"""数据隔离：划分的核对（isolation.audit），以及拟合步骤从不碰测试蛋的检查：
- 深度模型：测试蛋的图像填上离谱的大数（"放毒"），标准化参数只要算进了测试蛋就会看到；训练和早停验证用到的蛋按编号核对。
- 基线：SNV 对每条光谱单独缩放，放大数会被它抹平，所以改按"身份"核对——测试蛋 SNV 之后的每一条光谱，
  都不能原样出现在任何一次基线拟合的输入里。
用两种人为制造的泄漏核对过这些检查确实会失败（见 TestChecksCatchLeaks）。"""
import argparse
import logging
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
logging.disable(logging.WARNING)

import deep_train as dt  # noqa: E402
from baselines import snv  # noqa: E402
import isolation  # noqa: E402
import train_demo  # noqa: E402
from isolation import FoldPlan, IsolationError, audit, plan_folds  # noqa: E402
from synth_data import make_dataset  # noqa: E402
from test_e2e import EndToEndBase  # noqa: E402

POISON = 1e4            # 正常反射率在 0~1 之间，标准化后也就几十；见到上千就是碰了测试蛋
SEEN_LIMIT = 1e3


def tiny(n_eggs: int = 48):
    with mock.patch("synth_data.EGGS_PER_BATCH", 4):
        return make_dataset(n_eggs=n_eggs, seed=0)


class FitSpy:
    """记录标准化拟合看到的最大绝对值、基线拟合用到的每一条光谱，以及深度模型训练、早停验证用到的蛋编号。"""

    def __init__(self) -> None:
        self.max_seen: list[float] = []
        self.fit_rows: set[bytes] = set()
        self.loader_eggs: list[set[str]] = []
        self._norm_fit = dt.Normalizer.fit
        self._pipe_fit = Pipeline.fit
        self._loader = dt._loader

    def norm_fit(self, x, mask):
        self.max_seen.append(float(np.abs(x[mask]).max()))
        return self._norm_fit(x, mask)

    def pipe_fit(self, pipe, X, y=None, **kw):
        self.fit_rows.update(np.ascontiguousarray(row).tobytes() for row in np.asarray(X))
        return self._pipe_fit(pipe, X, y, **kw)

    def loader(self, data, idx, norm, seed=None):
        self.loader_eggs.append({data.egg_ids[i] for i in idx})
        return self._loader(data, idx, norm, seed)

    def patches(self) -> list:
        return [mock.patch.object(dt.Normalizer, "fit", side_effect=self.norm_fit),
                mock.patch.object(Pipeline, "fit", autospec=True, side_effect=self.pipe_fit),
                mock.patch.object(dt, "_loader", side_effect=self.loader)]


def poison(data, idx: np.ndarray):
    x = data.x.copy()
    x[idx] = POISON
    return replace(data, x=x)


def spectrum_rows(data, idx: np.ndarray) -> set[bytes]:
    """这些蛋有数据的每一帧 SNV 之后的光谱（和基线拟合时看到的逐位相同）。"""
    X = snv(data.spec)[idx][data.mask[idx]]
    return {np.ascontiguousarray(row).tobytes() for row in X}


class TestAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = tiny()
        cls.dev = train_demo.subset(cls.data, np.arange(40))
        cls.hold = train_demo.subset(cls.data, np.arange(40, 48))        # 最后两个批次
        cls.plans = plan_folds(cls.dev, 5, 0)

    def test_real_plan_passes_and_is_recorded(self) -> None:
        rec = audit(self.dev, self.hold, self.plans)
        self.assertEqual(rec["结果"], isolation.PASS)
        self.assertEqual(len(rec["各折"]), len(self.plans))
        self.assertEqual(rec["独立测试集"], {"蛋数": 8, "批次数": 2})
        self.assertEqual(sum(f["测试蛋数"] for f in rec["各折"]), 40)

    def test_plan_matches_training_seeds(self) -> None:
        """训练和消融用 plan_folds；第 k 折的早停划分必须和种子 seed + k 一致。"""
        from evaluation import inner_split, outer_folds
        for k, ((tr, te), p) in enumerate(zip(outer_folds(self.dev.sex, self.dev.batch, 5, 0), self.plans)):
            fit, val, _ = inner_split(tr, self.dev.sex, self.dev.batch, k)
            np.testing.assert_array_equal(p.test, te)
            np.testing.assert_array_equal(p.fit, fit)
            np.testing.assert_array_equal(p.val, val)

    def assertFails(self, pattern: str, dev=None, hold=None, plans=None) -> None:
        with self.assertRaisesRegex(IsolationError, pattern):
            audit(dev or self.dev, hold, plans or self.plans)

    def test_holdout_sharing_an_egg(self) -> None:
        self.assertFails("独立测试集和开发集之间有 1 个共同的蛋", hold=train_demo.subset(self.data, np.arange(39, 48)))

    def test_holdout_sharing_a_batch(self) -> None:
        hold = replace(self.hold, batch=np.array([self.dev.batch[0]] * 8, dtype=object))
        self.assertFails("共同的批次", hold=hold)

    def test_fold_train_and_test_share_a_batch(self) -> None:
        p = self.plans[0]
        moved = p.test[0]                                  # 测试里的一颗蛋，它的批次同伴还留在测试里
        bad_train = np.append(p.train, moved)
        bad = FoldPlan(bad_train, p.test, np.append(p.fit, moved), p.val, p.grouped)
        self.assertFails("第 1 折的训练和测试之间有 1 个共同的蛋", plans=[bad, *self.plans[1:]])

    def test_batch_split_across_train_and_test(self) -> None:
        p = self.plans[0]
        moved = p.test[:1]                                 # 只搬一颗：蛋不重复，但它的批次两边都有
        train = np.concatenate([p.train, moved])
        test = np.setdiff1d(p.test, moved)
        bad = FoldPlan(train, test, np.concatenate([p.fit, moved]), p.val, p.grouped)
        others = [FoldPlan(q.train, q.test, q.fit, q.val, q.grouped) for q in self.plans[1:]]
        with self.assertRaisesRegex(IsolationError, "第 1 折的训练和测试之间有 1 个共同的批次") as cm:
            audit(self.dev, None, [bad, *others])
        self.assertIn("每颗开发蛋恰好一次", str(cm.exception))

    def test_validation_overlapping_test(self) -> None:
        p = self.plans[0]
        bad = FoldPlan(p.train, p.test, p.fit, np.append(p.val, p.test[0]), p.grouped)
        self.assertFails("早停验证和测试之间", plans=[bad, *self.plans[1:]])

    def test_fit_plus_val_must_equal_train(self) -> None:
        p = self.plans[0]
        bad = FoldPlan(p.train, p.test, p.fit[1:], p.val, p.grouped)
        self.assertFails("拟合 \\+ 早停验证不等于训练部分", plans=[bad, *self.plans[1:]])

    def test_duplicate_egg_ids(self) -> None:
        dev = replace(self.dev, egg_ids=(self.dev.egg_ids[1], *self.dev.egg_ids[1:]))
        self.assertFails("蛋编号重复", dev=dev)

    def test_ungrouped_validation_is_a_note_in_demo_mode(self) -> None:
        p = self.plans[0]
        loose = FoldPlan(p.train, p.test, p.fit, p.val, False)
        rec = audit(self.dev, None, [loose, *self.plans[1:]])
        self.assertTrue(any("按蛋随机" in n for n in rec["提醒"]))

    def test_ungrouped_validation_is_an_error_in_formal_mode(self) -> None:
        """正式模式：早停验证没能按批次分就停下，不再悄悄换成按蛋随机分。"""
        p = self.plans[0]
        loose = FoldPlan(p.train, p.test, p.fit, p.val, False)
        with self.assertRaises(IsolationError) as caught:
            audit(self.dev, None, [loose, *self.plans[1:]], strict_val=True)
        self.assertIn("--allow-ungrouped-val", str(caught.exception))
        audit(self.dev, None, self.plans, strict_val=True)                     # 都按批次分的照常通过

    def test_strict_only_for_real_data_without_override(self) -> None:
        ns = lambda **k: argparse.Namespace(**{"data": None, "fertility_check": False, **k})  # noqa: E731
        self.assertFalse(train_demo.strict_val(ns()))
        self.assertTrue(train_demo.strict_val(ns(data="meta.csv")))
        self.assertTrue(train_demo.strict_val(ns(fertility_check=True)))
        self.assertFalse(train_demo.strict_val(ns(data="meta.csv", allow_ungrouped_val=True)))

    def test_no_batch_info_note(self) -> None:
        solo = replace(self.dev, batch=np.array(self.dev.egg_ids, dtype=object))
        rec = audit(solo, None, plan_folds(solo, 5, 0))
        self.assertTrue(any("没有批次信息" in n for n in rec["提醒"]))


def run_fold_spied(data, plan: FoldPlan, *leaks) -> FitSpy:
    """跑一折（每个深度模型只训练 1 轮），返回记录；leaks 是额外的 mock，用来人为制造泄漏。"""
    spy = FitSpy()
    with mock.patch.object(dt, "MAX_EPOCHS", 1), mock.patch.object(dt, "PATIENCE", 1):
        p = spy.patches()
        with p[0], p[1], p[2]:
            for leak in leaks:
                leak.start()
            try:
                train_demo.run_fold(data, plan, 0.5, 0)
            finally:
                for leak in leaks:
                    leak.stop()
    return spy


def leaks_into_fit(spy: FitSpy, data, plan: FoldPlan) -> list[str]:
    """一折里拟合碰到测试蛋的地方；空列表 = 没碰到。"""
    test_eggs = {data.egg_ids[i] for i in plan.test}
    found = []
    if max(spy.max_seen) >= SEEN_LIMIT:
        found.append("深度模型的标准化参数")
    if spy.fit_rows & spectrum_rows(data, plan.test):
        found.append("基线拟合")
    if set().union(*spy.loader_eggs) & test_eggs:
        found.append("深度模型的训练或早停验证")
    return found


class TestFoldNeverFitsOnTestEggs(unittest.TestCase):
    def setUp(self) -> None:
        data = tiny()
        self.plan = plan_folds(data, 4, 0)[0]
        self.data = poison(data, self.plan.test)

    def test_no_fit_touches_test_eggs(self) -> None:
        spy = run_fold_spied(self.data, self.plan)
        self.assertEqual(leaks_into_fit(spy, self.data, self.plan), [])
        # 检查本身在工作：确实记到了拟合，训练蛋的光谱也确实在里面
        self.assertTrue(spy.fit_rows & spectrum_rows(self.data, self.plan.fit))
        self.assertEqual(len(spy.loader_eggs), 2)              # 训练、早停验证
        self.assertEqual(spy.loader_eggs[0], {self.data.egg_ids[i] for i in self.plan.fit})


class TestChecksCatchLeaks(unittest.TestCase):
    """人为制造泄漏，上面的检查必须报出来；否则检查形同虚设。"""

    def setUp(self) -> None:
        data = tiny()
        self.plan = plan_folds(data, 4, 0)[0]
        self.data = poison(data, self.plan.test)

    def test_baseline_fitted_on_all_eggs(self) -> None:
        real = train_demo.per_day_baselines
        leak = mock.patch.object(train_demo, "per_day_baselines",
                                 lambda spec, mask, sex, tr, te, **kw: real(spec, mask, sex, np.arange(len(sex)), te, **kw))
        spy = run_fold_spied(self.data, self.plan, leak)
        self.assertIn("基线拟合", leaks_into_fit(spy, self.data, self.plan))

    def test_deep_model_trained_on_all_eggs(self) -> None:
        real = dt.train_early_stop
        leak = mock.patch.object(dt, "train_early_stop",
                                 lambda d, fit, val, w, s: real(d, np.arange(len(d.sex)), val, w, s))
        spy = run_fold_spied(self.data, self.plan, leak)
        found = leaks_into_fit(spy, self.data, self.plan)
        self.assertIn("深度模型的标准化参数", found)
        self.assertIn("深度模型的训练或早停验证", found)

    def test_early_stopping_on_test_eggs(self) -> None:
        real = dt.train_early_stop
        leak = mock.patch.object(dt, "train_early_stop",
                                 lambda d, fit, val, w, s: real(d, fit, self.plan.test, w, s))
        spy = run_fold_spied(self.data, self.plan, leak)
        self.assertEqual(leaks_into_fit(spy, self.data, self.plan), ["深度模型的训练或早停验证"])


class TestWholeRunNeverFitsOnHoldout(EndToEndBase):
    def test_holdout_never_reaches_any_fit(self) -> None:
        """整条训练（交叉验证 + 最终模型 + 独立测试）：独立测试集的蛋放毒，所有拟合都不应该碰到。"""
        real_split = train_demo.split_holdout
        held = []

        def poisoned_split(data, args):
            dev, hold = real_split(data, args)
            hold = poison(hold, np.arange(len(hold.egg_ids)))
            held.append(hold)
            return dev, hold

        spy = FitSpy()
        with mock.patch.object(train_demo, "split_holdout", side_effect=poisoned_split),                 mock.patch.object(train_demo, "make_dataset", side_effect=lambda seed: make_dataset(32, seed)):
            p = spy.patches()
            with p[0], p[1], p[2]:
                run = self.train()
        (hold,) = held
        self.assertTrue((run / "holdout.csv").exists())        # 独立测试确实做了：这些蛋被预测过，但没被拟合
        self.assertLess(max(spy.max_seen), SEEN_LIMIT)
        self.assertFalse(spy.fit_rows & spectrum_rows(hold, np.arange(len(hold.egg_ids))))
        self.assertFalse(set().union(*spy.loader_eggs) & set(hold.egg_ids))
        self.assertEqual(self.config(run)["隔离检查"]["结果"], isolation.PASS)


if __name__ == "__main__":
    unittest.main()
