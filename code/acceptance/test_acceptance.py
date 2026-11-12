"""完整验收测试：不替换发布门控、不缩小训练设置，按正式流程把真实数据格式的整条链跑一遍。

和 tests/ 里的日常测试分开（日常测试为了快，用极小的数据和极少的轮数，并在测整条链时让发布门控放行；
门控本身另有专门的测试）。这里用全尺寸的样例数据（160 颗蛋、10 个批次，按真实数据格式导出：
原始图像 + 白板暗电流 + 波长文件）和正式的训练设置，只把输出改到临时文件夹，不碰 output/ 里的模型：

1. 没做自检就正式训练 → 发布门控拦下（退出码 2），不看独立测试集、不保存模型；
2. 打乱标签的流程自检 → 必须通过（退出码 0）；
3. 再正式训练 → 门控找到对应的自检，发布模型，独立测试集正式查看一次；
4. 用保存的模型判断新蛋；
5. 再正式训练一次 → 独立测试集已看过，开头就被拦下（退出码 3）。

要跑一两分钟（比日常测试慢，所以分开）。运行：双击 9-完整验收测试.bat，或在 code 文件夹里 py -3.11 -m unittest discover -s acceptance -v
数据是模拟的（性别信号是人为放进去的），通过只说明整条流程在正式设置下能走通，不代表真实效果。
"""
import contextlib
import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import deploy  # noqa: E402
import export_sample  # noqa: E402
import holdout_lock  # noqa: E402
import predict  # noqa: E402
import run_record  # noqa: E402
import train_demo  # noqa: E402

N_EGGS = 160            # 16 颗一批，10 个批次：够留独立测试集，每折训练部分也够按批次分早停验证


class TestFullAcceptance(unittest.TestCase):
    def setUp(self) -> None:
        self.out = Path(tempfile.mkdtemp(prefix="egg_acceptance_"))
        self.sample = self.out / "sample"
        self.csv = str(self.sample / "meta.csv")
        self.stack = contextlib.ExitStack()
        for p in (mock.patch.object(deploy, "MODEL_PATH", self.out / "egg_model.pt"),
                  mock.patch.object(deploy, "BASELINE_MODEL_PATH", self.out / "baseline_model.pkl"),
                  mock.patch.object(predict, "MODEL_PATH", self.out / "egg_model.pt"),
                  mock.patch.object(predict, "BASELINE_MODEL_PATH", self.out / "baseline_model.pkl"),
                  mock.patch.object(predict, "PREDICTIONS_CSV", self.out / "predictions.csv"),
                  mock.patch.object(predict, "DECISION_CSV", self.out / "decision.csv"),
                  mock.patch.object(run_record, "RUNS_DIR", self.out / "runs"),
                  mock.patch.object(holdout_lock, "SYNTH_HOLDOUT_LOCK", self.out / "holdout_lock.json")):
            self.stack.enter_context(p)
        with mock.patch.object(export_sample, "OUT_DIR", self.sample), \
                mock.patch.object(export_sample, "N_EGGS", N_EGGS):
            export_sample.main()

    def tearDown(self) -> None:
        self.stack.close()
        print(f"\n验收测试的全部输出在：{self.out}（确认无误后可以删掉）")

    def train(self, *args: str) -> tuple[int, dict, Path]:
        runs = self.out / "runs"
        before = set(runs.iterdir()) if runs.exists() else set()
        with mock.patch.object(sys, "argv", ["train_demo.py", "--data", self.csv, *args]):
            code = train_demo.main()
        (run,) = set(runs.iterdir()) - before
        return code, json.loads((run / "config.json").read_text(encoding="utf-8")), run

    def test_formal_chain(self) -> None:
        code, config, _ = self.train()
        self.assertEqual(code, 2, "没做自检的正式训练应该被发布门控拦下")
        self.assertEqual(config["发布检查"]["流程自检"]["状态"], "未执行")
        self.assertNotIn("holdout_views", config)
        self.assertFalse((self.out / "egg_model.pt").exists())

        code, config, check_run = self.train("--shuffle-labels")
        self.assertEqual(code, 0, f"打乱标签的流程自检没通过：{config.get('自检')}")
        self.assertEqual(config["自检"]["总体"], "通过")

        code, config, run = self.train()
        self.assertEqual(code, 0, f"正式训练没有发布：{config.get('模型发布')}")
        gate = config["发布检查"]
        self.assertEqual((gate["模式"], gate["流程自检"]["状态"], gate["流程自检"]["自检运行"]),
                         ("正式", "通过", check_run.name))
        self.assertIn(gate["背景负对照"]["状态"], ("通过", "不适用"))
        self.assertEqual(config["holdout_views"], 1)
        self.assertTrue(config["独立测试集"]["正式结果"])
        self.assertTrue((run / train_demo.HOLDOUT_CSV).exists())
        self.assertTrue(all(f["早停验证"] == "按批次" for f in config["隔离检查"]["各折"]))
        self.assertIn("同信息对照", config["决策日结果"])
        lock = json.loads((self.sample / "holdout_lock.json").read_text(encoding="utf-8"))
        (record,) = lock["sets"].values()
        self.assertEqual([e["状态"] for e in record["evaluations"]], ["完成"])

        with mock.patch.object(sys, "argv", ["predict.py", "--data", self.csv]):
            predict.main()
        with (self.out / "decision.csv").open(encoding="utf-8-sig", newline="") as f:
            self.assertEqual(len(list(csv.DictReader(f))), N_EGGS)

        code, config, _ = self.train()
        self.assertEqual(code, 3, "独立测试集已经看过，再次正式训练应该在开头被拦下")
        self.assertIn("已拦截", config)


if __name__ == "__main__":
    unittest.main()
