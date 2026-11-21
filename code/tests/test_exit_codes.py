"""退出码：自检没通过用专门的退出码 2（和程序出错的 1 分开），窗口界面据此显示不同的提示。"""
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
logging.disable(logging.WARNING)

import gui_messages  # noqa: E402
import run_record  # noqa: E402
import train_demo  # noqa: E402
from run_record import EXIT_SELF_CHECK_FAILED  # noqa: E402


class TestExitCodes(unittest.TestCase):
    def test_self_check_failure_has_its_own_code(self) -> None:
        self.assertEqual(EXIT_SELF_CHECK_FAILED, 2)
        check = train_demo.check_record
        self.assertEqual(train_demo.exit_code(check({"a": "报警"})), 2)
        self.assertEqual(train_demo.exit_code(check({"a": "无法确认"})), 2)
        self.assertEqual(train_demo.exit_code(check({"a": "通过"})), 0)
        self.assertEqual(train_demo.exit_code(None), 0)

    def test_gui_messages(self) -> None:
        self.assertIsNone(gui_messages.failure_message("训练", 0))
        failed = gui_messages.failure_message("流程自检", EXIT_SELF_CHECK_FAILED)
        self.assertIn("[自检未通过]", failed)
        self.assertIn("不能用", failed)
        self.assertNotIn("截图", failed)
        error = gui_messages.failure_message("训练", 1)
        self.assertIn("[出错]", error)
        self.assertIn("截图留存", error)
        self.assertTrue(gui_messages.failure_message("训练", -1).startswith(gui_messages.FAIL_MARK))


class TestRunDir(unittest.TestCase):
    def test_same_second_runs_get_distinct_folders(self) -> None:
        fixed = datetime(2026, 1, 1, 12, 0, 0)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(run_record, "RUNS_DIR", Path(tmp) / "runs"), \
                mock.patch.object(run_record, "datetime", mock.Mock(now=mock.Mock(return_value=fixed))):
            dirs = [run_record.new_run_dir("模拟数据") for _ in range(3)]
            self.assertEqual(len(set(dirs)), 3)
            self.assertTrue(all(d.is_dir() for d in dirs))
            self.assertEqual([d.name[-2:] for d in dirs[1:]], ["_2", "_3"])


class TestNoTkinterNeeded(unittest.TestCase):
    def test_test_files_load_without_tkinter(self) -> None:
        """很多 Linux 机器和 CI 没装 tkinter：测试文件不能依赖它，否则整个文件都加载不了，里面的测试一项也跑不到。

        要测窗口界面的提示文字，从 gui_messages 导入，不要导入 gui。
        """
        tests_dir = Path(__file__).resolve().parent
        names = sorted(p.stem for p in tests_dir.glob("test_*.py"))
        code = "import sys\nsys.modules['tkinter'] = None\n" + "".join(f"import {n}\n" for n in names)
        result = subprocess.run([sys.executable, "-c", code], cwd=tests_dir, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=600,
                                env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        self.assertEqual(result.returncode, 0, result.stderr[-3000:])


if __name__ == "__main__":
    unittest.main()
