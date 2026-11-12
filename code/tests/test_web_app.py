"""网页界面的后台：请求怎么变成命令、任务怎么跑和停、运行记录怎么读，以及只认本机和口令。"""
import http.client
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import web_app  # noqa: E402
from run_record import EXIT_HOLDOUT_USED, EXIT_SELF_CHECK_FAILED, save_csv, save_json  # noqa: E402

WAIT_SECONDS = 20


def wait_done(runner: web_app.TaskRunner) -> dict:
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        task = runner.snapshot()["task"]
        if task and not task["running"]:
            return task
        time.sleep(0.05)
    raise AssertionError("任务一直没有结束")


class TestBuildCommand(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.csv = Path(self.tmp.name) / "meta.csv"
        self.csv.write_text("egg_id\n", encoding="utf-8")

    def test_train_defaults_to_synthetic_data(self) -> None:
        self.assertEqual(web_app.build_command("train", {}), ("训练", ["train_demo.py"]))

    def test_train_with_options(self) -> None:
        _, args = web_app.build_command("train", {"data": f' "{self.csv}" ', "decision_day": " 6 ", "skip_holdout": True})
        self.assertEqual(args, ["train_demo.py", "--data", str(self.csv), "--decision-day", "6", "--skip-holdout"])

    def test_self_check_never_skips_holdout_flag(self) -> None:
        _, args = web_app.build_command("self_check", {"skip_holdout": True})
        self.assertEqual(args, ["train_demo.py", "--shuffle-labels"])

    def test_other_actions(self) -> None:
        self.assertEqual(web_app.build_command("predict", {})[1], ["predict.py"])
        self.assertEqual(web_app.build_command("predict", {"data": str(self.csv)})[1], ["predict.py", "--data", str(self.csv)])
        self.assertEqual(web_app.build_command("fertility_check", {"data": str(self.csv)})[1],
                         ["train_demo.py", "--fertility-check"])       # 真实光谱检验用固定的数据，填了也不带
        self.assertEqual(web_app.build_command("fertility_self_check", {})[1],
                         ["train_demo.py", "--fertility-check", "--shuffle-labels"])
        self.assertEqual(web_app.build_command("ablation", {"decision_day": "4"})[1], ["ablation.py", "--decision-day", "4"])

    def test_bad_input_is_explained(self) -> None:
        with self.assertRaisesRegex(ValueError, "找不到数据表"):
            web_app.build_command("train", {"data": str(self.csv) + ".没有"})
        with self.assertRaisesRegex(ValueError, "整数"):
            web_app.build_command("train", {"decision_day": "4; rm"})
        with self.assertRaisesRegex(ValueError, "不认识"):
            web_app.build_command("format_disk", {})


class TestOutcome(unittest.TestCase):
    def test_each_ending_has_its_own_message(self) -> None:
        self.assertEqual(web_app.outcome(0, False)[0], "ok")
        self.assertEqual(web_app.outcome(EXIT_SELF_CHECK_FAILED, False)[0], "self_check_failed")
        self.assertIn("没有保存新模型", web_app.outcome(EXIT_SELF_CHECK_FAILED, False)[1])
        self.assertEqual(web_app.outcome(EXIT_HOLDOUT_USED, False)[0], "holdout_blocked")
        self.assertEqual(web_app.outcome(1, False)[0], "error")
        self.assertEqual(web_app.outcome(-9, False)[0], "error")
        self.assertEqual(web_app.outcome(1, True)[0], "stopped")       # 手动停止不算出错


class TestTaskRunner(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        self.runner = web_app.TaskRunner(self.dir)
        self.addCleanup(self.runner.stop)

    def script(self, name: str, body: str) -> str:
        (self.dir / name).write_text(body, encoding="utf-8")
        return name

    def test_output_and_exit_code_are_reported(self) -> None:
        script = self.script("blocked.py", f"import sys\nprint('第一行')\nprint('第二行')\nsys.exit({EXIT_HOLDOUT_USED})\n")
        self.runner.start("train", "训练", [script])
        task = wait_done(self.runner)
        self.assertEqual(task["lines"], ["第一行", "第二行"])
        self.assertEqual((task["exit_code"], task["outcome"]), (EXIT_HOLDOUT_USED, "holdout_blocked"))
        self.assertEqual(self.runner.snapshot(since=1)["task"]["lines"], ["第二行"])     # 只取新的
        self.assertEqual(self.runner.snapshot(since=task["next"])["task"]["lines"], [])

    def test_only_one_task_at_a_time_and_stop(self) -> None:
        script = self.script("slow.py", "import time\nprint('开始', flush=True)\ntime.sleep(60)\n")
        self.runner.start("train", "训练", [script])
        with self.assertRaisesRegex(RuntimeError, "还在运行"):
            self.runner.start("predict", "判断新蛋", [script])
        self.assertTrue(self.runner.stop())
        task = wait_done(self.runner)
        self.assertEqual(task["outcome"], "stopped")
        self.assertFalse(self.runner.stop())        # 已经停了，再停没有可停的

    def test_missing_script_is_an_error_not_a_hang(self) -> None:
        self.runner.start("train", "训练", ["没有这个脚本.py"])
        self.assertEqual(wait_done(self.runner)["outcome"], "error")

    def test_long_log_keeps_line_numbers(self) -> None:
        task = web_app.Task("训练", "train", "now")
        for i in range(web_app.MAX_LOG_LINES + 5):
            self.runner._append(task, str(i))
        self.runner.task, task.running, task.exit_code = task, False, 0
        snap = self.runner.snapshot(since=web_app.MAX_LOG_LINES + 3)["task"]
        self.assertEqual(snap["lines"], [str(web_app.MAX_LOG_LINES + 3), str(web_app.MAX_LOG_LINES + 4)])
        self.assertEqual(snap["next"], web_app.MAX_LOG_LINES + 5)


class RunsCase(unittest.TestCase):
    """临时的 runs 文件夹，里面放几种结局的运行。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.runs = self.root / "runs"
        self.add("2026-01-01_100000_模拟数据", {"时间": "t1", "决策日": 4, "部署方法": "PLS-DA",
                                              "决策日结果": {"深度模型决策效用": 0.6, "基线决策效用": 0.7, "最好的基线": "PLS-DA"},
                                              "模型发布": {"已发布": True, "未发布原因": []}})
        self.add("2026-01-02_100000_模拟数据", {"已拦截": "测试集看过了"})
        self.add("2026-01-03_100000_模拟数据", {"决策日结果": {}, "自检": {"总体": "报警"},
                                              "模型发布": {"已发布": False, "未发布原因": ["自检报警"]}})
        self.add("2026-01-04_100000_流程自检", {"流程自检（打乱标签）": True, "自检": {"总体": "通过"}})
        self.add("2026-01-05_100000_消融实验", {"类型": "消融实验", "决策日": 4})
        self.add("2026-01-06_100000_真实光谱检验", {"决策日结果": {}, "标签含义": "受精，不是性别", "部署方法": "PLS-DA"})
        (self.runs / "2026-01-07_100000_模拟数据").mkdir()        # 中途停掉的：没有 config.json

    def add(self, name: str, config: dict) -> Path:
        run_dir = self.runs / name
        run_dir.mkdir(parents=True)
        save_json(run_dir / "config.json", config)
        return run_dir


class TestRuns(RunsCase):
    def test_list_is_newest_first_with_status(self) -> None:
        runs = web_app.list_runs(self.runs)
        self.assertEqual([r["name"][:10] for r in runs], [f"2026-01-0{d}" for d in range(7, 0, -1)])
        status = {r["name"][8:10]: r["status"] for r in runs}
        self.assertEqual(status, {"07": "incomplete", "06": "ok", "05": "ok", "04": "ok",
                                  "03": "failed", "02": "blocked", "01": "ok"})
        self.assertEqual(runs[-1]["kind"], "模拟数据")
        self.assertEqual((runs[-1]["chosen"], runs[-1]["base_utility"]), ("PLS-DA", 0.7))
        self.assertTrue(runs[1]["not_sex"])
        self.assertFalse(runs[-1]["not_sex"])

    def test_missing_runs_folder_is_empty(self) -> None:
        self.assertEqual(web_app.list_runs(self.root / "没有"), [])

    def test_old_format_self_check_flag(self) -> None:
        self.assertEqual(web_app.run_status({"决策日结果": {}, "自检通过": False})[0], "failed")
        self.assertEqual(web_app.run_status({"决策日结果": {}, "自检通过": True})[0], "ok")
        self.assertEqual(web_app.run_status({"时间": "只写了开头"})[0], "incomplete")

    def test_names_cannot_escape_the_folder(self) -> None:
        (self.root / "secret.csv").write_text("a\n1\n", encoding="utf-8")
        for name in ("..", ".", "", "../runs", "a/b", "a\\b", "没有这次运行"):
            with self.assertRaises(FileNotFoundError, msg=name):
                web_app.find_run(self.runs, name)
        run_dir = web_app.find_run(self.runs, "2026-01-01_100000_模拟数据")
        for file in ("../../secret.csv", "..\\..\\secret.csv", "config.json", "没有.csv"):
            with self.assertRaises(FileNotFoundError, msg=file):
                web_app.find_table(run_dir, file)

    def test_detail_and_table(self) -> None:
        run_dir = self.runs / "2026-01-01_100000_模拟数据"
        save_csv(run_dir / "results_per_day.csv", [{"方法": "PLS-DA", "孵化日": d, "决策效用均值": 0.5} for d in range(5)])
        (run_dir / "log.txt").write_text("日志内容", encoding="utf-8")
        detail = web_app.run_detail(run_dir)
        self.assertEqual(detail["tables"], ["results_per_day.csv"])
        self.assertEqual(detail["log"], "日志内容")
        self.assertEqual(detail["config"]["部署方法"], "PLS-DA")
        table = web_app.read_table(web_app.find_table(run_dir, "results_per_day.csv"), limit=3)
        self.assertEqual(table["columns"], ["方法", "孵化日", "决策效用均值"])        # 不带 BOM
        self.assertEqual((len(table["rows"]), table["total"]), (3, 5))
        self.assertEqual(web_app.run_detail(self.runs / "2026-01-07_100000_模拟数据")["config"], None)

    def test_current_model_is_matched_by_content(self) -> None:
        model = self.root / "egg_model.pt"
        self.assertIsNone(web_app.current_model(self.runs, model))
        model.write_bytes(b"model-A")
        self.assertIsNone(web_app.current_model(self.runs, model)["run"])           # 有模型，但对不上任何运行
        (self.runs / "2026-01-06_100000_真实光谱检验" / "egg_model.pt").write_bytes(b"model-B")
        (self.runs / "2026-01-01_100000_模拟数据" / "egg_model.pt").write_bytes(b"model-A")
        info = web_app.current_model(self.runs, model)
        self.assertEqual((info["run"], info["chosen"], info["decision_day"]), ("2026-01-01_100000_模拟数据", "PLS-DA", 4))


class TestServer(RunsCase):
    TOKEN = "test-token"

    def setUp(self) -> None:
        super().setUp()
        web = self.root / "web"
        web.mkdir()
        (web / "index.html").write_text(f"<meta content='{web_app.TOKEN_PLACEHOLDER}'>", encoding="utf-8")
        (web / "app.js").write_text("// js", encoding="utf-8")
        (web / "secret.txt").write_text("不该被读到", encoding="utf-8")
        self.runner = web_app.TaskRunner(self.root)
        app = web_app.App(self.runner, self.TOKEN, runs_dir=self.runs, model_path=self.root / "egg_model.pt",
                          decision_csv=self.root / "decision.csv", predictions_csv=self.root / "daily.csv",
                          sample_csv=self.root / "sample.csv", web_dir=web)
        self.server = web_app.make_server(app, 0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.addCleanup(self.runner.stop)

    def request(self, method: str, path: str, body: dict | None = None, token: str | None = None,
                host: str | None = None) -> tuple[int, dict | str]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=WAIT_SECONDS)
        self.addCleanup(conn.close)
        conn.putrequest(method, path, skip_host=True)
        conn.putheader("Host", host or f"127.0.0.1:{self.port}")
        data = json.dumps(body).encode() if body is not None else b""
        conn.putheader("Content-Length", str(len(data)))
        if token:
            conn.putheader(web_app.TOKEN_HEADER, token)
        conn.endheaders(data)
        response = conn.getresponse()
        text = response.read().decode("utf-8")
        is_json = response.getheader("Content-Type", "").startswith("application/json")
        return response.status, json.loads(text) if is_json else text

    def test_page_gets_the_token(self) -> None:
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(self.TOKEN, page)
        self.assertNotIn(web_app.TOKEN_PLACEHOLDER, page)
        self.assertEqual(self.request("GET", "/app.js"), (200, "// js"))

    def test_only_known_static_files(self) -> None:
        for path in ("/secret.txt", "/../web_app.py", "/..%2Fweb_app.py", "/web/index.html", "/nothing.html"):
            self.assertEqual(self.request("GET", path)[0], 404, path)

    def test_read_apis(self) -> None:
        status, runs = self.request("GET", "/api/runs")
        self.assertEqual((status, len(runs)), (200, 7))
        name = "2026-01-02_100000_模拟数据"
        from urllib.parse import quote
        status, detail = self.request("GET", f"/api/runs/{quote(name)}")
        self.assertEqual((status, detail["status"], detail["config"]["已拦截"]), (200, "blocked", "测试集看过了"))
        self.assertEqual(self.request("GET", "/api/runs/..%2F..")[0], 404)
        self.assertEqual(self.request("GET", f"/api/runs/{quote(name)}/table/..%2Fconfig.json")[0], 404)
        status, state = self.request("GET", "/api/state")
        self.assertEqual((status, state["task"], state["model"], state["sample_csv"]), (200, None, None, ""))
        status, predictions = self.request("GET", "/api/predictions")
        self.assertEqual((status, predictions["saved"], predictions["rows"]), (200, None, []))

    def test_other_hosts_are_refused(self) -> None:
        self.assertEqual(self.request("GET", "/api/runs", host="evil.example")[0], 403)
        self.assertEqual(self.request("POST", "/api/stop", {}, token=self.TOKEN, host="evil.example:80")[0], 403)
        self.assertEqual(self.request("GET", "/api/runs", host=f"localhost:{self.port}")[0], 200)

    def test_changes_need_the_token(self) -> None:
        self.assertEqual(self.request("POST", "/api/run", {"action": "train"})[0], 403)
        self.assertEqual(self.request("POST", "/api/run", {"action": "train"}, token="wrong")[0], 403)
        self.assertEqual(self.request("POST", "/api/stop", {})[0], 403)
        self.assertIsNone(self.runner.snapshot()["task"])           # 什么都没启动

    def test_run_through_the_api(self) -> None:
        status, reply = self.request("POST", "/api/run", {"action": "nope"}, token=self.TOKEN)
        self.assertEqual(status, 400)
        self.assertIn("不认识", reply["error"])
        status, reply = self.request("POST", "/api/run", {"action": "train", "decision_day": "x"}, token=self.TOKEN)
        self.assertEqual((status, "整数" in reply["error"]), (400, True))
        (self.root / "predict.py").write_text("import time\nprint('判断中', flush=True)\ntime.sleep(60)\n", encoding="utf-8")
        self.assertEqual(self.request("POST", "/api/run", {"action": "predict"}, token=self.TOKEN), (200, {"started": "判断新蛋"}))
        self.assertEqual(self.request("POST", "/api/run", {"action": "predict"}, token=self.TOKEN)[0], 409)   # 已经有一个在跑
        self.assertEqual(self.request("POST", "/api/stop", {}, token=self.TOKEN), (200, {"stopped": True}))
        self.assertEqual(wait_done(self.runner)["outcome"], "stopped")
        _, state = self.request("GET", "/api/state?since=0")
        self.assertEqual((state["task"]["action"], state["task"]["running"]), ("predict", False))

    def test_bad_body(self) -> None:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=WAIT_SECONDS)
        self.addCleanup(conn.close)
        conn.request("POST", "/api/run", body=b"not json", headers={web_app.TOKEN_HEADER: self.TOKEN})
        self.assertEqual(conn.getresponse().status, 400)


if __name__ == "__main__":
    unittest.main()
