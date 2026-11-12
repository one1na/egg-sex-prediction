"""种蛋模型的网页界面：在浏览器里训练、自检、判断新蛋、翻看每次运行的结果。

运行：双击 打开网页界面.bat，或 py -3.11 web_app.py（会自动打开浏览器）。
只用 Python 自带的库；页面文件在 web/ 文件夹里。按钮背后调用的仍然是 train_demo.py / predict.py。

安全：只监听本机（127.0.0.1），别的电脑连不上；会改动东西的请求必须带页面里的随机口令，
别的网站没法借你的浏览器偷偷启动训练。
"""
import argparse
import csv
import filecmp
import json
import logging
import os
import secrets
import subprocess
import sys
import threading
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import paths
from run_record import EXIT_HOLDOUT_USED, EXIT_SELF_CHECK_FAILED

CODE_DIR = Path(__file__).resolve().parent
WEB_DIR = CODE_DIR / "web"
HOST = "127.0.0.1"
DEFAULT_PORT = 8765
PORT_TRIES = 10                 # 默认端口被占用时，往后再试几个
MAX_TABLE_ROWS = 2000           # 一张表最多传这么多行给页面，多了浏览器会卡
MAX_LOG_LINES = 20000           # 任务日志最多留这么多行
MAX_BODY_BYTES = 64 * 1024
TOKEN_HEADER = "X-Egg-Token"
TOKEN_PLACEHOLDER = "__EGG_TOKEN__"
ABLATION_KIND = "消融实验"       # ablation.py 写在 config.json 里的类型
REPEAT_KIND = "重复交叉验证"     # repeat_cv.py 的
STATIC_TYPES = {".html": "text/html", ".css": "text/css", ".js": "text/javascript", ".svg": "image/svg+xml"}

log = logging.getLogger("egg_web")


# ---------- 任务：要运行哪个脚本、带什么参数 ----------
ACTION_NAMES = {"train": "训练", "self_check": "流程自检", "predict": "判断新蛋", "ablation": "消融实验",
                "repeat_cv": "重复交叉验证",
                "fertility_check": "真实光谱检验", "fertility_self_check": "真实光谱自检"}


def _data_arg(params: dict, required: bool = False) -> list[str]:
    data = str(params.get("data") or "").strip().strip('"')
    if not data:
        if required:
            raise ValueError("请先填写数据表的位置")
        return []
    if not Path(data).is_file():
        raise ValueError(f"找不到数据表：{data}")
    return ["--data", data]


def _day_arg(params: dict) -> list[str]:
    day = str(params.get("decision_day") or "").strip()
    if not day:
        return []
    if not day.isdigit():
        raise ValueError("决策日要填整数，比如 4")
    return ["--decision-day", day]


def build_command(action: str, params: dict) -> tuple[str, list[str]]:
    """页面请求 -> (任务名, 脚本和参数)。输入有问题时抛 ValueError，提示文字直接显示给用户。"""
    if action not in ACTION_NAMES:
        raise ValueError(f"不认识的操作：{action}")
    name = ACTION_NAMES[action]
    if action == "predict":
        return name, ["predict.py", *_data_arg(params)]
    if action == "fertility_check":
        return name, ["train_demo.py", "--fertility-check"]
    if action == "fertility_self_check":
        return name, ["train_demo.py", "--fertility-check", "--shuffle-labels"]
    if action == "ablation":
        return name, ["ablation.py", *_data_arg(params), *_day_arg(params)]
    if action == "repeat_cv":
        return name, ["repeat_cv.py", *_data_arg(params), *_day_arg(params)]
    args = ["train_demo.py", *_data_arg(params), *_day_arg(params)]
    if action == "self_check":
        return name, [*args, "--shuffle-labels"]
    return name, [*args, *(["--skip-holdout"] if params.get("skip_holdout") else [])]


OUTCOMES = {
    0: ("ok", "完成"),
    EXIT_SELF_CHECK_FAILED: ("self_check_failed", "自检未通过：流程可能有信息泄漏，或统计量算不出来无法确认。这次的结果不能用，"
                                                  "也没有保存新模型（之前的模型原样保留）。详见日志和这次运行的记录。"),
    EXIT_HOLDOUT_USED: ("holdout_blocked", "已拦截：独立测试集已经看过，不能再做正式评估，这次没有训练。只想重新训练就勾选"
                                           "“只重新训练，不看独立测试集”；其他办法见日志里的说明。"),
}
STOPPED = ("stopped", "已手动停止，这次运行没有结果。")
FAILED = ("error", "出错了，原因见下面的日志。")


def outcome(code: int, stopped: bool) -> tuple[str, str]:
    """子程序结束后页面上显示什么：成功、自检未通过、被拦截、手动停止、出错分开说。"""
    if stopped:
        return STOPPED
    return OUTCOMES.get(code, FAILED)


@dataclass
class Task:
    name: str
    action: str
    started: str
    lines: list[str] = field(default_factory=list)
    dropped: int = 0                # 日志太长时从开头丢掉的行数
    running: bool = True
    stopped: bool = False
    exit_code: int | None = None
    proc: subprocess.Popen | None = None


class TaskRunner:
    """同一时间只跑一个任务，输出一行一行存下来，页面定时来取。"""

    def __init__(self, code_dir: Path = CODE_DIR) -> None:
        self.code_dir = code_dir
        self.lock = threading.Lock()
        self.task: Task | None = None

    def start(self, action: str, name: str, args: list[str]) -> None:
        with self.lock:
            if self.task and self.task.running:
                raise RuntimeError(f"“{self.task.name}”还在运行，等它结束或先停止")
            task = Task(name, action, f"{datetime.now():%Y-%m-%d %H:%M:%S}")
            self.task = task
        threading.Thread(target=self._work, args=(task, args), daemon=True).start()

    def _work(self, task: Task, args: list[str]) -> None:
        code = 1
        try:
            if not (self.code_dir / args[0]).is_file():     # 不先查的话 Python 自己会以退出码 2 结束，和"自检未通过"撞车
                raise FileNotFoundError(f"找不到 {self.code_dir / args[0]}")
            proc = subprocess.Popen([sys.executable, *args], cwd=self.code_dir,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")   # 和本窗口共用控制台：关窗口时一起停
            with self.lock:
                task.proc = proc
                if task.stopped:        # 还没启动好就点了停止
                    proc.terminate()
            with proc.stdout:
                for line in proc.stdout:
                    self._append(task, line.rstrip("\n"))
            code = proc.wait()
        except OSError as e:
            self._append(task, f"无法启动 {args[0]}：{e}")
        finally:        # 无论成败都要标记结束，页面上的按钮才会恢复
            with self.lock:
                task.exit_code, task.running = code, False

    def _append(self, task: Task, line: str) -> None:
        with self.lock:
            task.lines.append(line)
            extra = len(task.lines) - MAX_LOG_LINES
            if extra > 0:
                del task.lines[:extra]
                task.dropped += extra

    def stop(self) -> bool:
        with self.lock:
            task = self.task
            if not task or not task.running:
                return False
            task.stopped = True
            if task.proc:
                task.proc.terminate()
            return True

    def snapshot(self, since: int = 0) -> dict:
        """当前任务的状态，以及第 since 行之后的新输出。"""
        with self.lock:
            task = self.task
            if not task:
                return {"task": None}
            first = max(since - task.dropped, 0)
            info = {"name": task.name, "action": task.action, "started": task.started, "running": task.running,
                    "lines": task.lines[first:], "next": task.dropped + len(task.lines)}
            if not task.running:
                info["exit_code"] = task.exit_code
                info["outcome"], info["message"] = outcome(task.exit_code, task.stopped)
            return {"task": info}


# ---------- 读运行记录 ----------
def read_table(path: Path, limit: int = MAX_TABLE_ROWS) -> dict:
    """CSV -> {columns, rows, total}；rows 最多 limit 行，total 是文件里的总行数。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        columns = next(reader, [])
        rows, total = [], 0
        for row in reader:
            total += 1
            if total <= limit:
                rows.append(row)
    return {"columns": columns, "rows": rows, "total": total}


def _read_config(run_dir: Path) -> dict | None:
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return config if isinstance(config, dict) else None


def run_status(config: dict | None) -> tuple[str, str]:
    """一次运行的结论：(类别, 文字)。类别给页面上色用。"""
    if config is None:
        return "incomplete", "没有跑完"
    if "已拦截" in config:
        return "blocked", "已拦截"
    if config.get("类型") in (ABLATION_KIND, REPEAT_KIND):
        return "ok", "完成"
    published = (config.get("模型发布") or {}).get("已发布")
    check = (config.get("自检") or {}).get("总体")
    if published is False or (check and check != "通过") or config.get("自检通过") is False:
        return "failed", "自检未通过"
    if config.get("流程自检（打乱标签）"):
        return "ok", "自检通过"
    if "决策日结果" not in config:
        return "incomplete", "没有跑完"
    return "ok", "完成"


def _kind(run_name: str) -> str:
    """文件夹名 <日期>_<时间>_<类型>[_序号] 里的类型。"""
    parts = run_name.split("_")
    return parts[2] if len(parts) > 2 else run_name


def run_summary(run_dir: Path) -> dict:
    config = _read_config(run_dir)
    category, text = run_status(config)
    config = config or {}
    result = config.get("决策日结果") or {}
    return {"name": run_dir.name, "kind": _kind(run_dir.name), "time": config.get("时间"),
            "status": category, "status_text": text, "chosen": config.get("部署方法"),
            "decision_day": config.get("决策日"), "deep_utility": result.get("深度模型决策效用"),
            "base_utility": result.get("基线决策效用"), "base_name": result.get("最好的基线"),
            "not_sex": "标签含义" in config}


def list_runs(runs_dir: Path) -> list[dict]:
    """所有运行，最新的在前。"""
    if not runs_dir.is_dir():
        return []
    dirs = sorted((d for d in runs_dir.iterdir() if d.is_dir()), key=lambda d: d.name, reverse=True)
    return [run_summary(d) for d in dirs]


def find_run(runs_dir: Path, name: str) -> Path:
    """按名字找运行文件夹；名字里不许带路径，免得读到别处去。"""
    if not name or name != Path(name).name or name in (".", ".."):
        raise FileNotFoundError(name)
    run_dir = runs_dir / name
    if not run_dir.is_dir():
        raise FileNotFoundError(name)
    return run_dir


def find_table(run_dir: Path, file: str) -> Path:
    if file != Path(file).name or not file.lower().endswith(".csv"):
        raise FileNotFoundError(file)
    path = run_dir / file
    if not path.is_file():
        raise FileNotFoundError(file)
    return path


def run_detail(run_dir: Path) -> dict:
    config = _read_config(run_dir)
    log_path = run_dir / "log.txt"
    return {**run_summary(run_dir), "config": config,
            "tables": sorted(p.name for p in run_dir.glob("*.csv")),
            "log": log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""}


def current_model(runs_dir: Path, model_path: Path) -> dict | None:
    """判断新蛋现在用的是哪次训练的模型：找内容和 output/egg_model.pt 完全一样的那次运行。"""
    if not model_path.is_file():
        return None
    info = {"saved": f"{datetime.fromtimestamp(model_path.stat().st_mtime):%Y-%m-%d %H:%M}", "run": None}
    for summary in list_runs(runs_dir):
        candidate = runs_dir / summary["name"] / model_path.name
        if candidate.is_file() and filecmp.cmp(candidate, model_path, shallow=False):
            return {**info, "run": summary["name"], "chosen": summary["chosen"],
                    "decision_day": summary["decision_day"]}
    return info


def pick_file() -> str:
    """在电脑上弹出选文件的窗口（浏览器自己拿不到文件的完整位置）。取消返回空字符串。"""
    script = ("import tkinter as tk\nfrom tkinter import filedialog\nr = tk.Tk()\nr.withdraw()\n"
              "r.attributes('-topmost', True)\n"
              "print(filedialog.askopenfilename(title='选择数据表 meta.csv', filetypes=[('CSV', '*.csv')]))")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"}, check=False)
    return done.stdout.strip()


# ---------- 网页服务 ----------
@dataclass
class App:
    runner: TaskRunner
    token: str
    runs_dir: Path = paths.RUNS_DIR
    model_path: Path = paths.MODEL_PATH
    decision_csv: Path = paths.DECISION_CSV
    predictions_csv: Path = paths.PREDICTIONS_CSV
    sample_csv: Path = paths.SAMPLE_CSV
    web_dir: Path = WEB_DIR

    def state(self, since: int) -> dict:
        return {**self.runner.snapshot(since), "model": current_model(self.runs_dir, self.model_path),
                "sample_csv": str(self.sample_csv) if self.sample_csv.is_file() else ""}

    def predictions(self) -> dict:
        if not self.decision_csv.is_file():
            return {"columns": [], "rows": [], "total": 0, "saved": None}
        saved = f"{datetime.fromtimestamp(self.decision_csv.stat().st_mtime):%Y-%m-%d %H:%M}"
        return {**read_table(self.decision_csv), "saved": saved, "daily_csv": str(self.predictions_csv)}

    def run(self, body: dict) -> dict:
        name, args = build_command(str(body.get("action")), body)
        self.runner.start(str(body.get("action")), name, args)
        return {"started": name}


class Handler(BaseHTTPRequestHandler):
    app: App        # make_server 里填上
    server_version = "EggWeb"

    def log_message(self, format: str, *args) -> None:   # noqa: A002  （父类的参数名）
        pass        # 不往窗口里刷每一次请求

    # --- 回应 ---
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict | list, status: int = HTTPStatus.OK) -> None:
        self._send(status, json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json")

    def _error(self, status: int, message: str) -> None:
        self._json({"error": message}, status)

    def _local_host(self) -> bool:
        """只认本机地址，挡住把别的域名解析到 127.0.0.1 的把戏。"""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
        return host in ("127.0.0.1", "localhost")

    # --- GET ---
    def do_GET(self) -> None:   # noqa: N802  （父类规定的名字）
        if not self._local_host():
            return self._error(HTTPStatus.FORBIDDEN, "只能在本机打开")
        url = urlsplit(self.path)
        parts = [unquote(p) for p in url.path.split("/") if p]
        query = parse_qs(url.query)
        try:
            if not parts or parts[0] != "api":
                return self._static(parts)
            if parts[1:] == ["state"]:
                since = query.get("since", ["0"])[0]
                return self._json(self.app.state(int(since) if since.isdigit() else 0))
            if parts[1:] == ["predictions"]:
                return self._json(self.app.predictions())
            if parts[1:] == ["runs"]:
                return self._json(list_runs(self.app.runs_dir))
            if len(parts) == 3 and parts[1] == "runs":
                return self._json(run_detail(find_run(self.app.runs_dir, parts[2])))
            if len(parts) == 5 and parts[1] == "runs" and parts[3] == "table":
                return self._json(read_table(find_table(find_run(self.app.runs_dir, parts[2]), parts[4])))
            self._error(HTTPStatus.NOT_FOUND, "没有这个地址")
        except FileNotFoundError:
            self._error(HTTPStatus.NOT_FOUND, "找不到这条记录")

    def _static(self, parts: list[str]) -> None:
        name = parts[0] if parts else "index.html"
        path = self.app.web_dir / name
        if len(parts) > 1 or name != Path(name).name or path.suffix not in STATIC_TYPES or not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "没有这个页面")
        body = path.read_bytes()
        if name == "index.html":
            body = body.replace(TOKEN_PLACEHOLDER.encode(), self.app.token.encode())
        self._send(HTTPStatus.OK, body, STATIC_TYPES[path.suffix])

    # --- POST ---
    def do_POST(self) -> None:   # noqa: N802
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY_BYTES:
            self.close_connection = True
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "请求太大")
        raw = self.rfile.read(length)       # 先把请求读完再回应，不然对方可能收不到回应
        if not self._local_host():
            return self._error(HTTPStatus.FORBIDDEN, "只能在本机打开")
        if not secrets.compare_digest((self.headers.get(TOKEN_HEADER) or "").encode(), self.app.token.encode()):
            return self._error(HTTPStatus.FORBIDDEN, "页面已过期，请刷新后再试")
        try:
            body = json.loads(raw or b"{}")
            if not isinstance(body, dict):
                raise ValueError("请求格式不对")
        except ValueError:
            return self._error(HTTPStatus.BAD_REQUEST, "请求格式不对")
        path = urlsplit(self.path).path
        try:
            if path == "/api/run":
                return self._json(self.app.run(body))
            if path == "/api/stop":
                return self._json({"stopped": self.app.runner.stop()})
            if path == "/api/pick":
                return self._json({"path": pick_file()})
            self._error(HTTPStatus.NOT_FOUND, "没有这个地址")
        except ValueError as e:
            self._error(HTTPStatus.BAD_REQUEST, str(e))
        except RuntimeError as e:
            self._error(HTTPStatus.CONFLICT, str(e))


def make_server(app: App, port: int) -> ThreadingHTTPServer:
    handler = type("EggHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server


def open_server(app: App, port: int) -> ThreadingHTTPServer:
    """端口被占用（多半是上一次没关）就往后换一个。"""
    for candidate in range(port, port + PORT_TRIES):
        try:
            return make_server(app, candidate)
        except OSError:
            continue
    raise OSError(f"{port} 到 {port + PORT_TRIES - 1} 这些端口都被占用了，请先关掉之前打开的网页界面窗口")


def main() -> None:
    parser = argparse.ArgumentParser(description="种蛋模型的网页界面")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    runner = TaskRunner()
    server = open_server(App(runner, secrets.token_urlsafe(24)), args.port)
    url = f"http://{HOST}:{server.server_address[1]}/"
    log.info(f"网页界面已启动：{url}\n关掉这个窗口就停止（正在跑的训练也会一起停）。")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        server.server_close()


if __name__ == "__main__":
    main()
