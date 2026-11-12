"""种蛋模型的窗口界面：不用改代码，点按钮就能训练、自检、判断、看结果。

运行：双击 打开模型界面.bat，或 py -3.11 gui.py
按钮背后调用的就是 train_demo.py / predict.py / baseline_fertility.py。
"""
import csv
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from paths import DECISION_CSV, PREDICTIONS_CSV, SAMPLE_CSV
from run_record import EXIT_HOLDOUT_USED, EXIT_SELF_CHECK_FAILED

CODE_DIR = Path(__file__).resolve().parent
POLL_MS = 100
FONT = ("Microsoft YaHei UI", 10)
RESULT_COLUMNS = ("蛋编号", "决策日", "当天有图像", "采用方法", "判断", "分类分数")


FAIL_MARK = "\n[出错]"
SELF_CHECK_MARK = "\n[自检未通过]"
HOLDOUT_MARK = "\n[已拦截]"


def failure_message(name: str, code: int) -> str | None:
    """子程序结束后要在界面上追加的提示；成功返回 None。自检没通过和程序出错分开说。"""
    if code == 0:
        return None
    if code == EXIT_SELF_CHECK_FAILED:
        return (f"{SELF_CHECK_MARK} 流程可能有信息泄漏，或统计量算不出来无法确认，详见日志和 output/runs/ 里最新一次"
                "运行的 config.json；这次的结果不能用，也没有保存新模型（之前的模型原样保留）\n")
    if code == EXIT_HOLDOUT_USED:
        return (f"{HOLDOUT_MARK} 独立测试集已经看过，不能再做正式评估，这次没有训练。只想重新训练就勾选"
                "“只重新训练，不看独立测试集”；其他办法见上面的说明\n")
    return f"{FAIL_MARK} {name}失败，请把上面的内容截图留存，对照使用说明排查\n"


class EggApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.busy = False
        self.train_csv = tk.StringVar(value="")
        self.decision_day = tk.StringVar(value="")
        self.skip_holdout = tk.BooleanVar(value=False)
        self.predict_csv = tk.StringVar(value=str(SAMPLE_CSV))
        root.title("种蛋性别与发育模型")
        root.geometry("860x740")
        self._build()

    # ---------- 界面 ----------
    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}
        step1 = ttk.LabelFrame(self.root, text="第 1 步  训练模型（分组交叉验证 + 和传统方法对比 + 最终模型）")
        step1.pack(fill="x", **pad)
        self._file_row(step1, "训练数据：", self.train_csv, "不选 = 用模拟数据")
        self._day_row(step1)
        ttk.Checkbutton(step1, variable=self.skip_holdout,
                        text="只重新训练，不看独立测试集（测试集只能正式看一次；已经看过之后要勾上才能再训练）"
                        ).pack(anchor="w", padx=8, pady=2)
        buttons = ttk.Frame(step1)
        buttons.pack(anchor="w", padx=8, pady=6)
        self.btn_train = ttk.Button(buttons, text="开始训练", command=self.train)
        self.btn_train.pack(side="left")
        self.btn_check = ttk.Button(buttons, text="流程自检（打乱标签）", command=self.self_check)
        self.btn_check.pack(side="left", padx=8)

        step2 = ttk.LabelFrame(self.root, text="第 2 步  判断新蛋（用训练好的模型判断没答案的蛋）")
        step2.pack(fill="x", **pad)
        self._file_row(step2, "要判断的数据：", self.predict_csv, "默认是样例数据")
        self.btn_predict = ttk.Button(step2, text="开始判断", command=self.predict)
        self.btn_predict.pack(anchor="w", padx=8, pady=6)

        extra = ttk.LabelFrame(self.root, text="练习  真实公开数据（伊利诺伊大学，判断是否受精）")
        extra.pack(fill="x", **pad)
        self.btn_base = ttk.Button(extra, text="运行真实数据基线", command=self.baseline)
        self.btn_base.pack(anchor="w", padx=8, pady=6)

        self.status = ttk.Label(self.root, text="准备就绪", font=FONT)
        self.status.pack(anchor="w", padx=12)

        self.table = ttk.Treeview(self.root, columns=RESULT_COLUMNS, show="headings", height=6)
        for col in RESULT_COLUMNS:
            self.table.heading(col, text=col)
            self.table.column(col, width=110, anchor="center")
        self.table.pack(fill="x", **pad)

        self.log = tk.Text(self.root, height=12, font=("Consolas", 10), wrap="none")
        self.log.pack(fill="both", expand=True, **pad)

    def _file_row(self, parent: ttk.Frame, label: str, var: tk.StringVar, hint: str) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=label, font=FONT).pack(side="left")
        ttk.Entry(row, textvariable=var, width=60).pack(side="left", padx=4)
        ttk.Button(row, text="选择…", command=lambda: self._pick(var)).pack(side="left")
        ttk.Label(row, text=hint, foreground="gray").pack(side="left", padx=6)

    def _day_row(self, parent: ttk.Frame) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="决策日：", font=FONT).pack(side="left")
        ttk.Entry(row, textvariable=self.decision_day, width=6).pack(side="left", padx=4)
        ttk.Label(row, text="第几天必须出结果，主指标只看这一天；留空 = 用 train_demo.py 里的默认值",
                  foreground="gray").pack(side="left", padx=6)

    def _pick(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="选择数据表 meta.csv", filetypes=[("CSV", "*.csv")])
        if path:
            var.set(path)

    # ---------- 按钮 ----------
    def _train_args(self) -> list[str] | None:
        """训练和自检共用的参数；输入有误时弹窗并返回 None。"""
        data = self.train_csv.get().strip()
        if data and not Path(data).exists():
            messagebox.showerror("找不到文件", f"训练数据不存在：\n{data}")
            return None
        day = self.decision_day.get().strip()
        if day and not day.isdigit():
            messagebox.showerror("决策日有误", "决策日要填整数，比如 4")
            return None
        return (["--data", data] if data else []) + (["--decision-day", day] if day else [])

    def train(self) -> None:
        args = self._train_args()
        if args is not None:
            skip = ["--skip-holdout"] if self.skip_holdout.get() else []
            self._run("训练", ["train_demo.py", *args, *skip])

    def self_check(self) -> None:
        args = self._train_args()
        if args is not None:
            self._run("流程自检", ["train_demo.py", "--shuffle-labels", *args])

    def predict(self) -> None:
        data = self.predict_csv.get().strip()
        if not data or not Path(data).exists():
            messagebox.showerror("找不到文件", "请先选择要判断的数据表")
            return
        self._run("判断", ["predict.py", "--data", data], on_done=self._show_predictions)

    def baseline(self) -> None:
        self._run("真实数据基线", ["baseline_fertility.py"])

    # ---------- 后台运行，把输出实时显示出来 ----------
    def _run(self, name: str, args: list[str], on_done: Callable[[], None] | None = None) -> None:
        if self.busy:
            messagebox.showinfo("请稍等", "上一个任务还没结束")
            return
        self.busy = True
        self._set_buttons("disabled")
        self.log.delete("1.0", "end")
        self.status.config(text=f"正在{name}，请稍等……")
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

        def work() -> None:
            code = 1
            try:
                proc = subprocess.Popen([sys.executable, *args], cwd=CODE_DIR, env=env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        text=True, encoding="utf-8", errors="replace",
                                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                for line in proc.stdout:
                    self.lines.put(line)
                code = proc.wait()
            except OSError as e:
                self.lines.put(f"无法启动 {args[0]}：{e}\n")
            finally:   # 无论成败都通知界面结束，按钮才会恢复
                self.lines.put(failure_message(name, code))
                self.lines.put("__END__")

        threading.Thread(target=work, daemon=True).start()
        self.root.after(POLL_MS, self._poll, name, on_done, [True])

    def _poll(self, name: str, on_done: Callable[[], None] | None, ok: list[bool]) -> None:
        while not self.lines.empty():
            line = self.lines.get()
            if line == "__END__":
                self.busy = False
                self._set_buttons("normal")
                self.status.config(text=f"{name}完成" if ok[0] else f"{name}未通过" if len(ok) > 1 else f"{name}出错")
                if ok[0] and on_done:
                    on_done()
                return
            if line is None:
                continue
            if line.startswith((FAIL_MARK, SELF_CHECK_MARK, HOLDOUT_MARK)):
                ok[0] = False
                if line.startswith((SELF_CHECK_MARK, HOLDOUT_MARK)):
                    ok.append(False)        # 第二项只用来区分"自检未通过"和"出错"
            self.log.insert("end", line)
            self.log.see("end")
        self.root.after(POLL_MS, self._poll, name, on_done, ok)

    def _set_buttons(self, state: str) -> None:
        for btn in (self.btn_train, self.btn_check, self.btn_predict, self.btn_base):
            btn.config(state=state)

    def _show_predictions(self) -> None:
        self.table.delete(*self.table.get_children())
        with DECISION_CSV.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            self.table.insert("", "end", values=(r["egg_id"], r["决策日"], r["决策日有图像"], r["采用方法"],
                                                 r["判断"], r["分类分数"]))
        judged = sum(r["判断"] != "无法判断" for r in rows)
        self.status.config(text=f"判断完成：{len(rows)} 颗蛋，决策日给出判断 {judged} 颗；"
                                f"逐日结果在 {PREDICTIONS_CSV}")


if __name__ == "__main__":
    app_root = tk.Tk()
    EggApp(app_root)
    app_root.mainloop()
