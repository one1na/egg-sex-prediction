"""窗口界面在子程序结束后显示的提示。

单独放在这里、不导入 tkinter：很多 Linux 机器和 CI 没装 tkinter，这些提示照样能测试。
"""
from run_record import EXIT_HOLDOUT_USED, EXIT_SELF_CHECK_FAILED

FAIL_MARK = "\n[出错]"
SELF_CHECK_MARK = "\n[自检未通过]"
HOLDOUT_MARK = "\n[已拦截]"


def failure_message(name: str, code: int) -> str | None:
    """子程序结束后要在界面上追加的提示；成功返回 None。自检没通过和程序出错分开说。"""
    if code == 0:
        return None
    if code == EXIT_SELF_CHECK_FAILED:
        return (f"{SELF_CHECK_MARK} 流程可能有信息泄漏，或统计量算不出来无法确认；真实数据还可能是没先做对应的流程自检。"
                "详见日志和 output/runs/ 里最新一次运行的 config.json（“发布检查”一项）；"
                "这次的结果不能用，也没有保存新模型（之前的模型原样保留）\n")
    if code == EXIT_HOLDOUT_USED:
        return (f"{HOLDOUT_MARK} 独立测试集已经看过，不能再做正式评估，这次没有训练。只想重新训练就勾选"
                "“只重新训练，不看独立测试集”；其他办法见上面的说明\n")
    return f"{FAIL_MARK} {name}失败，请把上面的内容截图留存，对照使用说明排查\n"
