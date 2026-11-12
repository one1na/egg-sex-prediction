"""项目里所有文件位置都在这里定义，换位置只改这一个文件。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # egg模型 文件夹
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

SAMPLE_DIR = DATA_DIR / "sample"                # export_sample.py 导出的样例数据
SAMPLE_CSV = SAMPLE_DIR / "meta.csv"
MODEL_PATH = OUTPUT_DIR / "egg_model.pt"        # 最近一次训练的最终模型
BASELINE_MODEL_PATH = OUTPUT_DIR / "baseline_model.pkl"   # 决策规则选了基线时，部署用的基线模型
PREDICTIONS_CSV = OUTPUT_DIR / "predictions.csv"                     # 每颗蛋每天一行
DECISION_CSV = OUTPUT_DIR / "predictions_decision_day.csv"          # 每颗蛋一行：决策日的判断
RUNS_DIR = OUTPUT_DIR / "runs"                  # 每次训练一个文件夹：配置、分折、结果、日志
CACHE_DIRNAME = "_cache"                        # 预处理缓存，建在 meta.csv 旁边，可以随时删

# 伊利诺伊大学公开的受精鉴定数据（真实数据，只有平均光谱）
FERTILITY_CSV = DATA_DIR / "illinois_fertility" / "spectra_and_reference_parameters.csv"
BASELINE_RESULT_CSV = OUTPUT_DIR / "baseline_fertility.csv"

HOLDOUT_LOCK_NAME = "holdout_lock.json"         # 独立测试集查看记录；真实数据放在 meta.csv 旁的 _cache 里
SYNTH_HOLDOUT_LOCK = OUTPUT_DIR / HOLDOUT_LOCK_NAME   # 模拟数据的
