"""把预处理设置绑到模型上：训练时存进模型文件，判断新蛋时核对，不一致就拒绝。

模型看到的图像取决于校正常数、找蛋和裁剪的各项门槛、缩放尺寸，以及这些步骤的代码本身。
任何一项和训练时不同，模型看到的就不再是训练时那种数据，判断结果不可信。
"""
import hashlib
import json

import calibration
import real_data
import roi

CALIBRATION_SETTINGS = ("TARGET_SIZE", "MIN_REF_SPAN_RATIO", "MIN_REF_SNR", "REF_NOISE_FLOOR", "REFL_RANGE",
                        "MAX_OUT_OF_RANGE")
ROI_SETTINGS = ("ROI_MODE", "FIXED_ROI", "MIN_CONTRAST", "CONTRAST_NOISE_FLOOR", "MIN_EGG_BG_DIFF",
                "EGG_AREA_RANGE", "EGG_FILL_RANGE", "MAX_EGG_ASPECT", "ROI_CANDIDATES", "RING_WIDTH",
                "MIN_CROP_IN_MASK", "CROP_SHRINK", "MIN_CROP_SIDE", "HIST_BINS", "MAD_TO_SIGMA")
CODE_VERSION_KEY = "预处理代码版本"
CODE_FILES = "calibration.py、roi.py、real_data.py"


def preprocess_settings() -> dict:
    """现在生效的全部预处理设置（保存模型和核对时都用这一个函数）。值统一成 JSON 能表示的类型（元组变列表）。"""
    settings = {name: getattr(calibration, name) for name in CALIBRATION_SETTINGS}
    settings.update({name: getattr(roi, name) for name in ROI_SETTINGS})
    settings["MAT_VARIABLE"] = real_data.MAT_VARIABLE
    settings[CODE_VERSION_KEY] = hashlib.sha1(real_data._code_version()).hexdigest()[:16]
    return json.loads(json.dumps(settings))


def check_preprocess(trained: dict | None, source: str) -> None:
    """核对训练时存下的预处理设置和现在的是否一致；不一致、或旧模型没有记录，都直接报错。"""
    if trained is None:
        raise ValueError(f"{source} 没有记录预处理设置（旧版本训练的），无法确认预处理和训练时一致，请先重新训练")
    current = preprocess_settings()
    names = [n for n in dict.fromkeys([*trained, *current]) if n != CODE_VERSION_KEY]
    diffs = [f"  {n}：训练时 {trained.get(n, '（无）')}，现在 {current.get(n, '（无）')}"
             for n in names if trained.get(n) != current.get(n)]
    if diffs:
        raise ValueError(f"{source} 训练时的预处理设置和现在不同，模型看到的图像会和训练时不一样：\n"
                         + "\n".join(diffs) + "\n请重新训练，或把这些设置改回训练时的值")
    if trained.get(CODE_VERSION_KEY) != current[CODE_VERSION_KEY]:
        raise ValueError(f"预处理代码改过了（{CODE_FILES} 的内容和训练出 {source} 时不同），请重新训练")
