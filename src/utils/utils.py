import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf

warnings.filterwarnings("ignore")


def load_data(
    data_path: str = "output/feature_matrix_balanced.feather",
    param_path: str = "output/param_dict.json",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_feather(data_path)
    with open(param_path, "r", encoding="utf-8") as f:
        param_dict = json.load(f)
    return df, param_dict


def save_json(data: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_divide(
    a: np.ndarray | pd.Series,
    b: np.ndarray | pd.Series,
    eps: float = 1e-6,
    clip_value: float | None = 1000,
) -> np.ndarray | pd.Series:
    """
    安全除法，防止除零和极端值

    Args:
        a: 被除数
        b: 除数
        eps: 防止除零的小值
        clip_value: 结果裁剪阈值，None表示不裁剪
    """
    result = a / (b + eps)
    if clip_value is not None:
        result = np.clip(result, -clip_value, clip_value)
    return result


def rolling_stat(series: pd.Series, window: int, stat: str) -> pd.Series:
    rolling = series.rolling(window=window, min_periods=1)
    stat_map = {
        "mean": rolling.mean,
        "std": rolling.std,
        "max": rolling.max,
        "min": rolling.min,
        "median": rolling.median,
        "skew": rolling.skew,
        "kurt": rolling.kurt,
    }
    if stat in stat_map:
        return stat_map[stat]()
    if stat == "range":
        return rolling.max() - rolling.min()
    if stat == "iqr":
        return rolling.apply(
            lambda x: np.percentile(x, 75) - np.percentile(x, 25) if len(x) > 0 else 0
        )
    if stat == "cv":
        return rolling.std() / (rolling.mean() + eps)
    raise ValueError(f"Unknown stat: {stat}")


def compute_in_range_ratio(data: pd.Series, low: float, high: float) -> float:
    return ((data >= low) & (data <= high)).sum() / len(data) * 100


def classify_parameters(param_dict: dict[str, Any]) -> dict[str, list[str]]:
    categories: dict[str, list[str]] = {"控制变量": [], "状态变量": [], "其他变量": []}
    control_keywords = [
        "给煤",
        "煤量",
        "控制",
        "指令",
        "设定",
        "转速",
        "开度",
        "阀门",
        "联络阀",
        "液偶",
    ]
    other_keywords = ["时间戳", "源文件"]

    for param_id, info in param_dict.items():
        desc = info.get("描述", "") or ""
        name = info.get("简称", "") or ""
        if any(kw in desc or kw in name for kw in control_keywords):
            categories["控制变量"].append(param_id)
        elif any(kw in desc or kw in name for kw in other_keywords):
            categories["其他变量"].append(param_id)
        else:
            categories["状态变量"].append(param_id)
    return categories


def print_section(title: str, width: int = 80) -> None:
    print("\n" + "=" * width)
    print(title)
    print("=" * width)


def setup_gpu(gpus: str = "all", memory_growth: bool = True) -> tf.distribute.Strategy:

    """配置GPU"""
    gpus_list = tf.config.list_physical_devices("GPU")

    if not gpus_list:
        print("未检测到GPU，使用CPU训练")
        return tf.distribute.get_strategy()

    if gpus != "all":
        gpu_ids = [int(g.strip()) for g in gpus.split(",")]
        gpus_list = [gpus_list[i] for i in gpu_ids if i < len(gpus_list)]

    tf.config.set_visible_devices(gpus_list, "GPU")

    if memory_growth:
        for gpu in gpus_list:
            tf.config.experimental.set_memory_growth(gpu, True)

    print(f"可用GPU: {len(gpus_list)} 个")
    for i, gpu in enumerate(gpus_list):
        print(f"  - {gpu.name}")

    if len(gpus_list) > 1:
        devices = [f"/GPU:{i}" for i in range(len(gpus_list))]
        print(f"使用 MirroredStrategy 多GPU训练")
        strategy = tf.distribute.MirroredStrategy(
            devices=devices,
            cross_device_ops=tf.distribute.NcclAllReduce()
        )
    else:
        strategy = tf.distribute.get_strategy()
        print(f"使用单GPU训练")

    return strategy


# =============================================================================
# 物理约束损失函数
# =============================================================================

_huber = tf.keras.losses.Huber(delta=1.0, reduction='sum_over_batch_size')
_mse = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')


def physics_guided_loss(y_true, y_pred, smoothness_weight: float = 0.001):
    """
    物理约束损失函数

    包含：
    1. 基础预测损失（Huber + MSE）
    2. 平滑性约束（防止预测剧烈跳变）
    """
    # 分离目标变量
    y_true_p = y_true[:, :, 0:1]  # 负压
    y_true_o = y_true[:, :, 1:2]  # 含氧量
    y_pred_p = y_pred[:, :, 0:1]
    y_pred_o = y_pred[:, :, 1:2]

    # 基础损失
    loss_p = _huber(y_true_p, y_pred_p)
    loss_o = _mse(y_true_o, y_pred_o)
    base_loss = loss_p + loss_o

    # 平滑性约束
    p_change = y_pred_p[:, 1:, :] - y_pred_p[:, :-1, :]
    o_change = y_pred_o[:, 1:, :] - y_pred_o[:, :-1, :]
    smoothness = tf.reduce_mean(tf.abs(p_change)) + tf.reduce_mean(tf.abs(o_change))

    return base_loss + smoothness_weight * smoothness