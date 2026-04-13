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


def physics_guided_loss(
    y_true,
    y_pred,
    smoothness_weight: float = 0.001,
    range_weight: float = 0.01,
    p_range: tuple[float, float] | None = None,
    o_range: tuple[float, float] | None = None,
):
    """
    物理约束损失函数

    包含：
    1. 基础预测损失（Huber + MSE）
    2. 平滑性约束（惩罚预测变化偏离真实变化）
    3. 范围约束（惩罚预测值超出合理范围）

    Args:
        y_true: 真实值 (batch, output_steps, 2)
        y_pred: 预测值 (batch, output_steps, 2)
        smoothness_weight: 平滑性约束权重
        range_weight: 范围约束权重
        p_range: 负压的合理范围（标准化后的值），默认使用 config.NORMALIZED_RANGES
        o_range: 含氧量的合理范围（标准化后的值），默认使用 config.NORMALIZED_RANGES
    """
    # 延迟导入避免循环依赖
    from src.utils.config import NORMALIZED_RANGES

    # 使用 config 中标准化后的物理范围作为默认值
    if p_range is None:
        p_range = NORMALIZED_RANGES["pressure_constraint"]
    if o_range is None:
        o_range = NORMALIZED_RANGES["oxygen_constraint"]
    # 分离目标变量
    y_true_p = y_true[:, :, 0:1]  # 负压
    y_true_o = y_true[:, :, 1:2]  # 含氧量
    y_pred_p = y_pred[:, :, 0:1]
    y_pred_o = y_pred[:, :, 1:2]

    # 基础损失
    loss_p = _huber(y_true_p, y_pred_p)
    loss_o = _mse(y_true_o, y_pred_o)
    base_loss = loss_p + loss_o

    # 平滑性约束：惩罚预测变化偏离真实变化（而非简单惩罚变化）
    true_p_change = y_true_p[:, 1:, :] - y_true_p[:, :-1, :]
    pred_p_change = y_pred_p[:, 1:, :] - y_pred_p[:, :-1, :]
    true_o_change = y_true_o[:, 1:, :] - y_true_o[:, :-1, :]
    pred_o_change = y_pred_o[:, 1:, :] - y_pred_o[:, :-1, :]

    # 预测变化偏离真实变化的程度
    p_smoothness = tf.reduce_mean(tf.abs(pred_p_change - true_p_change))
    o_smoothness = tf.reduce_mean(tf.abs(pred_o_change - true_o_change))
    smoothness = p_smoothness + o_smoothness

    # 范围约束：惩罚超出合理范围的预测值
    # 使用 ReLU: max(0, value - upper) + max(0, lower - value)
    p_lower, p_upper = p_range
    o_lower, o_upper = o_range

    # 超出上限的部分
    p_above_upper = tf.nn.relu(y_pred_p - p_upper)
    # 超出下限的部分
    p_below_lower = tf.nn.relu(p_lower - y_pred_p)
    p_range_loss = tf.reduce_mean(p_above_upper + p_below_lower)

    o_above_upper = tf.nn.relu(y_pred_o - o_upper)
    o_below_lower = tf.nn.relu(o_lower - y_pred_o)
    o_range_loss = tf.reduce_mean(o_above_upper + o_below_lower)

    range_loss = p_range_loss + o_range_loss

    return base_loss + smoothness_weight * smoothness + range_weight * range_loss


# =============================================================================
# 改进的物理约束损失函数 v2
# =============================================================================

_huber_v2 = tf.keras.losses.Huber(delta=0.5, reduction='sum_over_batch_size')
_mse_v2 = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')


def physics_guided_loss_v2(
    y_true,
    y_pred,
    smoothness_weight: float = 0.0001,
    range_weight: float = 0.01,
    step_weights: list[float] | None = None,
    output_steps: int = 10,  # 通过参数传入，而非动态获取
    p_range: tuple[float, float] | None = None,
    o_range: tuple[float, float] | None = None,
):
    """
    改进的物理约束损失函数 v2

    改进点：
    1. 加权各预测步（近期步更重要）
    2. 平滑性约束改为相对变化匹配（而非绝对变化惩罚）
    3. 使用更小的Huber delta，增强对大误差的敏感度

    Args:
        y_true: 真实值 (batch, output_steps, 2)
        y_pred: 预测值 (batch, output_steps, 2)
        smoothness_weight: 平滑性约束权重（降低以减少限制）
        range_weight: 范围约束权重
        step_weights: 各步权重列表，近期步权重更高
        output_steps: 输出步数（必须显式传入）
        p_range: 负压的合理范围（标准化后的值）
        o_range: 含氧量的合理范围（标准化后的值）
    """
    from src.utils.config import NORMALIZED_RANGES

    if p_range is None:
        p_range = NORMALIZED_RANGES["pressure_constraint"]
    if o_range is None:
        o_range = NORMALIZED_RANGES["oxygen_constraint"]

    # 默认步数权重：线性递减
    if step_weights is None:
        step_weights = [1.0 - 0.05 * i for i in range(output_steps)]

    # 确保权重列表长度足够
    if len(step_weights) < output_steps:
        step_weights = list(step_weights) + [0.5] * (output_steps - len(step_weights))

    step_weights_tensor = tf.constant(step_weights[:output_steps], dtype=tf.float32)
    step_weights_tensor = tf.reshape(step_weights_tensor, [1, output_steps, 1])

    # 分离目标变量
    y_true_p = y_true[:, :, 0:1]  # 负压
    y_true_o = y_true[:, :, 1:2]  # 含氧量
    y_pred_p = y_pred[:, :, 0:1]
    y_pred_o = y_pred[:, :, 1:2]

    # 加权基础损失
    # 负压用Huber（对小误差更宽容，大误差更敏感）
    weighted_true_p = y_true_p * step_weights_tensor
    weighted_pred_p = y_pred_p * step_weights_tensor
    loss_p = _huber_v2(weighted_true_p, weighted_pred_p)

    # 含氧量用MSE
    weighted_true_o = y_true_o * step_weights_tensor
    weighted_pred_o = y_pred_o * step_weights_tensor
    loss_o = _mse_v2(weighted_true_o, weighted_pred_o)

    base_loss = loss_p + loss_o

    # 改进的平滑性约束：只惩罚预测变化方向与真实不一致
    # 计算真实变化方向（正/负）
    true_p_change = y_true_p[:, 1:, :] - y_true_p[:, :-1, :]
    pred_p_change = y_pred_p[:, 1:, :] - y_pred_p[:, :-1, :]
    true_o_change = y_true_o[:, 1:, :] - y_true_o[:, :-1, :]
    pred_o_change = y_pred_o[:, 1:, :] - y_pred_o[:, :-1, :]

    # 变化方向一致性损失：当真实变化方向与预测不同时惩罚
    # sign相同时不惩罚，不同时惩罚
    p_direction_loss = tf.reduce_mean(
        tf.nn.relu(-true_p_change * pred_p_change)  # 异号时惩罚
    )
    o_direction_loss = tf.reduce_mean(
        tf.nn.relu(-true_o_change * pred_o_change)
    )
    direction_loss = p_direction_loss + o_direction_loss

    # 变化幅度匹配损失：预测变化幅度应接近真实变化幅度
    p_change_ratio = tf.abs(pred_p_change) / (tf.abs(true_p_change) + 1e-6)
    o_change_ratio = tf.abs(pred_o_change) / (tf.abs(true_o_change) + 1e-6)
    # 惩罚变化幅度过大或过小
    p_change_mismatch = tf.reduce_mean(tf.abs(p_change_ratio - 1.0))
    o_change_mismatch = tf.reduce_mean(tf.abs(o_change_ratio - 1.0))
    change_mismatch = p_change_mismatch + o_change_mismatch

    # 综合平滑性损失（权重较低）
    smoothness = direction_loss + 0.1 * change_mismatch

    # 范围约束
    p_lower, p_upper = p_range
    o_lower, o_upper = o_range

    p_above_upper = tf.nn.relu(y_pred_p - p_upper)
    p_below_lower = tf.nn.relu(p_lower - y_pred_p)
    p_range_loss = tf.reduce_mean(p_above_upper + p_below_lower)

    o_above_upper = tf.nn.relu(y_pred_o - o_upper)
    o_below_lower = tf.nn.relu(o_lower - y_pred_o)
    o_range_loss = tf.reduce_mean(o_above_upper + o_below_lower)

    range_loss = p_range_loss + o_range_loss

    return base_loss + smoothness_weight * smoothness + range_weight * range_loss


# =============================================================================
# 改进的物理约束损失函数 v3 - 强制惩罚恒定预测
# =============================================================================

_mse_v3 = tf.keras.losses.MeanSquaredError(reduction='sum_over_batch_size')


def physics_guided_loss_v3(
    y_true,
    y_pred,
    smoothness_weight: float = 0.001,
    range_weight: float = 0.01,
    change_weight: float = 0.5,  # 变化幅度匹配权重（提高）
    variance_weight: float = 0.1,  # 预测方差约束权重
    step_weights: list[float] | None = None,
    output_steps: int = 10,
    p_range: tuple[float, float] | None = None,
    o_range: tuple[float, float] | None = None,
):
    """
    改进的物理约束损失函数 v3 - 解决恒定预测问题

    核心改进：
    1. 预测方差约束 - 强制惩罚恒定预测（预测值变化太小）
    2. 绝对变化损失 - 直接惩罚预测变化与真实变化的差距
    3. 提高变化幅度匹配权重 - 从 0.1 提高到 0.5
    4. 两变量独立权重 - 负压变化约束更强

    Args:
        y_true: 真实值 (batch, output_steps, 2)
        y_pred: 预测值 (batch, output_steps, 2)
        smoothness_weight: 平滑性约束权重
        range_weight: 范围约束权重
        change_weight: 变化幅度匹配权重（提高到 0.5）
        variance_weight: 预测方差约束权重（惩罚恒定预测）
        step_weights: 各步权重列表
        output_steps: 输出步数
        p_range: 负压的合理范围（标准化后的值）
        o_range: 含氧量的合理范围（标准化后的值）
    """
    from src.utils.config import NORMALIZED_RANGES

    if p_range is None:
        p_range = NORMALIZED_RANGES["pressure_constraint"]
    if o_range is None:
        o_range = NORMALIZED_RANGES["oxygen_constraint"]

    if step_weights is None:
        step_weights = [1.0 - 0.05 * i for i in range(output_steps)]

    if len(step_weights) < output_steps:
        step_weights = list(step_weights) + [0.5] * (output_steps - len(step_weights))

    step_weights_tensor = tf.constant(step_weights[:output_steps], dtype=tf.float32)
    step_weights_tensor = tf.reshape(step_weights_tensor, [1, output_steps, 1])

    # 分离目标变量
    y_true_p = y_true[:, :, 0:1]  # 负压
    y_true_o = y_true[:, :, 1:2]  # 含氧量
    y_pred_p = y_pred[:, :, 0:1]
    y_pred_o = y_pred[:, :, 1:2]

    # === 1. 加权基础损失 ===
    # 负压用 MSE（而非 Huber，避免对小变化不敏感）
    weighted_true_p = y_true_p * step_weights_tensor
    weighted_pred_p = y_pred_p * step_weights_tensor
    loss_p = _mse_v3(weighted_true_p, weighted_pred_p)

    # 含氧量用 MSE
    weighted_true_o = y_true_o * step_weights_tensor
    weighted_pred_o = y_pred_o * step_weights_tensor
    loss_o = _mse_v3(weighted_true_o, weighted_pred_o)

    base_loss = loss_p + loss_o

    # === 2. 绝对变化损失（核心改进）===
    # 直接惩罚预测变化与真实变化的差距，而非比例
    true_p_change = y_true_p[:, 1:, :] - y_true_p[:, :-1, :]
    pred_p_change = y_pred_p[:, 1:, :] - y_pred_p[:, :-1, :]
    true_o_change = y_true_o[:, 1:, :] - y_true_o[:, :-1, :]
    pred_o_change = y_pred_o[:, 1:, :] - y_pred_o[:, :-1, :]

    # 绝对变化损失：预测变化应该接近真实变化
    # 当预测恒定时，pred_change = 0，这个损失很大
    p_change_abs_loss = tf.reduce_mean(tf.abs(pred_p_change - true_p_change))
    o_change_abs_loss = tf.reduce_mean(tf.abs(pred_o_change - true_o_change))

    # 负压变化损失权重更高（因为负压波动更大）
    change_abs_loss = 2.0 * p_change_abs_loss + o_change_abs_loss

    # === 3. 变化方向损失 ===
    # 当真实变化方向与预测不同时惩罚
    p_direction_loss = tf.reduce_mean(tf.nn.relu(-true_p_change * pred_p_change))
    o_direction_loss = tf.reduce_mean(tf.nn.relu(-true_o_change * pred_o_change))
    direction_loss = 2.0 * p_direction_loss + o_direction_loss

    # === 4. 预测方差约束（关键改进 - 强制惩罚恒定预测）===
    # 预测值的方差应该接近真实值的方差
    # 恒定预测的方差 = 0，会被大力惩罚

    # 计算真实值的变化范围（标准差）
    true_p_std = tf.math.reduce_std(y_true_p, axis=1)  # (batch, 1)
    true_o_std = tf.math.reduce_std(y_true_o, axis=1)

    # 计算预测值的变化范围
    pred_p_std = tf.math.reduce_std(y_pred_p, axis=1)
    pred_o_std = tf.math.reduce_std(y_pred_o, axis=1)

    # 惩罚预测方差小于真实方差（恒定预测方差接近0）
    # 使用 ReLU 只惩罚预测方差太小的情况
    p_variance_loss = tf.reduce_mean(tf.nn.relu(true_p_std - pred_p_std - 0.1))  # 允许 10% 的差距
    o_variance_loss = tf.reduce_mean(tf.nn.relu(true_o_std - pred_o_std - 0.05))

    # 负压方差约束更重要
    variance_loss = 2.0 * p_variance_loss + o_variance_loss

    # === 5. 变化幅度匹配损失（比例形式）===
    p_change_ratio = tf.abs(pred_p_change) / (tf.abs(true_p_change) + 1e-6)
    o_change_ratio = tf.abs(pred_o_change) / (tf.abs(true_o_change) + 1e-6)
    p_change_mismatch = tf.reduce_mean(tf.abs(p_change_ratio - 1.0))
    o_change_mismatch = tf.reduce_mean(tf.abs(o_change_ratio - 1.0))
    change_mismatch = 2.0 * p_change_mismatch + o_change_mismatch

    # === 6. 综合变化约束 ===
    smoothness = (
        direction_loss +
        change_weight * change_abs_loss +  # 绝对变化损失（权重提高）
        change_weight * 0.5 * change_mismatch  # 比例变化损失
    )

    # === 7. 范围约束 ===
    p_lower, p_upper = p_range
    o_lower, o_upper = o_range

    p_above_upper = tf.nn.relu(y_pred_p - p_upper)
    p_below_lower = tf.nn.relu(p_lower - y_pred_p)
    p_range_loss = tf.reduce_mean(p_above_upper + p_below_lower)

    o_above_upper = tf.nn.relu(y_pred_o - o_upper)
    o_below_lower = tf.nn.relu(o_lower - y_pred_o)
    o_range_loss = tf.reduce_mean(o_above_upper + o_below_lower)

    range_loss = p_range_loss + o_range_loss

    # === 最终损失 ===
    return (
        base_loss +
        smoothness_weight * smoothness +
        range_weight * range_loss +
        variance_weight * variance_loss  # 添加方差约束
    )