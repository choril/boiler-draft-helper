"""
超参数配置 - 窗口配置、模型配置、MPC配置

配置分类：
- 窗口配置：历史窗口长度、预测步长
- 模型配置：NARX-LSTM、TFT、代理模型的超参数
- MPC配置：优化算法参数、约束权重
"""

from typing import Final


# ========== 窗口配置 ==========

WINDOW_CONFIG: Final[dict] = {
    # 历史窗口长度（分钟）
    "history_length": 15,  # L = 15

    # 预测步长（分钟）
    "prediction_horizon": 5,  # H = 5

    # 数据采样间隔（秒）
    "sampling_interval": 60,  # 1分钟

    # 步长（构建序列时的步长，用于控制样本数量）
    "stride": 1,
}

# 简写常量
L: Final[int] = WINDOW_CONFIG["history_length"]       # 30
H: Final[int] = WINDOW_CONFIG["prediction_horizon"]   # 5


# ========== 数据划分配置 ==========

DATA_SPLIT_CONFIG: Final[dict] = {
    # 数据划分比例（按时间顺序）
    "train_ratio": 0.70,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # 训练集是否打乱顺序
    "shuffle_train": True,
}


# ========== NARX-LSTM 配置 ==========

NARX_LSTM_CONFIG: Final[dict] = {
    # 编码器配置
    "encoder_hidden_units": 256,  # 增大（128 → 256）
    "encoder_num_layers": 2,      # 增加层数（1 → 2）
    "encoder_bidirectional": True,

    # 解码器配置
    "decoder_hidden_units": 256,  # 增大（128 → 256）
    "decoder_num_layers": 2,      # 增加层数（1 → 2）

    # 输出配置
    "output_steps": H,  # 预测步数

    # 训练配置
    "dropout_rate": 0.2,
    "learning_rate": 0.001,
    "l2_reg": 1e-4,
    "batch_size": 64,
    "epochs": 100,
    "early_stop_patience": 15,

    # Teacher forcing配置
    "teacher_forcing_ratio": 0.5,  # 训练时使用真实值的概率

    # 多步预测权重（近期权重更高）
    "step_weights": [1.0, 0.9, 0.8, 0.7, 0.6],  # H=5时的权重
}


# ========== 直接多步预测模型配置 ==========

DIRECT_PREDICTOR_CONFIG: Final[dict] = {
    "encoder_hidden": 256,
    "encoder_layers": 2,
    "d_model": 128,
    "n_heads": 4,
    "control_hidden": 64,
    "decoder_layers": 2,
    "output_steps": H,
    "dropout": 0.2,
    "use_residual": True,

    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "epochs": 100,
    "early_stop_patience": 15,

    "step_weights": [1.0, 0.95, 0.9, 0.85, 0.8],

    "physics_weight": 0.1,
    "scheduled_sampling": False,
}


# ========== TFT 配置 ==========

TFT_CONFIG: Final[dict] = {
    # 预测配置
    "prediction_length": H,
    "context_length": L * 2,  # 历史上下文长度

    # 模型架构
    "hidden_size": 64,
    "attention_heads": 4,
    "dropout": 0.1,

    # 变量选择
    "variable_selection": True,

    # 训练配置
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "early_stop_patience": 10,
}


# ========== 代理模型（MLP）配置 ==========

PROXY_MLP_CONFIG: Final[dict] = {
    # 网络结构
    "hidden_layers": [256, 128, 64],
    "activation": "relu",
    "dropout_rate": 0.1,

    # 输入输出
    "input_dim": None,  # 动态计算
    "output_steps": H,
    "output_dim": 7,  # 7个目标变量

    # 训练配置
    "learning_rate": 0.001,
    "batch_size": 256,
    "epochs": 30,
    "early_stop_patience": 5,

    # 物理约束权重
    "physics_weight": 0.1,
}


# ========== 物理约束损失配置 ==========

PHYSICS_LOSS_CONFIG: Final[dict] = {
    # 单调性约束权重
    "monotonicity_weight": 0.05,
    "monotonicity_pairs": [
        # (控制变量, 目标变量, 方向)
        # 引风机频率↑ → 负压↓（负相关）
        ("DPU61AX107", "2BK10CP004", "negative"),
        ("DPU61AX108", "2BK10CP004", "negative"),
        # 二次风机频率↑ → 含氧↑（正相关）
        ("2LA30A12C11", "2BK10CQ1", "positive"),
        ("2LA40A12C11", "2BK10CQ1", "positive"),
    ],

    # 边界约束权重
    "boundary_weight": 0.1,
    "pressure_boundary": (-200.0, -20.0),  # Pa
    "oxygen_boundary": (1.0, 6.0),         # %

    # 平滑性约束权重
    "smoothness_weight": 0.01,

    # 空间一致性约束权重
    "spatial_weight": 0.02,
    "pressure_consistency_threshold": 20.0,  # Pa (4个负压测点差异阈值)
}


# ========== MPC 配置 ==========

MPC_CONFIG: Final[dict] = {
    # 优化配置
    "horizon": H,  # 预测步长
    "n_evaluations": 30,  # 贝叶斯优化评估次数
    "n_initial_samples": 5,  # 初始随机采样次数

    # 目标函数权重
    "pressure_weight": 1.0,
    "oxygen_weight": 1.0,
    "control_change_weight": 0.1,  # 控制变化惩罚权重

    # 约束权重（软约束）
    "safety_weight": 100.0,  # 安全约束权重（高）
    "boundary_weight": 10.0,  # 边界约束权重

    # 分组优化配置（紧急工况）
    "group_optimization": {
        "enabled": True,
        "pressure_group": ["DPU61AX107", "DPU61AX108"],  # 引风机调负压
        "oxygen_group": ["2LA30A12C11", "2LA40A12C11"],  # 二次风机调氧
        "load_group": ["D62AX002", "2LA10A12C11", "2LA20A12C11"],  # 给煤+一次风调负荷
    },

    # 优化超时（秒）
    "optimization_timeout": 30.0,

    # 滚动执行策略
    "execute_only_first_step": True,  # 只执行第一步
}


# ========== 模型保存配置 ==========

SAVE_CONFIG: Final[dict] = {
    # 模型保存目录
    "model_dir": "output/models",

    # 各模型子目录
    "narx_lstm_dir": "narx_lstm",
    "tft_dir": "tft",
    "proxy_dir": "proxy",

    # 保存内容
    "save_weights": True,
    "save_config": True,
    "save_scaler": True,
}


# ========== 辅助函数 ==========

def get_narx_lstm_config(**overrides) -> dict:
    """获取NARX-LSTM配置，允许覆盖默认值

    Args:
        **overrides: 需要覆盖的配置项

    Returns:
        配置字典
    """
    config = NARX_LSTM_CONFIG.copy()
    config.update(overrides)
    return config


def get_tft_config(**overrides) -> dict:
    """获取TFT配置，允许覆盖默认值"""
    config = TFT_CONFIG.copy()
    config.update(overrides)
    return config


def get_proxy_config(**overrides) -> dict:
    """获取代理模型配置，允许覆盖默认值"""
    config = PROXY_MLP_CONFIG.copy()
    config.update(overrides)
    return config


def get_mpc_config(**overrides) -> dict:
    """获取MPC配置，允许覆盖默认值"""
    config = MPC_CONFIG.copy()
    config.update(overrides)
    return config


__all__ = [
    "WINDOW_CONFIG",
    "L",
    "H",
    "DATA_SPLIT_CONFIG",
    "NARX_LSTM_CONFIG",
    "DIRECT_PREDICTOR_CONFIG",
    "TFT_CONFIG",
    "PROXY_MLP_CONFIG",
    "PHYSICS_LOSS_CONFIG",
    "MPC_CONFIG",
    "SAVE_CONFIG",
    "get_narx_lstm_config",
    "get_direct_predictor_config",
    "get_tft_config",
    "get_proxy_config",
    "get_mpc_config",
]


def get_direct_predictor_config(**overrides) -> dict:
    """获取直接多步预测模型配置"""
    config = DIRECT_PREDICTOR_CONFIG.copy()
    config.update(overrides)
    return config