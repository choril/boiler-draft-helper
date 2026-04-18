"""
统一配置模块

包含：
- 变量定义（Y/U/X）
- 理想目标值
- 窗口参数
- 模型参数
- 标准化策略
- 训练参数
- MPC优化参数

使用方式：
    from src.predictor.config import Config, TARGET_VARS, CONTROL_VARS, ...
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# ========== 项目路径 ==========

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "output" / "all_data_cleaned.feather"
MODEL_DIR = PROJECT_ROOT / "output" / "models" / "predictor_v2"


# ========== 变量定义 ==========

# 目标变量 Y（7维）- 负压4点 + 含氧3点
TARGET_VARS: List[str] = [
    # 负压测点（Pa）
    "2BK10CP004",  # 炉膛压力1（主测点）
    "2BK2CP004",   # 炉膛压力2
    "2BK10CP005",  # 炉膛压力3
    "2BK2CP005",   # 炐膛压力4
    # 含氧测点（%）
    "2BK10CQ1",    # 含氧量1（主测点）
    "2BK2CQ1",     # 含氧量2
    "2BK2CQ2",     # 含氧量3
]

# 负压变量（用于提取）
PRESSURE_VARS: List[str] = TARGET_VARS[:4]

# 含氧变量（用于提取）
OXYGEN_VARS: List[str] = TARGET_VARS[4:]

# 主测点（用于优化目标）
MAIN_PRESSURE_VAR = "2BK10CP004"
MAIN_OXYGEN_VAR = "2BK10CQ1"


# 控制变量 U（7维）- 风机频率 + 给煤量
CONTROL_VARS: List[str] = [
    # 引风机（调负压）
    "DPU61AX107",   # 引风机A频率
    "DPU61AX108",   # 引风机B频率
    # 二次风机（调含氧）
    "2LA30A12C11",  # 二次风机A频率
    "2LA40A12C11",  # 二次风机B频率
    # 一次风机（调负荷）
    "2LA10A12C11",  # 一次风机A频率
    "2LA20A12C11",  # 一次风机B频率
    # 给煤量
    "D62AX002",     # 给煤量
]

# 控制变量分组
INDUCED_FAN_VARS = ["DPU61AX107", "DPU61AX108"]      # 引风机（调负压）
SECONDARY_FAN_VARS = ["2LA30A12C11", "2LA40A12C11"]  # 二次风机（调含氧）
PRIMARY_FAN_VARS = ["2LA10A12C11", "2LA20A12C11"]    # 一次风机（调负荷）
COAL_FEED_VAR = ["D62AX002"]                         # 给煤量

# 控制变量中文名称
CONTROL_NAMES: Dict[str, str] = {
    "DPU61AX107": "引风机A频率",
    "DPU61AX108": "引风机B频率",
    "2LA30A12C11": "二次风机A频率",
    "2LA40A12C11": "二次风机B频率",
    "2LA10A12C11": "一次风机A频率",
    "2LA20A12C11": "一次风机B频率",
    "D62AX002": "给煤量",
}


# ========== 理想目标值 ==========

# 负压理想值（Pa）
PRESSURE_IDEAL: float = -115.0

# 含氧理想值（%）
OXYGEN_IDEAL: float = 2.0

# 负压安全范围（Pa）
PRESSURE_SAFE_RANGE: Tuple[float, float] = (-200.0, -20.0)

# 含氧安全范围（%）
OXYGEN_SAFE_RANGE: Tuple[float, float] = (1.0, 6.0)


# ========== 物理约束参数（动态计算） ==========

# 注意：以下增益参数应从实际数据动态计算，不应硬编码
# 在dataset.py的GainEstimator类中自动计算
# 这里仅作为默认值/占位符，实际使用时会被覆盖

# 引风机→负压增益（Pa/Hz）- 频率增加，负压下降（负相关）
INDUCED_FAN_PRESSURE_GAIN_DEFAULT: float = -3.0

# 二次风机→含氧增益（%/Hz）- 频率增加，含氧上升（正相关）
SECONDARY_FAN_OXYGEN_GAIN_DEFAULT: float = 0.05


@dataclass
class GainConfig:
    """增益估计配置"""
    # 增益计算窗口（分钟）
    estimation_window: int = 60

    # 增益计算方法
    method: str = "correlation"  # 'correlation' | 'regression' | 'steady_state'

    # 最小相关性阈值（低于此值认为无显著关系）
    min_correlation: float = 0.3

    # 默认增益值（当数据不足以估计时使用）
    induced_fan_pressure_gain_default: float = INDUCED_FAN_PRESSURE_GAIN_DEFAULT
    secondary_fan_oxygen_gain_default: float = SECONDARY_FAN_OXYGEN_GAIN_DEFAULT

    # 动态计算的增益结果（运行时填充）
    induced_fan_pressure_gain: float = INDUCED_FAN_PRESSURE_GAIN_DEFAULT
    secondary_fan_oxygen_gain: float = SECONDARY_FAN_OXYGEN_GAIN_DEFAULT


# ========== 配置类 ==========

@dataclass
class WindowConfig:
    """窗口配置"""
    history_length: int = 15      # 历史窗口长度 L（分钟）
    prediction_horizon: int = 5   # 预测步长 H（分钟）
    sampling_interval: int = 60   # 采样间隔（秒）
    stride: int = 1               # 构建样本时的步长


@dataclass
class ModelConfig:
    """模型配置"""
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True


@dataclass
class TrainConfig:
    """训练配置"""
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 100
    early_stop_patience: int = 15
    weight_decay: float = 1e-4

    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Teacher forcing
    teacher_forcing_ratio: float = 0.5

    # 多步预测权重（基于数据分析的MSE权重方案）
    # MSE递增表示远期预测更难，应给远期更多关注
    # 分析结果: MSE权重 = [0.81, 0.93, 1.01, 1.09, 1.16]
    step_weights: List[float] = field(default_factory=lambda: [0.85, 0.95, 1.0, 1.05, 1.15])


@dataclass
class NormalizeConfig:
    """标准化配置"""
    mode: str = "revin"  # 'global' | 'revin'

    # RevIN最小标准差阈值（防止除零 + 控制极端值 + 减少窗口差异）
    # 增大阈值使更多窗口标准化一致，减少跳变/稳态窗口分布差异
    pressure_min_std: float = 30.0   # 负压最小标准差 Pa（增大到均值附近）
    oxygen_min_std: float = 2.0      # 含氧最小标准差 %（增大）


@dataclass
class LossConfig:
    """损失函数配置"""
    # MSE权重
    mse_weight: float = 1.0

    # 差分损失权重（强迫预测变化趋势，解决非平稳滞后问题）
    # 数据分析结果：含氧跳变比例21%，中等非平稳，推荐值0.1
    diff_weight: float = 0.1

    # 跳变加权（可选）
    jump_weight_threshold: float = 50.0  # 负压变化阈值 Pa
    jump_sample_weight: float = 3.0      # 跳变样本权重倍数


@dataclass
class MPCConfig:
    """MPC优化配置"""
    # 优化目标权重
    pressure_weight: float = 1.0
    oxygen_weight: float = 1.0
    control_change_weight: float = 0.1  # 控制变化惩罚

    # 安全约束权重
    safety_weight: float = 100.0

    # 优化算法
    n_evaluations: int = 30  # 贝叶斯优化评估次数

    # 控制变量变化幅度约束
    max_control_change: Dict[str, float] = field(default_factory=lambda: {
        "DPU61AX107": 5.0,   # 引风机最大变化 Hz
        "DPU61AX108": 5.0,
        "2LA30A12C11": 3.0,  # 二次风机最大变化 Hz
        "2LA40A12C11": 3.0,
        "2LA10A12C11": 3.0,  # 一次风机最大变化 Hz
        "2LA20A12C11": 3.0,
        "D62AX002": 2.0,     # 给煤量最大变化 t/h
    })


@dataclass
class Config:
    """总配置类"""
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    normalize: NormalizeConfig = field(default_factory=NormalizeConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    gain: GainConfig = field(default_factory=GainConfig)  # 新增

    # 数据路径
    data_path: Path = DATA_PATH
    model_dir: Path = MODEL_DIR

    # 设备
    device: str = "cuda"

    # 状态变量维度（动态设置）
    n_x: int = 38  # 默认值，实际会根据数据动态调整

    def __post_init__(self):
        """初始化后处理"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # 简写
        self.L = self.window.history_length
        self.H = self.window.prediction_horizon
        self.n_y = len(TARGET_VARS)
        self.n_u = len(CONTROL_VARS)

    def to_dict(self) -> Dict:
        """转换为字典（用于保存）"""
        return {
            "window": {
                "history_length": self.window.history_length,
                "prediction_horizon": self.window.prediction_horizon,
                "sampling_interval": self.window.sampling_interval,
                "stride": self.window.stride,
            },
            "model": {
                "hidden_size": self.model.hidden_size,
                "num_layers": self.model.num_layers,
                "dropout": self.model.dropout,
                "bidirectional": self.model.bidirectional,
            },
            "train": {
                "learning_rate": self.train.learning_rate,
                "batch_size": self.train.batch_size,
                "epochs": self.train.epochs,
                "early_stop_patience": self.train.early_stop_patience,
                "weight_decay": self.train.weight_decay,
                "teacher_forcing_ratio": self.train.teacher_forcing_ratio,
                "step_weights": self.train.step_weights,
            },
            "normalize": {
                "mode": self.normalize.mode,
                "pressure_min_std": self.normalize.pressure_min_std,
                "oxygen_min_std": self.normalize.oxygen_min_std,
            },
            "loss": {
                "mse_weight": self.loss.mse_weight,
                "diff_weight": self.loss.diff_weight,
                "jump_weight_threshold": self.loss.jump_weight_threshold,
                "jump_sample_weight": self.loss.jump_sample_weight,
            },
            "mpc": {
                "pressure_weight": self.mpc.pressure_weight,
                "oxygen_weight": self.mpc.oxygen_weight,
                "control_change_weight": self.mpc.control_change_weight,
                "safety_weight": self.mpc.safety_weight,
                "n_evaluations": self.mpc.n_evaluations,
                "max_control_change": self.mpc.max_control_change,
            },
            "gain": {
                "estimation_window": self.gain.estimation_window,
                "method": self.gain.method,
                "min_correlation": self.gain.min_correlation,
                "induced_fan_pressure_gain": self.gain.induced_fan_pressure_gain,
                "secondary_fan_oxygen_gain": self.gain.secondary_fan_oxygen_gain,
            },
            "data_path": str(self.data_path),
            "model_dir": str(self.model_dir),
            "device": self.device,
        }


# ========== 辅助函数 ==========

def get_state_vars(all_columns: List[str]) -> List[str]:
    """从所有列中筛选状态变量

    状态变量 = 所有列 - 目标变量 - 控制变量 - 时间/源文件
    """
    exclude = ["TIME", "source_file"] + TARGET_VARS + CONTROL_VARS
    return [col for col in all_columns if col not in exclude]


def classify_var(var_name: str) -> str:
    """判断变量类型"""
    if var_name in TARGET_VARS:
        return "target"
    elif var_name in CONTROL_VARS:
        return "control"
    elif var_name in ["TIME", "source_file"]:
        return "exclude"
    else:
        return "state"


def get_var_index(var_name: str, var_list: List[str]) -> int:
    """获取变量在列表中的索引"""
    return var_list.index(var_name)


# ========== 导出 ==========

__all__ = [
    # 路径
    "PROJECT_ROOT",
    "DATA_PATH",
    "MODEL_DIR",
    # 变量
    "TARGET_VARS",
    "PRESSURE_VARS",
    "OXYGEN_VARS",
    "MAIN_PRESSURE_VAR",
    "MAIN_OXYGEN_VAR",
    "CONTROL_VARS",
    "INDUCED_FAN_VARS",
    "SECONDARY_FAN_VARS",
    "PRIMARY_FAN_VARS",
    "COAL_FEED_VAR",
    "CONTROL_NAMES",
    # 理想值
    "PRESSURE_IDEAL",
    "OXYGEN_IDEAL",
    "PRESSURE_SAFE_RANGE",
    "OXYGEN_SAFE_RANGE",
    # 物理参数
    "INDUCED_FAN_PRESSURE_GAIN_DEFAULT",
    "SECONDARY_FAN_OXYGEN_GAIN_DEFAULT",
    # 配置类
    "WindowConfig",
    "ModelConfig",
    "TrainConfig",
    "NormalizeConfig",
    "LossConfig",
    "MPCConfig",
    "GainConfig",
    "Config",
    # 辅助函数
    "get_state_vars",
    "classify_var",
    "get_var_index",
]