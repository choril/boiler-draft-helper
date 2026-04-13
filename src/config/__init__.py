"""
配置模块 - 变量定义、物理约束、超参数
"""

from src.config.variables import *
from src.config.constraints import *
from src.config.hyperparams import *

__all__ = [
    # 变量定义
    "PRESSURE_VARIABLES",
    "OXYGEN_VARIABLES",
    "TARGET_VARIABLES",
    "CONTROL_VARIABLES",
    "STATE_VARIABLES",
    "ALL_VARIABLES",
    # 物理约束
    "CONTROL_CONSTRAINTS",
    "SAFETY_CONSTRAINTS",
    "IDEAL_RANGES",
    # 超参数
    "WINDOW_CONFIG",
    "MODEL_CONFIG",
    "MPC_CONFIG",
]