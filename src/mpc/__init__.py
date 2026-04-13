"""
MPC模块 - 模型预测控制

组件：
1. Controller: MPC控制器（滚动优化）
2. Optimizer: 贝叶斯优化器（候选控制评估）
3. SafetyMonitor: 安全监控（约束检查）
4. PressureJumpType: 负压跳变类型检测（新增）
"""

from src.mpc.controller import MPCController
from src.mpc.optimizer import BayesianMPCOptimizer
from src.mpc.safety_monitor import (
    SafetyMonitor,
    SafetyLevel,
    PressureJumpType,  # 新增
    PRESSURE_JUMP_THRESHOLDS,  # 新增
)

__all__ = [
    "MPCController",
    "BayesianMPCOptimizer",
    "SafetyMonitor",
    "SafetyLevel",
    "PressureJumpType",  # 新增
    "PRESSURE_JUMP_THRESHOLDS",  # 新增
]