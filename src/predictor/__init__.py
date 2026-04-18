"""
锅炉预测模块 - 简洁清晰的建模架构

模块结构：
- config.py: 统一配置 ✓
- dataset.py: 数据处理 + 增益计算 ✓
- model.py: 预测模型 ✓
- trainer.py: 训练器 ✓
- loss.py: 损失函数 ✓
- train.py: 训练脚本 ✓
- optimizer.py: MPC控制器 ✓
- utils.py: 工具函数 ✓

使用方式：
    from src.predictor import Config, BoilerDataset, BoilerPredictor, MPCOptimizer
    from src.predictor.config import TARGET_VARS, CONTROL_VARS
"""

# 配置模块
from .config import (
    Config,
    TARGET_VARS,
    CONTROL_VARS,
    PRESSURE_VARS,
    OXYGEN_VARS,
    PRESSURE_IDEAL,
    OXYGEN_IDEAL,
    get_state_vars,
)

# 数据处理模块
from .dataset import (
    BoilerDataset,
    GainEstimator,
    RevINNormalizer,
)

# 模型模块
from .model import (
    BoilerPredictor,
    create_model,
)

# 损失函数模块
from .loss import (
    PredictionLoss,
    SMAPELoss,
    create_loss_fn,
)

# 训练器模块
from .trainer import (
    Trainer,
    create_trainer,
)

# MPC控制器模块
from .optimizer import (
    MPCOptimizer,
    create_optimizer,
)

# 工具模块
from .utils import (
    get_logger,
    save_json,
    load_json,
)

__all__ = [
    # 配置
    "Config",
    "TARGET_VARS",
    "CONTROL_VARS",
    "PRESSURE_VARS",
    "OXYGEN_VARS",
    "PRESSURE_IDEAL",
    "OXYGEN_IDEAL",
    "get_state_vars",
    # 数据处理
    "BoilerDataset",
    "GainEstimator",
    "RevINNormalizer",
    # 模型
    "BoilerPredictor",
    "create_model",
    # 损失函数
    "PredictionLoss",
    "SMAPELoss",
    "create_loss_fn",
    # 训练器
    "Trainer",
    "create_trainer",
    # MPC控制器
    "MPCOptimizer",
    "create_optimizer",
    # 工具
    "get_logger",
    "save_json",
    "load_json",
]