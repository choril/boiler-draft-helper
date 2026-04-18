"""
MPC控制器模块

功能：
1. 基于预测模型优化控制输入
2. 考虑物理约束（理想负压、理想含氧）
3. 使用控制增益信息

使用方式：
    from src.predictor.optimizer import MPCOptimizer

    optimizer = MPCOptimizer(model, config, gains)
    optimal_control = optimizer.optimize(current_state, target_setpoint)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, List
from scipy.optimize import minimize

from .config import Config, PRESSURE_IDEAL, OXYGEN_IDEAL, PRESSURE_VARS, OXYGEN_VARS
from .model import BoilerPredictor
from .utils import get_logger

logger = get_logger(__name__)


class MPCOptimizer:
    """MPC优化控制器

    基于预测模型优化控制输入，使预测输出接近理想值。

    优化目标：
        min Σ_h ||y_pred[h] - y_ideal||² + λ_u ||u - u_current||²

    其中：
        y_pred = model(x_hist, u)  # 预测模型
        y_ideal: 理想负压/含氧值
        u_current: 当前控制输入（平滑约束）
        λ_u: 控制平滑权重
    """

    def __init__(
        self,
        model: BoilerPredictor,
        config: Config,
        gains: Optional[Dict[str, np.ndarray]] = None,
    ):
        """
        Args:
            model: 预测模型
            config: 配置对象
            gains: 控制增益字典（可选，用于初始化优化）
        """
        self.model = model
        self.config = config
        self.gains = gains

        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        # 理想值
        self.pressure_ideal = PRESSURE_IDEAL  # -115 Pa
        self.oxygen_ideal = OXYGEN_IDEAL      # 2.0%

        # 优化参数
        self.control_smooth_weight = config.mpc.control_smooth_weight
        self.max_control_change = config.mpc.max_control_change

        # 控制约束（来自config）
        self.u_min = config.mpc.u_min
        self.u_max = config.mpc.u_max

        logger.info(f"MPCOptimizer初始化:")
        logger.info(f"  理想负压: {self.pressure_ideal} Pa")
        logger.info(f"  理想含氧: {self.oxygen_ideal}%")
        logger.info(f"  控制平滑权重: {self.control_smooth_weight}")

    def predict(
        self,
        encoder_input: np.ndarray,
        control_sequence: np.ndarray,
    ) -> np.ndarray:
        """预测未来输出

        Args:
            encoder_input: 历史窗口数据 (L, n_features)
            control_sequence: 未来控制序列 (H, n_u)

        Returns:
            prediction: 预测输出 (H, n_y)
        """
        # 转换为Tensor
        encoder_tensor = torch.FloatTensor(encoder_input).unsqueeze(0).to(self.device)
        control_tensor = torch.FloatTensor(control_sequence).unsqueeze(0).to(self.device)

        # 预测
        with torch.no_grad():
            pred = self.model.predict(encoder_tensor, control_tensor)

        return pred.squeeze(0).cpu().numpy()

    def objective_function(
        self,
        u_flat: np.ndarray,
        encoder_input: np.ndarray,
        current_control: np.ndarray,
        target_pressure: Optional[float] = None,
        target_oxygen: Optional[float] = None,
    ) -> float:
        """优化目标函数

        Args:
            u_flat: 待优化的控制序列（扁平化）
            encoder_input: 历史窗口数据
            current_control: 当前控制输入
            target_pressure: 目标负压（可选）
            target_oxygen: 目标含氧（可选）

        Returns:
            loss: 总损失
        """
        H = self.config.H
        n_u = self.config.n_u

        # 恢复控制序列形状
        u_sequence = u_flat.reshape(H, n_u)

        # 预测输出
        prediction = self.predict(encoder_input, u_sequence)  # (H, n_y)

        # 目标值
        pressure_target = target_pressure or self.pressure_ideal
        oxygen_target = target_oxygen or self.oxygen_ideal

        # 计算损失
        loss = 0.0

        # 1. 负压偏离损失（权重高）
        pressure_pred = prediction[:, :4]  # 前4列为负压
        pressure_loss = np.mean((pressure_pred - pressure_target) ** 2)
        loss += 2.0 * pressure_loss  # 负压权重更高

        # 2. 含氧偏离损失
        oxygen_pred = prediction[:, 4:7]  # 后3列为含氧
        oxygen_loss = np.mean((oxygen_pred - oxygen_target) ** 2)
        loss += oxygen_loss

        # 3. 控制平滑损失
        control_smooth_loss = np.mean((u_sequence - current_control) ** 2)
        loss += self.control_smooth_weight * control_smooth_loss

        return loss

    def optimize(
        self,
        encoder_input: np.ndarray,
        current_control: np.ndarray,
        target_pressure: Optional[float] = None,
        target_oxygen: Optional[float] = None,
        method: str = 'L-BFGS-B',
    ) -> Tuple[np.ndarray, float]:
        """优化控制序列

        Args:
            encoder_input: 历史窗口数据 (L, n_features)
            current_control: 当前控制输入 (H, n_u)
            target_pressure: 目标负压（可选，默认使用理想值）
            target_oxygen: 目标含氧（可选）
            method: 优化方法

        Returns:
            optimal_control: 优化后的控制序列 (H, n_u)
            optimal_loss: 最优损失值
        """
        H = self.config.H
        n_u = self.config.n_u

        # 初始值：当前控制序列
        u_init = current_control.flatten()

        # 控制约束
        bounds = []
        for h in range(H):
            for i in range(n_u):
                u_min_i = self.u_min[i] if self.u_min is not None else -np.inf
                u_max_i = self.u_max[i] if self.u_max is not None else np.inf
                # 控制变化约束
                u_min_i = max(u_min_i, current_control[h, i] - self.max_control_change)
                u_max_i = min(u_max_i, current_control[h, i] + self.max_control_change)
                bounds.append((u_min_i, u_max_i))

        # 优化
        result = minimize(
            self.objective_function,
            u_init,
            args=(encoder_input, current_control, target_pressure, target_oxygen),
            method=method,
            bounds=bounds,
            options={'maxiter': 100, 'disp': False}
        )

        # 恢复形状
        optimal_control = result.x.reshape(H, n_u)

        return optimal_control, result.fun

    def compute_gain_based_control(
        self,
        current_pressure: float,
        current_oxygen: float,
        current_control: np.ndarray,
    ) -> np.ndarray:
        """基于控制增益计算控制调整

        Args:
            current_pressure: 当前负压均值
            current_oxygen: 当前含氧均值
            current_control: 当前控制输入

        Returns:
            suggested_control: 建议的控制调整
        """
        if self.gains is None:
            logger.warning("未提供控制增益，返回当前控制")
            return current_control

        # 计算偏差
        pressure_error = self.pressure_ideal - current_pressure
        oxygen_error = self.oxygen_ideal - current_oxygen

        # 基于增益计算控制调整（简化版）
        # 假设：引风机开度↑ → 负压↑（变得更负）
        #       二次风机开度↑ → 含氧↑

        suggested_control = current_control.copy()

        # 负压调节（使用引风机）
        if 'induced_draft' in self.gains:
            gain = self.gains['induced_draft']
            # gain: 负压变化 / 开度变化
            control_adjust = pressure_error / gain
            suggested_control[:, 0] += control_adjust  # 引风机开度索引

        # 含氧调节（使用二次风机）
        if 'secondary_air' in self.gains:
            gain = self.gains['secondary_air']
            control_adjust = oxygen_error / gain
            suggested_control[:, 1] += control_adjust  # 二次风机开度索引

        return suggested_control


def create_optimizer(
    model: BoilerPredictor,
    config: Config,
    gains: Optional[Dict[str, np.ndarray]] = None,
) -> MPCOptimizer:
    """创建MPC优化器（工厂函数）

    Args:
        model: 预测模型
        config: 配置对象
        gains: 控制增益（可选）

    Returns:
        optimizer: MPCOptimizer实例
    """
    return MPCOptimizer(model, config, gains)


__all__ = [
    "MPCOptimizer",
    "create_optimizer",
]