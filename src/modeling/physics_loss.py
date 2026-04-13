"""
物理约束损失函数 - PINN风格约束

约束类型：
1. 单调性约束：控制变量与目标变量的因果关系
2. 边界约束：预测值应在物理范围内
3. 平滑性约束：预测值时序应平滑
4. 空间一致性约束：多测点之间的一致性

用于：
- NARX-LSTM训练时作为额外损失
- 代理模型蒸馏时继承约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

from src.config.constraints import (
    PRESSURE_SAFETY_RANGE,
    OXYGEN_SAFETY_RANGE,
    PRESSURE_IDEAL_RANGE,
    OXYGEN_IDEAL_RANGE,
)
from src.config.variables import PRESSURE_VARIABLES, OXYGEN_VARIABLES
from src.config.hyperparams import PHYSICS_LOSS_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PhysicsConstraintLoss(nn.Module):
    """物理约束损失函数

    L_physics = λ_mono * L_mono + λ_boundary * L_boundary + λ_smooth * L_smooth + λ_spatial * L_spatial + λ_amp * L_amp

    新增变化幅度约束：解决负压突变预测滞后问题
    """

    def __init__(
        self,
        monotonicity_weight: float = 0.05,
        boundary_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        spatial_weight: float = 0.02,
        amplitude_weight: float = 0.15,  # 新增：变化幅度约束权重
        pressure_boundary: Tuple[float, float] = PRESSURE_SAFETY_RANGE,
        oxygen_boundary: Tuple[float, float] = OXYGEN_SAFETY_RANGE,
        pressure_amp_ratio: float = 3.0,  # 引风机变化1Hz → 负压变化约3Pa
        oxygen_amp_ratio: float = 0.05,  # 二次风机变化1Hz → 含氧变化约0.05%
    ):
        super().__init__()

        self.monotonicity_weight = monotonicity_weight
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight
        self.spatial_weight = spatial_weight
        self.amplitude_weight = amplitude_weight  # 新增

        # 边界约束
        self.pressure_low, self.pressure_high = pressure_boundary
        self.oxygen_low, self.oxygen_high = oxygen_boundary

        # 变化幅度比例（物理参数）
        self.pressure_amp_ratio = pressure_amp_ratio
        self.oxygen_amp_ratio = oxygen_amp_ratio

        # 单调性关系
        # 引风机频率↑ → 负压↓（负相关）
        # 二次风机频率↑ → 含氧↑（正相关）
        self.monotonic_pairs = [
            # (控制索引, 目标索引, 方向)
            # 方向: 'negative' = 负相关, 'positive' = 正相关
            ('induced_fan', 'pressure', 'negative'),  # 引风机-负压
            ('secondary_fan', 'oxygen', 'positive'),  # 二次风机-含氧
        ]

    def forward(
        self,
        prediction: torch.Tensor,
        control_input: Optional[torch.Tensor] = None,
        prev_prediction: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        计算物理约束损失

        Args:
            prediction: 预测值 (batch, H, n_y) 或 (batch, H*n_y)
            control_input: 控制输入变化量 (batch, H, n_u)，用于单调性约束
            prev_prediction: 上一步预测值，用于平滑性约束

        Returns:
            losses: 各项损失字典 {'total': ..., 'monotonicity': ..., ...}
        """
        # 处理输入形状
        if prediction.dim() == 2:
            # 扁平输入，假设 n_y = 7
            batch_size = prediction.size(0)
            n_y = 7  # 目标变量维度
            H = prediction.size(1) // n_y
            prediction = prediction.view(batch_size, H, n_y)

        losses = {}
        total_loss = 0.0

        # 1. 边界约束
        boundary_loss = self.compute_boundary_loss(prediction)
        losses['boundary'] = boundary_loss
        total_loss += self.boundary_weight * boundary_loss

        # 2. 平滑性约束
        if prev_prediction is not None:
            smoothness_loss = self.compute_smoothness_loss(prediction, prev_prediction)
            losses['smoothness'] = smoothness_loss
            total_loss += self.smoothness_weight * smoothness_loss
        else:
            # 内部平滑性（预测序列内部）
            smoothness_loss = self.compute_internal_smoothness_loss(prediction)
            losses['smoothness'] = smoothness_loss
            total_loss += self.smoothness_weight * smoothness_loss

        # 3. 空间一致性约束
        spatial_loss = self.compute_spatial_consistency_loss(prediction)
        losses['spatial'] = spatial_loss
        total_loss += self.spatial_weight * spatial_loss

        # 4. 单调性约束（需要控制输入）
        if control_input is not None:
            monotonicity_loss = self.compute_monotonicity_loss(prediction, control_input)
            losses['monotonicity'] = monotonicity_loss
            total_loss += self.monotonicity_weight * monotonicity_loss

            # 5. 变化幅度约束（新增）
            amplitude_loss = self.compute_amplitude_loss(prediction, control_input)
            losses['amplitude'] = amplitude_loss
            total_loss += self.amplitude_weight * amplitude_loss

        losses['total'] = total_loss

        return losses

    def compute_boundary_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """计算边界约束损失

        预测值超出物理范围时施加惩罚

        Args:
            prediction: (batch, H, n_y) - 前4个是负压，后3个是含氧

        Returns:
            boundary_loss: 边界损失
        """
        # 负压约束（前4个变量）
        pressure_pred = prediction[:, :, :4]  # (batch, H, 4)
        pressure_violation_low = F.relu(self.pressure_low - pressure_pred)  # 低于下界
        pressure_violation_high = F.relu(pressure_pred - self.pressure_high)  # 高于上界
        pressure_loss = (pressure_violation_low + pressure_violation_high).mean()

        # 含氧约束（后3个变量）
        oxygen_pred = prediction[:, :, 4:7]  # (batch, H, 3)
        oxygen_violation_low = F.relu(self.oxygen_low - oxygen_pred)
        oxygen_violation_high = F.relu(oxygen_pred - self.oxygen_high)
        oxygen_loss = (oxygen_violation_low + oxygen_violation_high).mean()

        return pressure_loss + oxygen_loss

    def compute_smoothness_loss(
        self,
        prediction: torch.Tensor,
        prev_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """计算平滑性约束损失（与上一步预测的差异）

        Args:
            prediction: 当前预测 (batch, H, n_y)
            prev_prediction: 上一步预测 (batch, n_y) 或 (batch, H, n_y)

        Returns:
            smoothness_loss: 平滑性损失
        """
        if prev_prediction.dim() == 2 and prev_prediction.size(1) == prediction.size(2):
            # 上一步是单步预测 (batch, n_y)
            # 比较当前第一步与上一步
            diff = prediction[:, 0, :] - prev_prediction
        else:
            # 上一步也是多步预测，比较第一步
            diff = prediction[:, 0, :] - prev_prediction[:, 0, :]

        # 二阶差分惩罚（变化率的惩罚）
        smoothness_loss = (diff ** 2).mean()

        return smoothness_loss

    def compute_internal_smoothness_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """计算内部平滑性损失（预测序列内部的平滑性）

        Args:
            prediction: (batch, H, n_y)

        Returns:
            smoothness_loss: 内部平滑性损失
        """
        # 一阶差分
        diff1 = prediction[:, 1:, :] - prediction[:, :-1, :]  # (batch, H-1, n_y)

        # 二阶差分
        if prediction.size(1) > 2:
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]  # (batch, H-2, n_y)
            smoothness_loss = (diff1 ** 2).mean() + (diff2 ** 2).mean() * 0.5
        else:
            smoothness_loss = (diff1 ** 2).mean()

        return smoothness_loss

    def compute_spatial_consistency_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """计算空间一致性损失

        4个负压测点之间差异不应过大，3个含氧测点之间差异不应过大

        Args:
            prediction: (batch, H, n_y)

        Returns:
            spatial_loss: 空间一致性损失
        """
        # 负压测点标准差
        pressure_pred = prediction[:, :, :4]  # (batch, H, 4)
        pressure_std = torch.std(pressure_pred, dim=2)  # (batch, H)

        # 含氧测点标准差
        oxygen_pred = prediction[:, :, 4:7]  # (batch, H, 3)
        oxygen_std = torch.std(oxygen_pred, dim=2)  # (batch, H)

        # 惩罚过大的差异
        threshold = 20.0  # 负压差异阈值（Pa）
        pressure_penalty = F.relu(pressure_std - threshold)
        oxygen_penalty = oxygen_std  # 含氧直接惩罚标准差

        spatial_loss = pressure_penalty.mean() + oxygen_penalty.mean() * 0.1

        return spatial_loss

    def compute_monotonicity_loss(
        self,
        prediction: torch.Tensor,
        control_input: torch.Tensor,
    ) -> torch.Tensor:
        """计算单调性约束损失

        基于物理因果关系的单调性约束：
        - 引风机频率↑ → 负压↓（负相关）
        - 二次风机频率↑ → 含氧↑（正相关）

        Args:
            prediction: (batch, H, n_y) - 预测值
            control_input: (batch, H, n_u) - 控制输入变化量

        Returns:
            monotonicity_loss: 单调性损失
        """
        # 控制输入变化量
        # 假设控制输入顺序：一次风机A/B, 二次风机A/B, 引风机A/B, 给煤量
        # 引风机索引：4, 5
        # 二次风机索引：2, 3

        control_diff = control_input[:, 1:, :] - control_input[:, :-1, :]  # (batch, H-1, n_u)
        pred_diff = prediction[:, 1:, :] - prediction[:, :-1, :]  # (batch, H-1, n_y)

        monotonicity_loss = 0.0

        # 引风机 - 负压（负相关）
        # 引风机频率增加，负压应该减小（更负）
        induced_fan_change = control_diff[:, :, 4:6].mean(dim=2)  # (batch, H-1)
        pressure_change = pred_diff[:, :, 0]  # 负压变化（取第一个测点），已经是(batch, H-1)

        # 负相关：induced_fan ↑ → pressure ↓
        # 如果induced_fan增加（正值）且pressure增加（正值），违反约束
        violation = induced_fan_change * pressure_change
        # 只有符号相同时才惩罚（正相关的情况违反负相关约束）
        negative_violation = F.relu(violation)  # 同时增加违反负相关
        monotonicity_loss += negative_violation.mean()

        # 二次风机 - 含氧（正相关）
        secondary_fan_change = control_diff[:, :, 2:4].mean(dim=2)  # (batch, H-1)
        oxygen_change = pred_diff[:, :, 4]  # 含氧变化（取第一个测点），已经是(batch, H-1)

        # 正相关：secondary_fan ↑ → oxygen ↑
        # 如果secondary_fan增加（正值）且oxygen减小（负值），违反约束
        positive_violation = F.relu(-secondary_fan_change * oxygen_change)  # 符号相反违反正相关
        monotonicity_loss += positive_violation.mean()

        return monotonicity_loss

    def compute_amplitude_loss(
        self,
        prediction: torch.Tensor,
        control_input: torch.Tensor,
    ) -> torch.Tensor:
        """计算变化幅度约束损失（新增）

        核心思想：预测的变化幅度应该与控制变化幅度有物理上的对应关系
        - 引风机变化Δf → 负压变化约 Δf * 3 Pa
        - 二次风机变化Δf → 含氧变化约 Δf * 0.05%

        如果预测变化幅度小于应有的物理幅度，说明模型响应不够灵敏（滞后）

        Args:
            prediction: (batch, H, n_y) - 预测值（标准化后的）
            control_input: (batch, H, n_u) - 控制输入（标准化后的）

        Returns:
            amplitude_loss: 变化幅度损失
        """
        # 控制变化量（标准化后的）
        control_diff = control_input[:, 1:, :] - control_input[:, :-1, :]  # (batch, H-1, n_u)

        # 预测变化量（标准化后的）
        pred_diff = prediction[:, 1:, :] - prediction[:, :-1, :]  # (batch, H-1, n_y)

        amplitude_loss = 0.0

        # 引风机变化 → 负压变化幅度约束
        # 引风机索引：4, 5 (假设标准化后)
        induced_fan_change = control_diff[:, :, 4:6].abs().mean(dim=2)  # (batch, H-1)
        pressure_change = pred_diff[:, :, :4].abs().mean(dim=2)  # 4个负压平均变化

        # 物理关系：引风机变化1单位（标准化） → 负压应变化约 pressure_amp_ratio 单位
        # 如果控制变化大但预测变化小，说明模型响应不足
        expected_pressure_change = induced_fan_change * self.pressure_amp_ratio

        # 惩罚预测变化小于应有变化的情况（响应不足）
        under_response = F.relu(expected_pressure_change - pressure_change)
        amplitude_loss += under_response.mean()

        # 二次风机变化 → 含氧变化幅度约束
        # 二次风机索引：2, 3
        secondary_fan_change = control_diff[:, :, 2:4].abs().mean(dim=2)  # (batch, H-1)
        oxygen_change = pred_diff[:, :, 4:7].abs().mean(dim=2)  # 3个含氧平均变化

        expected_oxygen_change = secondary_fan_change * self.oxygen_amp_ratio
        under_response_o2 = F.relu(expected_oxygen_change - oxygen_change)
        amplitude_loss += under_response_o2.mean() * 0.5  # 含氧权重较低

        return amplitude_loss


class CombinedLoss(nn.Module):
    """组合损失函数

    L = L_data + λ * L_physics

    数据损失 + 物理约束损失
    """

    def __init__(
        self,
        physics_weight: float = 0.1,
        monotonicity_weight: float = 0.05,
        boundary_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        spatial_weight: float = 0.02,
        amplitude_weight: float = 0.15,  # 新增
    ):
        super().__init__()

        self.physics_weight = physics_weight

        self.physics_loss = PhysicsConstraintLoss(
            monotonicity_weight=monotonicity_weight,
            boundary_weight=boundary_weight,
            smoothness_weight=smoothness_weight,
            spatial_weight=spatial_weight,
            amplitude_weight=amplitude_weight,  # 新增
        )

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        control_input: Optional[torch.Tensor] = None,
        prev_prediction: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        计算组合损失

        Args:
            prediction: 预测值
            target: 真实值
            control_input: 控制输入（可选）
            prev_prediction: 上一步预测（可选）

        Returns:
            losses: 损失字典 {'total': ..., 'data': ..., 'physics': ..., ...}
        """
        # 数据损失（MSE）
        data_loss = F.mse_loss(prediction, target)

        # 物理约束损失
        physics_losses = self.physics_loss(prediction, control_input, prev_prediction)

        # 组合损失
        total_loss = data_loss + self.physics_weight * physics_losses['total']

        losses = {
            'total': total_loss,
            'data': data_loss.item(),
            'physics': physics_losses['total'],
        }
        losses.update(physics_losses)

        return losses


def create_physics_loss(config: Optional[dict] = None) -> PhysicsConstraintLoss:
    """创建物理约束损失函数

    Args:
        config: 配置字典

    Returns:
        PhysicsConstraintLoss
    """
    if config is None:
        config = PHYSICS_LOSS_CONFIG

    return PhysicsConstraintLoss(
        monotonicity_weight=config.get('monotonicity_weight', 0.05),
        boundary_weight=config.get('boundary_weight', 0.1),
        smoothness_weight=config.get('smoothness_weight', 0.01),
        spatial_weight=config.get('spatial_weight', 0.02),
    )


__all__ = [
    "PhysicsConstraintLoss",
    "CombinedLoss",
    "create_physics_loss",
]