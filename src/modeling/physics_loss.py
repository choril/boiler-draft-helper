"""
物理约束损失函数 - PINN风格约束（V2修复版）

核心修复：
1. 所有物理约束在原始物理空间中计算（反标准化后再施加）
2. 标准化参数作为buffer注册，随模型保存/加载
3. 边界约束、幅度约束、空间一致性全部使用原始物理单位
4. 平滑性约束在标准化空间计算（纯正则化，无需物理单位）

约束类型：
1. 单调性约束：控制变量与目标变量的因果关系（原始空间）
2. 边界约束：预测值应在物理范围内（原始空间）
3. 平滑性约束：预测值时序应平滑（标准化空间）
4. 空间一致性约束：多测点之间的一致性（原始空间）
5. 变化幅度约束：预测响应应与控制变化物理匹配（原始空间）
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
    """物理约束损失函数（V2修复版）

    核心改进：
    - 预测值先反标准化到原始物理空间，再施加物理约束
    - 标准化参数(y_mean, y_scale, u_mean, u_scale)作为buffer注册
    - 边界约束、幅度约束、空间一致性在原始空间计算
    - 平滑性约束在标准化空间计算（纯正则化）

    L_physics = λ_mono * L_mono + λ_boundary * L_boundary
              + λ_smooth * L_smooth + λ_spatial * L_spatial + λ_amp * L_amp
    """

    def __init__(
        self,
        y_mean: np.ndarray = None,
        y_scale: np.ndarray = None,
        u_mean: np.ndarray = None,
        u_scale: np.ndarray = None,
        monotonicity_weight: float = 0.05,
        boundary_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        spatial_weight: float = 0.02,
        amplitude_weight: float = 0.15,
        pressure_boundary: Tuple[float, float] = PRESSURE_SAFETY_RANGE,
        oxygen_boundary: Tuple[float, float] = OXYGEN_SAFETY_RANGE,
        pressure_amp_ratio: float = 3.0,
        oxygen_amp_ratio: float = 0.05,
        spatial_pressure_threshold: float = 20.0,
        spatial_oxygen_threshold: float = 0.5,
    ):
        super().__init__()

        self.monotonicity_weight = monotonicity_weight
        self.boundary_weight = boundary_weight
        self.smoothness_weight = smoothness_weight
        self.spatial_weight = spatial_weight
        self.amplitude_weight = amplitude_weight

        self.pressure_low, self.pressure_high = pressure_boundary
        self.oxygen_low, self.oxygen_high = oxygen_boundary

        self.pressure_amp_ratio = pressure_amp_ratio
        self.oxygen_amp_ratio = oxygen_amp_ratio
        self.spatial_pressure_threshold = spatial_pressure_threshold
        self.spatial_oxygen_threshold = spatial_oxygen_threshold

        n_y = 7
        n_u = 7

        if y_mean is not None and y_scale is not None:
            self.register_buffer('y_mean', torch.tensor(y_mean, dtype=torch.float32))
            self.register_buffer('y_scale', torch.tensor(y_scale, dtype=torch.float32))
        else:
            self.register_buffer('y_mean', torch.zeros(n_y, dtype=torch.float32))
            self.register_buffer('y_scale', torch.ones(n_y, dtype=torch.float32))
            logger.warning("PhysicsConstraintLoss: 未提供Y标准化参数，使用默认值(0,1)，物理约束可能不准确")

        if u_mean is not None and u_scale is not None:
            self.register_buffer('u_mean', torch.tensor(u_mean, dtype=torch.float32))
            self.register_buffer('u_scale', torch.tensor(u_scale, dtype=torch.float32))
        else:
            self.register_buffer('u_mean', torch.zeros(n_u, dtype=torch.float32))
            self.register_buffer('u_scale', torch.ones(n_u, dtype=torch.float32))
            logger.warning("PhysicsConstraintLoss: 未提供U标准化参数，使用默认值(0,1)，幅度约束可能不准确")

        self._log_init_info()

    def _log_init_info(self):
        logger.info("PhysicsConstraintLoss V2 初始化:")
        logger.info(f"  Y标准化参数: mean={self.y_mean.tolist()}, scale={self.y_scale.tolist()}")
        logger.info(f"  U标准化参数: mean={self.u_mean.tolist()}, scale={self.u_scale.tolist()}")
        logger.info(f"  负压边界: [{self.pressure_low}, {self.pressure_high}] Pa")
        logger.info(f"  含氧边界: [{self.oxygen_low}, {self.oxygen_high}] %")
        logger.info(f"  引风机→负压增益: {self.pressure_amp_ratio} Pa/Hz")
        logger.info(f"  二次风机→含氧增益: {self.oxygen_amp_ratio} %/Hz")

    def inverse_transform_y(self, pred_standardized: torch.Tensor) -> torch.Tensor:
        """将标准化预测值还原为原始物理单位

        Args:
            pred_standardized: 标准化后的预测值 (..., n_y)

        Returns:
            pred_original: 原始物理单位的预测值 (..., n_y)
        """
        return pred_standardized * self.y_scale + self.y_mean

    def inverse_transform_u(self, ctrl_standardized: torch.Tensor) -> torch.Tensor:
        """将标准化控制值还原为原始物理单位

        Args:
            ctrl_standardized: 标准化后的控制值 (..., n_u)

        Returns:
            ctrl_original: 原始物理单位的控制值 (..., n_u)
        """
        return ctrl_standardized * self.u_scale + self.u_mean

    def set_scaler_params(
        self,
        y_mean: np.ndarray,
        y_scale: np.ndarray,
        u_mean: np.ndarray = None,
        u_scale: np.ndarray = None,
    ):
        """运行时更新标准化参数（用于训练脚本中从sampler获取参数后设置）

        Args:
            y_mean: Y均值 (n_y,)
            y_scale: Y标准差 (n_y,)
            u_mean: U均值 (n_u,)
            u_scale: U标准差 (n_u,)
        """
        self.y_mean.copy_(torch.tensor(y_mean, dtype=torch.float32))
        self.y_scale.copy_(torch.tensor(y_scale, dtype=torch.float32))
        if u_mean is not None:
            self.u_mean.copy_(torch.tensor(u_mean, dtype=torch.float32))
        if u_scale is not None:
            self.u_scale.copy_(torch.tensor(u_scale, dtype=torch.float32))
        logger.info("PhysicsConstraintLoss: 标准化参数已更新")

    def forward(
        self,
        prediction: torch.Tensor,
        control_input: Optional[torch.Tensor] = None,
        prev_prediction: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        计算物理约束损失

        Args:
            prediction: 预测值（标准化后） (batch, H, n_y) 或 (batch, H*n_y)
            control_input: 控制输入（标准化后） (batch, H, n_u)
            prev_prediction: 上一步预测值（标准化后）

        Returns:
            losses: 各项损失字典
        """
        if prediction.dim() == 2:
            batch_size = prediction.size(0)
            n_y = self.y_mean.shape[0]
            H = prediction.size(1) // n_y
            prediction = prediction.view(batch_size, H, n_y)

        pred_original = self.inverse_transform_y(prediction)

        ctrl_original = None
        if control_input is not None:
            ctrl_original = self.inverse_transform_u(control_input)

        losses = {}
        total_loss = 0.0

        boundary_loss = self.compute_boundary_loss(pred_original)
        losses['boundary'] = boundary_loss
        total_loss += self.boundary_weight * boundary_loss

        if prev_prediction is not None:
            smoothness_loss = self.compute_smoothness_loss(prediction, prev_prediction)
        else:
            smoothness_loss = self.compute_internal_smoothness_loss(prediction)
        losses['smoothness'] = smoothness_loss
        total_loss += self.smoothness_weight * smoothness_loss

        spatial_loss = self.compute_spatial_consistency_loss(pred_original)
        losses['spatial'] = spatial_loss
        total_loss += self.spatial_weight * spatial_loss

        if ctrl_original is not None:
            monotonicity_loss = self.compute_monotonicity_loss(pred_original, ctrl_original)
            losses['monotonicity'] = monotonicity_loss
            total_loss += self.monotonicity_weight * monotonicity_loss

            amplitude_loss = self.compute_amplitude_loss(pred_original, ctrl_original)
            losses['amplitude'] = amplitude_loss
            total_loss += self.amplitude_weight * amplitude_loss

        losses['total'] = total_loss
        return losses

    def compute_boundary_loss(self, pred_original: torch.Tensor) -> torch.Tensor:
        """计算边界约束损失（原始物理空间）

        预测值超出物理范围时施加惩罚

        Args:
            pred_original: 原始物理单位的预测值 (batch, H, n_y)
                          前4个是负压(Pa)，后3个是含氧(%)

        Returns:
            boundary_loss: 边界损失
        """
        pressure_pred = pred_original[:, :, :4]
        pressure_violation_low = F.relu(self.pressure_low - pressure_pred)
        pressure_violation_high = F.relu(pressure_pred - self.pressure_high)
        pressure_loss = (pressure_violation_low + pressure_violation_high).mean()

        oxygen_pred = pred_original[:, :, 4:7]
        oxygen_violation_low = F.relu(self.oxygen_low - oxygen_pred)
        oxygen_violation_high = F.relu(oxygen_pred - self.oxygen_high)
        oxygen_loss = (oxygen_violation_low + oxygen_violation_high).mean()

        return pressure_loss + oxygen_loss

    def compute_smoothness_loss(
        self,
        prediction: torch.Tensor,
        prev_prediction: torch.Tensor,
    ) -> torch.Tensor:
        """计算平滑性约束损失（标准化空间，纯正则化）

        Args:
            prediction: 当前预测（标准化） (batch, H, n_y)
            prev_prediction: 上一步预测（标准化） (batch, n_y) 或 (batch, H, n_y)

        Returns:
            smoothness_loss: 平滑性损失
        """
        if prev_prediction.dim() == 2 and prev_prediction.size(1) == prediction.size(2):
            diff = prediction[:, 0, :] - prev_prediction
        else:
            diff = prediction[:, 0, :] - prev_prediction[:, 0, :]

        smoothness_loss = (diff ** 2).mean()
        return smoothness_loss

    def compute_internal_smoothness_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """计算内部平滑性损失（标准化空间，纯正则化）

        Args:
            prediction: 标准化预测值 (batch, H, n_y)

        Returns:
            smoothness_loss: 内部平滑性损失
        """
        diff1 = prediction[:, 1:, :] - prediction[:, :-1, :]

        if prediction.size(1) > 2:
            diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
            smoothness_loss = (diff1 ** 2).mean() + (diff2 ** 2).mean() * 0.5
        else:
            smoothness_loss = (diff1 ** 2).mean()

        return smoothness_loss

    def compute_spatial_consistency_loss(self, pred_original: torch.Tensor) -> torch.Tensor:
        """计算空间一致性损失（原始物理空间）

        4个负压测点之间差异不应过大，3个含氧测点之间差异不应过大
        阈值使用物理单位（Pa, %）

        Args:
            pred_original: 原始物理单位的预测值 (batch, H, n_y)

        Returns:
            spatial_loss: 空间一致性损失
        """
        pressure_pred = pred_original[:, :, :4]
        pressure_std = torch.std(pressure_pred, dim=2)

        oxygen_pred = pred_original[:, :, 4:7]
        oxygen_std = torch.std(oxygen_pred, dim=2)

        pressure_penalty = F.relu(pressure_std - self.spatial_pressure_threshold)
        oxygen_penalty = F.relu(oxygen_std - self.spatial_oxygen_threshold)

        spatial_loss = pressure_penalty.mean() + oxygen_penalty.mean()
        return spatial_loss

    def compute_monotonicity_loss(
        self,
        pred_original: torch.Tensor,
        ctrl_original: torch.Tensor,
    ) -> torch.Tensor:
        """计算单调性约束损失（原始物理空间）

        基于物理因果关系的单调性约束：
        - 引风机频率↑ → 负压↓（更负，即数值减小）
        - 二次风机频率↑ → 含氧↑

        在原始空间中计算，方向更可靠

        Args:
            pred_original: 原始物理单位的预测值 (batch, H, n_y)
            ctrl_original: 原始物理单位的控制值 (batch, H, n_u)

        Returns:
            monotonicity_loss: 单调性损失
        """
        control_diff = ctrl_original[:, 1:, :] - ctrl_original[:, :-1, :]
        pred_diff = pred_original[:, 1:, :] - pred_original[:, :-1, :]

        monotonicity_loss = pred_original.sum() * 0.0

        induced_fan_change = control_diff[:, :, 4:6].mean(dim=2)
        pressure_change = pred_diff[:, :, 0]

        violation_neg = F.relu(induced_fan_change * pressure_change)
        monotonicity_loss = monotonicity_loss + violation_neg.mean()

        secondary_fan_change = control_diff[:, :, 2:4].mean(dim=2)
        oxygen_change = pred_diff[:, :, 4]

        violation_pos = F.relu(-secondary_fan_change * oxygen_change)
        monotonicity_loss = monotonicity_loss + violation_pos.mean()

        return monotonicity_loss

    def compute_amplitude_loss(
        self,
        pred_original: torch.Tensor,
        ctrl_original: torch.Tensor,
    ) -> torch.Tensor:
        """计算变化幅度约束损失（原始物理空间）

        核心思想：预测的变化幅度应该与控制变化幅度有物理上的对应关系
        - 引风机变化Δf Hz → 负压变化约 Δf * 3 Pa
        - 二次风机变化Δf Hz → 含氧变化约 Δf * 0.05%

        在原始空间中计算，物理比例关系才有意义

        Args:
            pred_original: 原始物理单位的预测值 (batch, H, n_y)
            ctrl_original: 原始物理单位的控制值 (batch, H, n_u)

        Returns:
            amplitude_loss: 变化幅度损失
        """
        control_diff = ctrl_original[:, 1:, :] - ctrl_original[:, :-1, :]
        pred_diff = pred_original[:, 1:, :] - pred_original[:, :-1, :]

        amplitude_loss = pred_original.sum() * 0.0

        induced_fan_change = control_diff[:, :, 4:6].abs().mean(dim=2)
        pressure_change = pred_diff[:, :, :4].abs().mean(dim=2)

        expected_pressure_change = induced_fan_change * self.pressure_amp_ratio
        under_response = F.relu(expected_pressure_change - pressure_change)
        amplitude_loss = amplitude_loss + under_response.mean()

        secondary_fan_change = control_diff[:, :, 2:4].abs().mean(dim=2)
        oxygen_change = pred_diff[:, :, 4:7].abs().mean(dim=2)

        expected_oxygen_change = secondary_fan_change * self.oxygen_amp_ratio
        under_response_o2 = F.relu(expected_oxygen_change - oxygen_change)
        amplitude_loss = amplitude_loss + under_response_o2.mean() * 0.5

        return amplitude_loss


class CombinedLoss(nn.Module):
    """组合损失函数（V2修复版）

    L = L_data + λ * L_physics

    数据损失 + 物理约束损失（物理约束在原始空间计算）
    """

    def __init__(
        self,
        y_mean: np.ndarray = None,
        y_scale: np.ndarray = None,
        u_mean: np.ndarray = None,
        u_scale: np.ndarray = None,
        physics_weight: float = 0.1,
        monotonicity_weight: float = 0.05,
        boundary_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        spatial_weight: float = 0.02,
        amplitude_weight: float = 0.15,
        pressure_amp_ratio: float = 3.0,
        oxygen_amp_ratio: float = 0.05,
    ):
        super().__init__()

        self.physics_weight = physics_weight

        self.physics_loss = PhysicsConstraintLoss(
            y_mean=y_mean,
            y_scale=y_scale,
            u_mean=u_mean,
            u_scale=u_scale,
            monotonicity_weight=monotonicity_weight,
            boundary_weight=boundary_weight,
            smoothness_weight=smoothness_weight,
            spatial_weight=spatial_weight,
            amplitude_weight=amplitude_weight,
            pressure_amp_ratio=pressure_amp_ratio,
            oxygen_amp_ratio=oxygen_amp_ratio,
        )

    def set_scaler_params(self, y_mean, y_scale, u_mean=None, u_scale=None):
        """运行时更新标准化参数"""
        self.physics_loss.set_scaler_params(y_mean, y_scale, u_mean, u_scale)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        control_input: Optional[torch.Tensor] = None,
        prev_prediction: Optional[torch.Tensor] = None,
    ) -> dict:
        data_loss = F.mse_loss(prediction, target)

        physics_losses = self.physics_loss(prediction, control_input, prev_prediction)

        total_loss = data_loss + self.physics_weight * physics_losses['total']

        losses = {
            'total': total_loss,
            'data': data_loss.item(),
            'physics': physics_losses['total'].item() if isinstance(physics_losses['total'], torch.Tensor) else physics_losses['total'],
            'physics_detail': physics_losses['total'].item() if isinstance(physics_losses['total'], torch.Tensor) else physics_losses['total'],
        }
        for k, v in physics_losses.items():
            if k not in ('total',):
                losses[k] = v.item() if isinstance(v, torch.Tensor) else v

        return losses


def create_physics_loss(
    config: Optional[dict] = None,
    y_mean: np.ndarray = None,
    y_scale: np.ndarray = None,
    u_mean: np.ndarray = None,
    u_scale: np.ndarray = None,
) -> PhysicsConstraintLoss:
    """创建物理约束损失函数

    Args:
        config: 配置字典
        y_mean: Y标准化均值
        y_scale: Y标准化标准差
        u_mean: U标准化均值
        u_scale: U标准化标准差

    Returns:
        PhysicsConstraintLoss
    """
    if config is None:
        config = PHYSICS_LOSS_CONFIG

    return PhysicsConstraintLoss(
        y_mean=y_mean,
        y_scale=y_scale,
        u_mean=u_mean,
        u_scale=u_scale,
        monotonicity_weight=config.get('monotonicity_weight', 0.05),
        boundary_weight=config.get('boundary_weight', 0.1),
        smoothness_weight=config.get('smoothness_weight', 0.01),
        spatial_weight=config.get('spatial_weight', 0.02),
        amplitude_weight=config.get('amplitude_weight', 0.15),
    )


__all__ = [
    "PhysicsConstraintLoss",
    "CombinedLoss",
    "create_physics_loss",
]
