"""
损失函数模块

核心方案：加权MSE + 差分损失

针对非平稳问题的设计：
1. 加权MSE：近期预测权重更高（step_weights）
2. 差分损失：强迫模型预测变化趋势，解决预测滞后问题

使用方式：
    from src.predictor.loss import PredictionLoss

    loss_fn = PredictionLoss(step_weights=[1.0, 0.9, 0.8, 0.7, 0.6], diff_weight=0.1)
    loss = loss_fn(prediction, target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from .config import Config
from .utils import get_logger

logger = get_logger(__name__)


class PredictionLoss(nn.Module):
    """预测损失函数

    组合：
    1. 加权MSE：多步预测权重递减（近期更重要）
    2. 差分损失：强迫预测变化趋势（解决非平稳滞后问题）

    公式：
        Loss = Σ w_h * MSE(y_pred[h], y_true[h]) + λ * MSE(Δy_pred, Δy_true)

    其中：
        w_h: 第h步的权重（递减）
        Δy: 变化量 y[t+1] - y[t]
        λ: 差分损失权重

    设计原理：
        - 非平稳跳变本质是"变化"，差分损失强迫模型学习变化趋势
        - 解决模型收敛于均值、跳变预测滞后的问题
    """

    def __init__(
        self,
        step_weights: Optional[List[float]] = None,
        diff_weight: float = 0.1,
        horizon: int = 5,
    ):
        """
        Args:
            step_weights: 各步预测权重（长度=H）
            diff_weight: 差分损失权重（推荐0.1~0.2）
            horizon: 预测步数
        """
        super().__init__()

        # 默认权重：近期权重高
        if step_weights is None:
            step_weights = [1.0 - 0.1 * h for h in range(horizon)]  # [1.0, 0.9, 0.8, ...]

        self.step_weights = torch.tensor(step_weights, dtype=torch.float32)
        self.diff_weight = diff_weight
        self.horizon = horizon

        logger.info(f"PredictionLoss初始化:")
        logger.info(f"  step_weights: {step_weights}")
        logger.info(f"  diff_weight: {diff_weight}")

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch, H, n_y)
            target: 真实值 (batch, H, n_y)

        Returns:
            loss: 总损失
        """
        batch_size, H, n_y = pred.shape

        # 1. 加权MSE
        mse = F.mse_loss(pred, target, reduction='none')  # (batch, H, n_y)

        # 扩展step_weights以广播，并移动到相同设备
        weights = self.step_weights.to(pred.device).view(1, -1, 1).expand(batch_size, H, n_y)
        weighted_mse = (mse * weights).mean()

        # 2. 差分损失（预测变化趋势）
        if H > 1 and self.diff_weight > 0:
            # 计算变化量：Δy = y[t+1] - y[t]
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]  # (batch, H-1, n_y)
            target_diff = target[:, 1:, :] - target[:, :-1, :]

            diff_loss = F.mse_loss(pred_diff, target_diff)
        else:
            diff_loss = 0.0

        # 总损失
        total_loss = weighted_mse + self.diff_weight * diff_loss

        return total_loss

    def compute_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> dict:
        """计算详细指标（用于日志）

        Args:
            pred: 预测值 (batch, H, n_y)
            target: 真实值 (batch, H, n_y)

        Returns:
            metrics: {'mse', 'weighted_mse', 'diff_loss', 'diff_accuracy'}
        """
        batch_size, H, n_y = pred.shape

        # MSE（无加权）
        mse = F.mse_loss(pred, target)

        # 加权MSE
        mse_per_step = F.mse_loss(pred, target, reduction='none').mean(dim=(0, 2))  # (H,)
        weights = self.step_weights.to(pred.device)
        weighted_mse = (mse_per_step * weights).mean()

        # 差分损失
        if H > 1:
            pred_diff = pred[:, 1:, :] - pred[:, :-1, :]
            target_diff = target[:, 1:, :] - target[:, :-1, :]
            diff_loss = F.mse_loss(pred_diff, target_diff)

            # 差分方向准确率（变化方向是否一致）
            pred_sign = torch.sign(pred_diff)
            target_sign = torch.sign(target_diff)
            diff_accuracy = (pred_sign == target_sign).float().mean()
        else:
            diff_loss = 0.0
            diff_accuracy = 1.0

        return {
            'mse': mse.item(),
            'weighted_mse': weighted_mse.item(),
            'diff_loss': diff_loss.item() if isinstance(diff_loss, torch.Tensor) else diff_loss,
            'diff_accuracy': diff_accuracy.item() if isinstance(diff_accuracy, torch.Tensor) else diff_accuracy,
            'mse_per_step': mse_per_step.tolist(),
        }


class SMAPELoss(nn.Module):
    """对称平均绝对百分比误差损失

    公式：
        SMAPE = mean(|y_pred - y_true| / (|y_pred| + |y_true|))

    特点：
    - 均衡大小数值的贡献
    - 百分比误差，不受尺度影响
    - 范围：[0, 1]

    适用场景：
    - 数据尺度差异大时（负压~100Pa，含氧~3%）
    - 可与MSE组合使用
    """

    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: 预测值 (batch, H, n_y)
            target: 真实值 (batch, H, n_y)

        Returns:
            loss: SMAPE损失
        """
        diff = torch.abs(pred - target)
        sum_abs = torch.abs(pred) + torch.abs(target) + self.epsilon

        smape = diff / sum_abs
        return smape.mean()


def create_loss_fn(config: Config) -> PredictionLoss:
    """创建损失函数（工厂函数）

    Args:
        config: 配置对象

    Returns:
        loss_fn: PredictionLoss实例
    """
    return PredictionLoss(
        step_weights=config.train.step_weights,
        diff_weight=config.loss.diff_weight,
        horizon=config.H,
    )


__all__ = [
    "PredictionLoss",
    "SMAPELoss",
    "create_loss_fn",
]