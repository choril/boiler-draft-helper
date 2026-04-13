"""
轻量PINN代理模型 - 用于MPC在线优化

特点：
1. 小型MLP结构（快速推理 < 10ms）
2. 物理约束损失（PINN风格）
3. 可从教师模型蒸馏训练
4. 支持批量评估（用于贝叶斯优化）

输入：
- 扁平化的历史窗口 + 未来控制序列

输出：
- 扁平化的未来预测（H步 × n_y）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional, Tuple, Callable
import numpy as np
import time

from src.config.hyperparams import PROXY_MLP_CONFIG, H
from src.modeling.physics_loss import PhysicsConstraintLoss, CombinedLoss
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PINNProxyMLP(nn.Module):
    """PINN约束的MLP代理模型

    结构：
    Input: [历史Y(L×n_y) + 历史U(L×n_u) + 历史X(L×n_x) + 未来U(H×n_u)]
           扁平化输入

    Hidden: [256, 128, 64] (可配置)

    Output: [未来Y(H×n_y)] 扁平化输出
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: list[int] = [256, 128, 64],
        dropout_rate: float = 0.1,
        activation: str = 'relu',
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        # 激活函数
        if activation == 'relu':
            act_fn = nn.ReLU
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU
        elif activation == 'gelu':
            act_fn = nn.GELU
        else:
            act_fn = nn.ReLU

        # 构建网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # 推理时间统计
        self._inference_times = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (batch, input_dim) 扁平化输入

        Returns:
            y: (batch, output_dim) 扁平化输出
        """
        return self.network(x)

    def predict_with_timing(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """带时间统计的预测

        Args:
            x: 输入张量

        Returns:
            y: 预测输出
            time_ms: 推理时间（毫秒）
        """
        start = time.perf_counter()
        with torch.no_grad():
            y = self.forward(x)
        end = time.perf_counter()
        time_ms = (end - start) * 1000
        self._inference_times.append(time_ms)
        return y, time_ms

    def get_avg_inference_time(self) -> float:
        """获取平均推理时间（毫秒）"""
        if len(self._inference_times) == 0:
            return 0.0
        return np.mean(self._inference_times)

    def reshape_output(
        self,
        output_flat: torch.Tensor,
        H: int,
        n_y: int,
    ) -> torch.Tensor:
        """将扁平输出重塑为序列

        Args:
            output_flat: (batch, H*n_y)
            H: 预测步数
            n_y: 目标变量维度

        Returns:
            output_seq: (batch, H, n_y)
        """
        batch_size = output_flat.size(0)
        return output_flat.view(batch_size, H, n_y)


class ProxyTrainer:
    """代理模型训练器

    支持：
    1. 从教师模型蒸馏训练
    2. 物理约束损失
    3. 早停和学习率衰减
    """

    def __init__(
        self,
        model: PINNProxyMLP,
        device: torch.device,
        learning_rate: float = 0.001,
        physics_weight: float = 0.1,
        H: int = 5,
        n_y: int = 7,
    ):
        self.model = model.to(device)
        self.device = device
        self.H = H
        self.n_y = n_y

        # 损失函数
        self.combined_loss = CombinedLoss(physics_weight=physics_weight)

        # 优化器
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        # 最佳模型
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def train_epoch(
        self,
        train_loader,
        control_input: Optional[torch.Tensor] = None,
    ) -> dict:
        """训练一个epoch

        Args:
            train_loader: 训练数据加载器
            control_input: 控制输入张量（用于物理约束）

        Returns:
            losses: 损失字典
        """
        self.model.train()
        total_losses = {'total': 0, 'data': 0, 'physics': 0}
        n_batches = 0

        for X, Y_teacher in train_loader:
            X = X.to(self.device)
            Y_teacher = Y_teacher.to(self.device)

            # 前向传播
            prediction_flat = self.model(X)
            prediction_seq = self.model.reshape_output(prediction_flat, self.H, self.n_y)
            target_seq = self.model.reshape_output(Y_teacher, self.H, self.n_y)

            # 计算损失
            losses = self.combined_loss(prediction_seq, target_seq)

            # 反向传播
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # 累计损失
            total_losses['total'] += losses['total'].item()
            total_losses['data'] += losses['data']
            total_losses['physics'] += losses['physics']
            n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    def validate(self, val_loader) -> dict:
        """验证

        Args:
            val_loader: 验证数据加载器

        Returns:
            losses: 验证损失字典
        """
        self.model.eval()
        total_losses = {'total': 0, 'data': 0, 'physics': 0}
        n_batches = 0

        with torch.no_grad():
            for X, Y_teacher in val_loader:
                X = X.to(self.device)
                Y_teacher = Y_teacher.to(self.device)

                prediction_flat = self.model(X)
                prediction_seq = self.model.reshape_output(prediction_flat, self.H, self.n_y)
                target_seq = self.model.reshape_output(Y_teacher, self.H, self.n_y)

                losses = self.combined_loss(prediction_seq, target_seq)

                total_losses['total'] += losses['total'].item()
                total_losses['data'] += losses['data']
                total_losses['physics'] += losses['physics']
                n_batches += 1

        return {k: v / n_batches for k, v in total_losses.items()}

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 30,
        patience: int = 5,
        verbose: bool = True,
    ) -> dict:
        """完整训练流程

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 最大epoch数
            patience: 早停耐心值
            verbose: 是否打印日志

        Returns:
            history: 训练历史
        """
        history = {'train_loss': [], 'val_loss': [], 'train_physics': [], 'val_physics': []}
        epochs_no_improve = 0

        logger.info("开始代理模型训练")
        logger.info(f"最大epochs: {epochs}, 早停patience: {patience}")

        for epoch in range(epochs):
            # 训练
            train_losses = self.train_epoch(train_loader)
            history['train_loss'].append(train_losses['total'])
            history['train_physics'].append(train_losses['physics'])

            # 验证
            val_losses = self.validate(val_loader)
            history['val_loss'].append(val_losses['total'])
            history['val_physics'].append(val_losses['physics'])

            # 学习率调度
            self.scheduler.step(val_losses['total'])

            # 打印日志
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train: {train_losses['total']:.4f} (data:{train_losses['data']:.4f}, phys:{train_losses['physics']:.4f}), "
                    f"Val: {val_losses['total']:.4f}"
                )

            # 保存最佳模型
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            # 早停
            if epochs_no_improve >= patience:
                logger.info(f"早停触发，最佳epoch: {epoch - patience + 1}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"加载最佳模型，验证损失: {self.best_val_loss:.4f}")

        return history

    def measure_inference_speed(
        self,
        n_samples: int = 1000,
        input_dim: Optional[int] = None,
    ) -> dict:
        """测量推理速度

        Args:
            n_samples: 测试样本数
            input_dim: 输入维度（可选）

        Returns:
            speed_metrics: 速度指标字典
        """
        if input_dim is None:
            input_dim = self.model.input_dim

        # 生成测试数据
        X_test = torch.randn(n_samples, input_dim).to(self.device)

        # 批量推理测试
        self.model.eval()
        times = []

        with torch.no_grad():
            for i in range(10):  # 重复10次取平均
                start = time.perf_counter()
                _ = self.model(X_test)
                end = time.perf_counter()
                times.append((end - start) * 1000)

        avg_time_ms = np.mean(times)
        per_sample_us = avg_time_ms * 1000 / n_samples

        logger.info(f"推理速度测试:")
        logger.info(f"  批量{n_samples}样本: {avg_time_ms:.2f} ms")
        logger.info(f"  单样本: {per_sample_us:.2f} μs")

        return {
            'batch_time_ms': avg_time_ms,
            'per_sample_us': per_sample_us,
            'n_samples': n_samples,
        }


def create_proxy_model(
    L: int = 10,
    H: int = 5,
    n_y: int = 7,
    n_u: int = 7,
    n_x: int = 50,
    config: Optional[dict] = None,
) -> PINNProxyMLP:
    """创建代理模型

    Args:
        L: 历史窗口长度
        H: 预测步数
        n_y: 目标变量维度
        n_u: 控制变量维度
        n_x: 状态变量维度
        config: 配置字典

    Returns:
        model: PINNProxyMLP模型
    """
    if config is None:
        config = PROXY_MLP_CONFIG

    # 输入维度：历史Y/U/X + 未来U
    input_dim = L * (n_y + n_u + n_x) + H * n_u

    # 输出维度：未来Y
    output_dim = H * n_y

    model = PINNProxyMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_layers=config.get('hidden_layers', [256, 128, 64]),
        dropout_rate=config.get('dropout_rate', 0.1),
        activation=config.get('activation', 'relu'),
    )

    logger.info(f"代理模型创建完成:")
    logger.info(f"  输入维度: {input_dim}")
    logger.info(f"  输出维度: {output_dim}")
    logger.info(f"  隐藏层: {config.get('hidden_layers', [256, 128, 64])}")

    return model


__all__ = [
    "PINNProxyMLP",
    "ProxyTrainer",
    "create_proxy_model",
]