"""
训练器模块

功能：
1. 训练循环（batch进度日志）
2. 验证评估
3. 早停机制
4. 学习率调度
5. 模型保存/加载

使用方式：
    from src.predictor.trainer import Trainer

    trainer = Trainer(model, config, loss_fn)
    history = trainer.fit(train_loader, val_loader)
    metrics = trainer.evaluate(test_loader)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple
from pathlib import Path
import json
import numpy as np

from .config import Config
from .loss import PredictionLoss, create_loss_fn
from .utils import get_logger, save_json

logger = get_logger(__name__)


class Trainer:
    """训练器

    功能：
    - 训练一个epoch（带batch进度日志）
    - 验证评估
    - 早停机制
    - 学习率调度
    - 模型保存

    特点：
    - 简洁清晰的训练流程
    - RevIN反标准化评估（可选）
    """

    def __init__(
        self,
        model: nn.Module,
        config: Config,
        loss_fn: Optional[PredictionLoss] = None,
    ):
        """
        Args:
            model: PyTorch模型
            config: 配置对象
            loss_fn: 损失函数（默认使用config创建）
        """
        self.model = model
        self.config = config

        # 设备
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # 损失函数
        if loss_fn is None:
            loss_fn = create_loss_fn(config)
        self.loss_fn = loss_fn

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )

        # 早停
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_no_improve = 0

        # 训练历史
        self.history = {'train_loss': [], 'val_loss': [], 'lr': []}

        logger.info(f"Trainer初始化:")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  学习率: {config.train.learning_rate}")
        logger.info(f"  早停patience: {config.train.early_stop_patience}")

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
        verbose: bool = True,
    ) -> float:
        """训练一个epoch

        Args:
            train_loader: 训练数据
            epoch: 当前epoch编号
            verbose: 是否打印batch进度

        Returns:
            avg_loss: 平均训练损失
        """
        self.model.train()

        total_loss = 0.0
        n_batches = len(train_loader)

        # batch进度日志间隔（约10次）
        log_interval = max(1, n_batches // 10)

        for batch_idx, batch in enumerate(train_loader):
            # 解包batch
            encoder_input = batch[0].to(self.device)
            decoder_input = batch[1].to(self.device)
            target = batch[2].to(self.device)

            # 前向传播
            prediction = self.model(encoder_input, decoder_input)

            # 计算损失
            loss = self.loss_fn(prediction, target)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            # 打印batch进度
            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss_so_far = total_loss / (batch_idx + 1)
                progress = (batch_idx + 1) / n_batches * 100
                logger.info(f"  Epoch {epoch} - Batch {batch_idx+1}/{n_batches} ({progress:.0f}%) - Loss: {loss.item():.4f} (avg: {avg_loss_so_far:.4f})")

        return total_loss / n_batches

    def validate(self, val_loader: DataLoader) -> Dict:
        """验证

        Args:
            val_loader: 验证数据

        Returns:
            metrics: {'loss', 'mse', 'diff_accuracy'}
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                encoder_input = batch[0].to(self.device)
                decoder_input = batch[1].to(self.device)
                target = batch[2].to(self.device)

                prediction = self.model(encoder_input, decoder_input)

                loss = self.loss_fn(prediction, target)
                total_loss += loss.item()

                all_preds.append(prediction.cpu())
                all_targets.append(target.cpu())

        avg_loss = total_loss / len(val_loader)

        # 合并所有预测和目标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算详细指标
        metrics = self.loss_fn.compute_metrics(all_preds, all_targets)
        metrics['loss'] = avg_loss

        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        patience: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict:
        """完整训练流程

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 最大epoch数（默认使用config）
            patience: 早停耐心值（默认使用config）
            verbose: 是否打印日志

        Returns:
            history: 训练历史
        """
        epochs = epochs or self.config.train.epochs
        patience = patience or self.config.train.early_stop_patience

        logger.info(f"开始训练:")
        logger.info(f"  最大epochs: {epochs}")
        logger.info(f"  早停patience: {patience}")
        logger.info(f"  总batch数: {len(train_loader)}")

        for epoch in range(epochs):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch + 1, verbose=verbose)
            self.history['train_loss'].append(train_loss)

            # 验证
            val_metrics = self.validate(val_loader)
            val_loss = val_metrics['loss']
            self.history['val_loss'].append(val_loss)

            # 学习率调度
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            self.scheduler.step(val_loss)

            # 打印epoch总结
            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs} 完成 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                self.epochs_no_improve = 0
                if verbose:
                    logger.info(f"  ✓ 新最佳模型，验证损失: {val_loss:.4f}")
            else:
                self.epochs_no_improve += 1

            # 早停
            if self.epochs_no_improve >= patience:
                logger.info(f"早停触发，最佳epoch: {epoch - patience + 1}")
                break

        # 加载最佳模型
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"加载最佳模型，验证损失: {self.best_val_loss:.4f}")

        return self.history

    def evaluate(
        self,
        test_loader: DataLoader,
        inverse_transform: bool = False,
    ) -> Dict:
        """评估测试集

        Args:
            test_loader: 测试数据
            inverse_transform: 是否反标准化（需要提供y_mean, y_std）

        Returns:
            metrics: 评估指标
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_y_mean = []
        all_y_std = []

        with torch.no_grad():
            for batch in test_loader:
                encoder_input = batch[0].to(self.device)
                decoder_input = batch[1].to(self.device)
                target = batch[2].to(self.device)
                y_mean = batch[3]  # 窗口均值
                y_std = batch[4]   # 窗口标准差

                prediction = self.model(encoder_input, decoder_input)

                all_preds.append(prediction.cpu())
                all_targets.append(target.cpu())
                all_y_mean.append(y_mean)
                all_y_std.append(y_std)

        # 合并
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # 计算指标
        metrics = self.loss_fn.compute_metrics(all_preds, all_targets)

        # 反标准化评估（可选）
        if inverse_transform:
            all_y_mean = torch.cat(all_y_mean, dim=0).numpy()
            all_y_std = torch.cat(all_y_std, dim=0).numpy()
            all_preds_orig = all_preds.numpy() * all_y_std[:, np.newaxis, :] + all_y_mean[:, np.newaxis, :]
            all_targets_orig = all_targets.numpy() * all_y_std[:, np.newaxis, :] + all_y_mean[:, np.newaxis, :]

            # 原始空间的RMSE/MAE
            metrics['rmse_original'] = np.sqrt(np.mean((all_preds_orig - all_targets_orig) ** 2))
            metrics['mae_original'] = np.mean(np.abs(all_preds_orig - all_targets_orig))

            # 每步RMSE
            rmse_per_step = []
            for h in range(self.config.H):
                rmse_h = np.sqrt(np.mean((all_preds_orig[:, h, :] - all_targets_orig[:, h, :]) ** 2))
                rmse_per_step.append(rmse_h)
            metrics['rmse_per_step_original'] = rmse_per_step

        logger.info(f"测试集评估:")
        logger.info(f"  MSE: {metrics['mse']:.4f}")
        logger.info(f"  差分准确率: {metrics['diff_accuracy']:.4f}")
        if inverse_transform:
            logger.info(f"  RMSE(原始): {metrics['rmse_original']:.4f}")
            logger.info(f"  MAE(原始): {metrics['mae_original']:.4f}")

        return metrics

    def save(self, output_dir: Path, model_name: str = "model") -> None:
        """保存模型和训练结果

        Args:
            output_dir: 输出目录
            model_name: 模型名称
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存模型权重
        torch.save(self.model.state_dict(), output_dir / f"{model_name}.pt")

        # 保存训练历史
        save_json(self.history, output_dir / "training_history.json")

        # 保存配置
        save_json(self.config.to_dict(), output_dir / "config.json")

        logger.info(f"模型已保存至: {output_dir}")

    def load(self, model_path: Path) -> None:
        """加载模型权重

        Args:
            model_path: 模型文件路径
        """
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        logger.info(f"模型已加载: {model_path}")


def create_trainer(model: nn.Module, config: Config, loss_fn: Optional[PredictionLoss] = None) -> Trainer:
    """创建训练器（工厂函数）

    Args:
        model: PyTorch模型
        config: 配置对象
        loss_fn: 损失函数（可选）

    Returns:
        trainer: Trainer实例
    """
    return Trainer(model, config, loss_fn)


__all__ = [
    "Trainer",
    "create_trainer",
]