"""
NARX-LSTM动态预测模型

架构特点：
1. Encoder-Decoder结构
2. 控制作为外生输入（未来控制序列作为decoder输入）
3. Teacher forcing训练策略
4. 多步预测能力

模型结构：
- Encoder: Bi-LSTM，读入历史Y/U/X
- Decoder: LSTM，读入未来控制U，自回归输出预测Y
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Optional
import numpy as np

from src.config.hyperparams import NARX_LSTM_CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NARXLSTM(nn.Module):
    """NARX-LSTM模型

    Nonlinear AutoRegressive with Exogenous inputs (NARX)
    使用LSTM实现，支持多步预测

    输入：
    - encoder_input: (batch, L, n_features) 历史窗口 Y/U/X
    - decoder_input: (batch, H, n_u) 未来控制序列 U

    输出：
    - prediction: (batch, H, n_y) 未来预测 Y
    """

    def __init__(
        self,
        n_y: int = 7,  # 目标变量维度
        n_u: int = 7,  # 控制变量维度
        n_x: int = 50,  # 状态变量维度
        encoder_hidden: int = 128,
        decoder_hidden: int = 128,
        encoder_layers: int = 1,
        decoder_layers: int = 1,
        dropout: float = 0.2,
        bidirectional: bool = True,
        output_steps: int = 5,  # 预测步数 H
    ):
        super().__init__()

        self.n_y = n_y
        self.n_u = n_u
        self.n_x = n_x
        self.n_features = n_y + n_u + n_x  # 输入特征总数
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.output_steps = output_steps
        self.bidirectional = bidirectional

        # 编码器方向数
        self.encoder_directions = 2 if bidirectional else 1

        # === Encoder ===
        encoder_input_size = self.n_features
        self.encoder = nn.LSTM(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if encoder_layers > 1 else 0,
        )

        # Encoder输出投影（用于Decoder初始化）
        encoder_output_size = encoder_hidden * self.encoder_directions
        self.encoder_to_decoder_h = nn.Linear(encoder_output_size, decoder_hidden)
        self.encoder_to_decoder_c = nn.Linear(encoder_output_size, decoder_hidden)

        # === Decoder ===
        # Decoder输入：未来控制U + 上一步预测Y（自回归）
        decoder_input_size = n_u + n_y
        self.decoder = nn.LSTM(
            input_size=decoder_input_size,
            hidden_size=decoder_hidden,
            num_layers=decoder_layers,
            batch_first=True,
            dropout=dropout if decoder_layers > 1 else 0,
        )

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(decoder_hidden, decoder_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden // 2, n_y),
        )

        # 初始Y投影（用于第一步预测）
        self.init_y_proj = nn.Linear(encoder_output_size, n_y)

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        teacher_forcing: bool = False,
        teacher_forcing_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            encoder_input: (batch, L, n_features) 历史窗口
            decoder_input: (batch, H, n_u) 未来控制序列
            teacher_forcing: 是否使用teacher forcing
            teacher_forcing_target: (batch, H, n_y) 真实目标值

        Returns:
            prediction: (batch, H, n_y) 预测结果
        """
        batch_size = encoder_input.size(0)

        # === Encoding ===
        encoder_output, (encoder_h, encoder_c) = self.encoder(encoder_input)

        # 取最后时刻的hidden state
        # encoder_h: (layers * directions, batch, hidden)
        if self.bidirectional:
            # 合并两个方向的hidden state
            encoder_h_final = torch.cat([
                encoder_h[-2],  # forward方向最后一层
                encoder_h[-1],  # backward方向最后一层
            ], dim=1)  # (batch, encoder_hidden * 2)
            encoder_c_final = torch.cat([
                encoder_c[-2],
                encoder_c[-1],
            ], dim=1)
        else:
            encoder_h_final = encoder_h[-1]  # (batch, encoder_hidden)
            encoder_c_final = encoder_c[-1]

        # 初始化Decoder hidden state（需要为所有层初始化）
        # decoder_h: (decoder_layers, batch, decoder_hidden)
        decoder_h_single = self.encoder_to_decoder_h(encoder_h_final)  # (batch, decoder_hidden)
        decoder_c_single = self.encoder_to_decoder_c(encoder_c_final)  # (batch, decoder_hidden)

        # 为所有decoder层复制相同的初始hidden state
        decoder_h = decoder_h_single.unsqueeze(0).repeat(self.decoder_layers, 1, 1)  # (decoder_layers, batch, decoder_hidden)
        decoder_c = decoder_c_single.unsqueeze(0).repeat(self.decoder_layers, 1, 1)  # (decoder_layers, batch, decoder_hidden)

        # === Decoding ===
        predictions = []

        # 第一步：使用encoder输出初始化Y
        y_prev = self.init_y_proj(encoder_h_final)  # (batch, n_y)
        predictions.append(y_prev)

        # 自回归预测
        for step in range(1, self.output_steps):
            # Decoder输入：当前控制 + 上一步预测Y
            u_current = decoder_input[:, step - 1:step, :]  # (batch, 1, n_u)
            y_input = y_prev.unsqueeze(1)  # (batch, 1, n_y)
            decoder_step_input = torch.cat([u_current, y_input], dim=2)  # (batch, 1, n_u+n_y)

            # Decoder一步
            decoder_output, (decoder_h, decoder_c) = self.decoder(
                decoder_step_input, (decoder_h, decoder_c)
            )

            # 预测下一步Y
            y_pred = self.output_layer(decoder_output.squeeze(1))  # (batch, n_y)

            # Teacher forcing或使用预测值
            if teacher_forcing and teacher_forcing_target is not None:
                y_prev = teacher_forcing_target[:, step - 1, :]
            else:
                y_prev = y_pred

            predictions.append(y_prev)

        # 合并所有预测步
        prediction = torch.stack(predictions, dim=1)  # (batch, H, n_y)

        return prediction

    def predict(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """推理模式（不使用teacher forcing）"""
        return self.forward(encoder_input, decoder_input, teacher_forcing=False)


class NARXLSTMTrainer:
    """NARX-LSTM训练器

    特点：
    1. Teacher forcing策略
    2. 早停和学习率衰减
    3. 物理约束损失
    """

    def __init__(
        self,
        model: NARXLSTM,
        device: torch.device,
        learning_rate: float = 0.001,
        step_weights: Optional[list[float]] = None,
        physics_weight: float = 0.0,
    ):
        """
        Args:
            model: NARX-LSTM模型
            device: 计算设备
            learning_rate: 学习率
            step_weights: 多步预测权重
            physics_weight: 物理约束损失权重
        """
        self.device = device
        self.learning_rate = learning_rate
        self.physics_weight = physics_weight

        # 物理约束损失
        if physics_weight > 0:
            from src.modeling.physics_loss import PhysicsConstraintLoss
            self.physics_loss_fn = PhysicsConstraintLoss(
                monotonicity_weight=0.08,
                amplitude_weight=0.15,
                boundary_weight=0.1,
                smoothness_weight=0.01,
                spatial_weight=0.02,
            )
        else:
            self.physics_loss_fn = None

        # 多步预测权重（近期权重更高）
        if step_weights is None:
            step_weights = [1.0, 0.9, 0.8, 0.7, 0.6][:model.output_steps]
        self.step_weights = torch.tensor(step_weights, device=device)

        # 模型移至设备
        self.model = model.to(device)
        self.output_steps = model.output_steps
        logger.info(f"训练设备: {device}")

        # 优化器
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        # 最佳模型状态
        self.best_model_state = None
        self.best_val_loss = float('inf')

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        decoder_input: Optional[torch.Tensor] = None,  # 新增：控制输入
    ) -> torch.Tensor:
        """计算加权MSE损失 + 物理约束损失（新增）

        Args:
            prediction: (batch, H, n_y)
            target: (batch, H, n_y)
            decoder_input: (batch, H, n_u) - 控制输入（用于物理约束）

        Returns:
            loss: 加权损失值
        """
        # MSE
        mse = F.mse_loss(prediction, target, reduction='none')  # (batch, H, n_y)

        # 按步加权
        weights = self.step_weights.view(1, -1, 1).expand_as(mse)  # (batch, H, n_y)
        weighted_mse = (mse * weights).mean()

        # 物理约束损失（新增）
        if self.physics_loss_fn is not None and decoder_input is not None:
            physics_losses = self.physics_loss_fn(prediction, control_input=decoder_input)
            total_loss = weighted_mse + self.physics_weight * physics_losses['total']
        else:
            total_loss = weighted_mse

        return total_loss

    def train_epoch(
        self,
        train_loader,
        teacher_forcing_ratio: float = 0.5,
        epoch: int = 0,
        verbose: bool = True,
    ) -> float:
        """训练一个epoch

        Args:
            train_loader: 训练数据加载器
            teacher_forcing_ratio: teacher forcing概率
            epoch: 当前epoch编号（用于日志）
            verbose: 是否打印batch进度日志

        Returns:
            avg_loss: 平均损失
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        n_total_batches = len(train_loader)

        # 每隔多少个batch打印一次进度（约打印10次）
        log_interval = max(1, n_total_batches // 10)

        for batch_idx, batch in enumerate(train_loader):
            encoder_input = batch['encoder_input'].to(self.device)
            decoder_input = batch['decoder_input'].to(self.device)
            target = batch['target'].to(self.device)

            # 随机决定是否使用teacher forcing
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio

            # 前向传播
            prediction = self.model(
                encoder_input, decoder_input,
                teacher_forcing=use_teacher_forcing,
                teacher_forcing_target=target if use_teacher_forcing else None,
            )

            # 计算损失（传入decoder_input用于物理约束）
            loss = self.compute_loss(prediction, target, decoder_input)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # 打印batch进度
            if verbose and (batch_idx + 1) % log_interval == 0:
                avg_loss_so_far = total_loss / n_batches
                progress = (batch_idx + 1) / n_total_batches * 100
                logger.info(f"  Epoch {epoch} - Batch {batch_idx+1}/{n_total_batches} ({progress:.0f}%) - Loss: {loss.item():.4f} (avg: {avg_loss_so_far:.4f})")

        return total_loss / n_batches

    def validate(self, val_loader) -> float:
        """验证

        Args:
            val_loader: 验证数据加载器

        Returns:
            avg_loss: 平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                target = batch['target'].to(self.device)

                prediction = self.model.predict(encoder_input, decoder_input)
                loss = self.compute_loss(prediction, target)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        patience: int = 15,
        teacher_forcing_ratio: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """完整训练流程

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 最大epoch数
            patience: 早停耐心值
            teacher_forcing_ratio: teacher forcing比例
            verbose: 是否打印日志

        Returns:
            history: 训练历史
        """
        history = {'train_loss': [], 'val_loss': []}
        epochs_no_improve = 0

        logger.info("开始NARX-LSTM训练")
        logger.info(f"最大epochs: {epochs}, 早停patience: {patience}")
        logger.info(f"总batch数: {len(train_loader)}")

        for epoch in range(epochs):
            # 训练（传入epoch编号用于batch进度日志）
            train_loss = self.train_epoch(
                train_loader,
                teacher_forcing_ratio,
                epoch=epoch + 1,
                verbose=verbose,
            )
            history['train_loss'].append(train_loss)

            # 验证
            val_loss = self.validate(val_loader)
            history['val_loss'].append(val_loss)

            # 学习率调度
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # 打印epoch总结日志
            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs} 完成 - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")

            # 保存最佳模型
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
                if verbose:
                    logger.info(f"  ✓ 新最佳模型，验证损失: {val_loss:.4f}")
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

    def evaluate(
        self,
        test_loader,
    ) -> dict:
        """评估模型

        Args:
            test_loader: 测试数据

        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                target = batch['target'].to(self.device)

                prediction = self.model.predict(encoder_input, decoder_input)

                predictions.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # 计算各步的指标
        metrics = {'per_step': {}}
        for step in range(self.output_steps):
            pred_step = predictions[:, step, :]
            true_step = targets[:, step, :]

            mse = float(np.mean((pred_step - true_step) ** 2))
            mae = float(np.mean(np.abs(pred_step - true_step)))
            rmse = float(np.sqrt(mse))

            metrics['per_step'][step + 1] = {
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
            }

        # 总体指标
        metrics['overall'] = {
            'mse': float(np.mean((predictions - targets) ** 2)),
            'mae': float(np.mean(np.abs(predictions - targets))),
            'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
        }

        return metrics

    def get_model_state_dict(self) -> dict:
        """获取模型状态字典（用于保存）"""
        return self.model.state_dict()


def create_narx_lstm_model(
    n_y: int,
    n_u: int,
    n_x: int,
    config: Optional[dict] = None,
) -> NARXLSTM:
    """创建NARX-LSTM模型

    Args:
        n_y: 目标变量维度
        n_u: 控制变量维度
        n_x: 状态变量维度
        config: 配置字典（可选）

    Returns:
        model: NARXLSTM模型
    """
    if config is None:
        config = NARX_LSTM_CONFIG

    model = NARXLSTM(
        n_y=n_y,
        n_u=n_u,
        n_x=n_x,
        encoder_hidden=config.get('encoder_hidden_units', 128),
        decoder_hidden=config.get('decoder_hidden_units', 128),
        encoder_layers=config.get('encoder_num_layers', 1),
        decoder_layers=config.get('decoder_num_layers', 1),
        dropout=config.get('dropout_rate', 0.2),
        bidirectional=config.get('encoder_bidirectional', True),
        output_steps=config.get('output_steps', 5),
    )

    logger.info(f"NARX-LSTM模型创建完成:")
    logger.info(f"  目标维度: {n_y}, 控制维度: {n_u}, 状态维度: {n_x}")
    logger.info(f"  编码器hidden: {config.get('encoder_hidden_units', 128)}")
    logger.info(f"  解码器hidden: {config.get('decoder_hidden_units', 128)}")
    logger.info(f"  预测步数: {config.get('output_steps', 5)}")

    return model


__all__ = [
    "NARXLSTM",
    "NARXLSTMTrainer",
    "create_narx_lstm_model",
]