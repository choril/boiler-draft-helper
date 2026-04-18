"""
改进版NARX-LSTM模型

改进目标：解决压力突变时预测响应滞后的问题

改进措施：
1. 控制变化率特征融合：decoder直接接收Δu信号
2. 增强控制融合权重：控制输入在decoder中权重放大
3. 双向编码器：捕获历史趋势
4. 残差连接：第一步预测直接从历史末尾延续
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ControlFusionLayer(nn.Module):
    """控制融合层

    增强控制输入在预测中的影响力：
    - 对控制输入进行放大变换
    - 将控制变化率(Δu)直接融入预测

    输入：
        control: 当前控制 u[t]
        control_diff: 控制变化率 Δu[t]
        hidden: LSTM隐藏状态

    输出：
        fused: 融合后的特征向量
    """

    def __init__(
        self,
        control_size: int,
        hidden_size: int,
        control_weight: float = 2.0,
    ):
        super().__init__()
        self.control_size = control_size
        self.hidden_size = hidden_size
        self.control_weight = control_weight

        # 控制值投影（带权重放大）
        self.control_proj = nn.Sequential(
            nn.Linear(control_size, hidden_size),
            nn.LayerNorm(hidden_size),  # 标准化后放大
        )

        # 控制变化率投影（直接反映响应趋势）
        self.diff_proj = nn.Sequential(
            nn.Linear(control_size, hidden_size),
            nn.Tanh(),  # 变化率用Tanh限制范围
        )

        # 融合门控：决定控制信号的影响程度
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_size + control_size * 2, hidden_size),
            nn.Sigmoid(),
        )

        # 最终融合
        self.fusion = nn.Linear(hidden_size * 3, hidden_size)

    def forward(
        self,
        control: torch.Tensor,
        control_diff: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            control: (batch, n_u) 当前控制值
            control_diff: (batch, n_u) 控制变化率
            hidden: (batch, hidden_size) LSTM隐藏状态

        Returns:
            fused: (batch, hidden_size) 融合特征
        """
        # 控制值投影（带权重放大）
        control_feat = self.control_proj(control) * self.control_weight

        # 控制变化率投影（强调突变响应）
        diff_feat = self.diff_proj(control_diff) * self.control_weight

        # 门控融合
        gate_input = torch.cat([hidden, control, control_diff], dim=-1)
        gate = self.fusion_gate(gate_input)

        # 融合所有特征
        concat = torch.cat([hidden, control_feat, diff_feat], dim=-1)
        fused = self.fusion(concat)

        # 门控输出
        output = gate * fused + (1 - gate) * hidden

        return output


class ImprovedEncoder(nn.Module):
    """改进版编码器

    特点：
    1. 双向LSTM捕获历史趋势
    2. 注意力机制关注历史末尾（突变检测）
    3. 输出历史末尾的残差连接
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 输出大小
        self.output_size = hidden_size * self.directions

        # 历史末尾注意力（关注最近几步）
        self.tail_attention = nn.Sequential(
            nn.Linear(self.output_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # 残差投影：从输入直接投影到输出维度
        self.residual_proj = nn.Linear(input_size, hidden_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, L, input_size) 历史输入

        Returns:
            dict:
                - 'output': (batch, L, output_size) LSTM输出序列
                - 'hidden': (h_n, c_n) LSTM隐藏状态
                - 'tail_context': (batch, output_size) 历史末尾上下文
                - 'residual': (batch, hidden_size) 从历史末尾的直接残差
        """
        batch_size, L, _ = x.shape

        # LSTM编码
        output, (h_n, c_n) = self.lstm(x)
        # output: (batch, L, hidden_size * directions)

        # 历史末尾注意力：关注最近的时刻
        attn_weights = F.softmax(self.tail_attention(output), dim=1)  # (batch, L, 1)
        tail_context = (output * attn_weights).sum(dim=1)  # (batch, output_size)

        # 残差连接：直接从历史末尾投影
        residual = self.residual_proj(x[:, -1, :])  # (batch, hidden_size)

        return {
            'output': output,
            'hidden': (h_n, c_n),
            'tail_context': tail_context,
            'residual': residual,
        }


class ImprovedDecoder(nn.Module):
    """改进版解码器

    特点：
    1. 控制变化率直接融合（响应突变）
    2. 残差预测策略（预测变化量而非绝对值）
    3. 增强的控制权重
    """

    def __init__(
        self,
        control_size: int,
        hidden_size: int,
        output_size: int,  # n_y
        num_layers: int = 2,
        dropout: float = 0.2,
        encoder_bidirectional: bool = True,
        control_weight: float = 2.0,
    ):
        super().__init__()
        self.control_size = control_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder_bidirectional = encoder_bidirectional
        self.control_weight = control_weight

        # 控制融合层
        self.control_fusion = ControlFusionLayer(
            control_size=control_size,
            hidden_size=hidden_size,
            control_weight=control_weight,
        )

        # LSTM解码器
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # 融合后的特征
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )

        # 隐藏状态投影（从编码器到解码器）
        if encoder_bidirectional:
            self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
            self.cell_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.hidden_proj = nn.Identity()
            self.cell_proj = nn.Identity()

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

        # 残差输出层：预测相对于历史末尾的变化量
        self.residual_output = nn.Linear(hidden_size * 2, output_size)

    def init_hidden_state(
        self,
        encoder_hidden: tuple,
        encoder_bidirectional: bool = True,
    ) -> tuple:
        """从编码器初始化解码器隐藏状态"""
        h_n, c_n = encoder_hidden

        if encoder_bidirectional:
            # 合并双向
            # h_n: (num_layers * 2, batch, hidden_size)
            h_forward = h_n[-2]  # forward最后一层
            h_backward = h_n[-1]  # backward最后一层
            h_combined = torch.cat([h_forward, h_backward], dim=-1)  # (batch, hidden*2)
            h_init = self.hidden_proj(h_combined)  # (batch, hidden)

            c_forward = c_n[-2]
            c_backward = c_n[-1]
            c_combined = torch.cat([c_forward, c_backward], dim=-1)
            c_init = self.cell_proj(c_combined)
        else:
            h_init = self.hidden_proj(h_n[-1])
            c_init = self.cell_proj(c_n[-1])

        # 为所有层复制
        h_init = h_init.unsqueeze(0).repeat(self.num_layers, 1, 1)
        c_init = c_init.unsqueeze(0).repeat(self.num_layers, 1, 1)

        return (h_init, c_init)

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_hidden: tuple,
        encoder_residual: torch.Tensor,  # 新增：编码器残差
        control_input: torch.Tensor,
        control_diff: torch.Tensor,  # 新增：控制变化率
        teacher_forcing_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
        y_prev: Optional[torch.Tensor] = None,  # 历史末尾Y值（用于残差预测）
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, L, hidden*2) 编码器输出序列
            encoder_hidden: (h_n, c_n) 编码器隐藏状态
            encoder_residual: (batch, hidden) 编码器残差连接
            control_input: (batch, H, n_u) 未来控制序列
            control_diff: (batch, H, n_u) 控制变化率序列
            teacher_forcing_target: (batch, H, n_y) Teacher forcing目标
            teacher_forcing_ratio: Teacher forcing概率
            y_prev: (batch, n_y) 历史末尾Y值

        Returns:
            predictions: (batch, H, n_y)
        """
        batch_size = control_input.size(0)
        H = control_input.size(1)

        # 初始化隐藏状态
        hidden = self.init_hidden_state(encoder_hidden, self.encoder_bidirectional)

        # 初始输入：编码器残差
        decoder_input = encoder_residual.unsqueeze(1)  # (batch, 1, hidden)

        predictions = []

        for step in range(H):
            # LSTM一步
            decoder_output, hidden = self.lstm(decoder_input, hidden)
            # decoder_output: (batch, 1, hidden)

            # 融合控制信息（关键改进）
            control_current = control_input[:, step, :]  # (batch, n_u)
            diff_current = control_diff[:, step, :]  # (batch, n_u)

            fused = self.control_fusion(
                control=control_current,
                control_diff=diff_current,
                hidden=decoder_output.squeeze(1),
            )  # (batch, hidden)

            # 输出预测
            # 结合LSTM输出和融合特征
            combined = torch.cat([decoder_output.squeeze(1), fused], dim=-1)
            pred_change = self.residual_output(combined)  # 预测变化量

            # 如果有历史末尾Y，则残差预测：pred = y_prev + pred_change
            if y_prev is not None:
                pred = y_prev + pred_change
            else:
                # 直接预测绝对值
                pred = self.output_layer(fused)

            predictions.append(pred)

            # 下一步输入：融合后的控制特征
            next_input = fused.unsqueeze(1)  # (batch, 1, hidden)

            # Teacher forcing
            if teacher_forcing_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # 使用真实值调整隐藏状态（但输入仍用控制）
                # 这里简化处理，不使用真实Y作为输入
                decoder_input = next_input
            else:
                decoder_input = next_input

        # 合并预测
        predictions = torch.stack(predictions, dim=1)  # (batch, H, n_y)

        return predictions


class ImprovedNARXLSTM(nn.Module):
    """改进版NARX-LSTM模型

    整体架构：
    - Encoder: 双向LSTM + 历史末尾注意力 + 残差连接
    - Decoder: LSTM + 控制融合层 + 残差预测

    输入：
        encoder_input: (batch, L, n_y + n_u + n_x)
        decoder_input: (batch, H, n_u)
        decoder_input_diff: (batch, H, n_u) 控制变化率

    输出：
        prediction: (batch, H, n_y)
    """

    def __init__(
        self,
        n_y: int = 7,
        n_u: int = 7,
        n_x: int = 50,
        encoder_hidden: int = 256,
        decoder_hidden: int = 256,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        output_steps: int = 5,
        control_weight: float = 2.0,
    ):
        super().__init__()
        self.n_y = n_y
        self.n_u = n_u
        self.n_x = n_x
        self.output_steps = output_steps
        self.encoder_hidden = encoder_hidden
        self.decoder_hidden = decoder_hidden
        self.control_weight = control_weight

        input_size = n_y + n_u + n_x

        # 编码器
        self.encoder = ImprovedEncoder(
            input_size=input_size,
            hidden_size=encoder_hidden,
            num_layers=encoder_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # 解码器
        self.decoder = ImprovedDecoder(
            control_size=n_u,
            hidden_size=decoder_hidden,
            output_size=n_y,
            num_layers=decoder_layers,
            dropout=dropout,
            encoder_bidirectional=bidirectional,
            control_weight=control_weight,
        )

        # 编码器输出投影（如果hidden_size不同）
        if encoder_hidden != decoder_hidden:
            self.encoder_to_decoder = nn.Linear(encoder_hidden * 2 if bidirectional else encoder_hidden,
                                                decoder_hidden)
        else:
            self.encoder_to_decoder = nn.Identity()

        logger.info(f"改进版NARX-LSTM创建完成:")
        logger.info(f"  控制融合权重: {control_weight}倍")
        logger.info(f"  包含控制变化率特征")
        logger.info(f"  残差预测策略")

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_input_diff: torch.Tensor,  # 新增：控制变化率
        teacher_forcing_ratio: float = 0.0,
        teacher_forcing_target: Optional[torch.Tensor] = None,
        y_prev: Optional[torch.Tensor] = None,  # 历史末尾Y值
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: (batch, L, n_y + n_u + n_x)
            decoder_input: (batch, H, n_u)
            decoder_input_diff: (batch, H, n_u) 控制变化率
            teacher_forcing_ratio: Teacher forcing概率
            teacher_forcing_target: (batch, H, n_y)
            y_prev: (batch, n_y) 历史末尾Y值（用于残差预测）

        Returns:
            prediction: (batch, H, n_y)
        """
        # 编码
        encoder_out = self.encoder(encoder_input)

        # 提取历史末尾Y值（如果未提供）
        if y_prev is None:
            # encoder_input的前n_y列是Y历史
            y_prev = encoder_input[:, -1, :self.n_y]  # (batch, n_y)

        # 解码
        prediction = self.decoder(
            encoder_output=encoder_out['output'],
            encoder_hidden=encoder_out['hidden'],
            encoder_residual=encoder_out['residual'],
            control_input=decoder_input,
            control_diff=decoder_input_diff,
            teacher_forcing_target=teacher_forcing_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
            y_prev=y_prev,
        )

        return prediction

    def predict(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        decoder_input_diff: torch.Tensor,
    ) -> torch.Tensor:
        """推理模式"""
        return self.forward(
            encoder_input,
            decoder_input,
            decoder_input_diff,
            teacher_forcing_ratio=0.0,
        )


class ImprovedNARXLSTMTrainer:
    """改进版训练器"""

    def __init__(
        self,
        model: ImprovedNARXLSTM,
        device: torch.device,
        learning_rate: float = 0.001,
        step_weights: list = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate

        if step_weights is None:
            step_weights = [1.5, 1.2, 1.0, 0.9, 0.8][:model.output_steps]
        self.step_weights = torch.tensor(step_weights, device=device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
        )

        self.best_model_state = None
        self.best_val_loss = float('inf')

    def compute_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """加权MSE损失"""
        mse = F.mse_loss(prediction, target, reduction='none')  # (batch, H, n_y)
        weights = self.step_weights.view(1, -1, 1).expand_as(mse)
        weighted_mse = (mse * weights).mean()
        return weighted_mse

    def train_epoch(
        self,
        train_loader: DataLoader,
        teacher_forcing_ratio: float = 0.5,
        epoch: int = 0,
    ) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            encoder_input = batch['encoder_input'].to(self.device)
            decoder_input = batch['decoder_input'].to(self.device)
            decoder_input_diff = batch['decoder_input_diff'].to(self.device)
            target = batch['target'].to(self.device)

            use_tf = batch.get('use_teacher_forcing', False)

            prediction = self.model(
                encoder_input,
                decoder_input,
                decoder_input_diff,
                teacher_forcing_ratio=teacher_forcing_ratio if use_tf else 0.0,
                teacher_forcing_target=target if use_tf else None,
            )

            loss = self.compute_loss(prediction, target)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def validate(self, val_loader: DataLoader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                decoder_input_diff = batch['decoder_input_diff'].to(self.device)
                target = batch['target'].to(self.device)

                prediction = self.model.predict(encoder_input, decoder_input, decoder_input_diff)
                loss = self.compute_loss(prediction, target)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        patience: int = 15,
        teacher_forcing_ratio: float = 0.5,
        verbose: bool = True,
    ) -> dict:
        """完整训练流程"""
        history = {'train_loss': [], 'val_loss': []}
        epochs_no_improve = 0

        logger.info(f"开始训练，最大epochs={epochs}, patience={patience}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, teacher_forcing_ratio, epoch)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            if verbose:
                logger.info(f"Epoch {epoch+1}/{epochs}: Train={train_loss:.4f}, Val={val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
                if verbose:
                    logger.info(f"  ✓ 新最佳模型，Val={val_loss:.4f}")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                logger.info(f"早停触发，最佳epoch={epoch-patience+1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            logger.info(f"加载最佳模型，Val={self.best_val_loss:.4f}")

        return history

    def evaluate(self, test_loader: DataLoader) -> dict:
        """评估"""
        self.model.eval()
        predictions = []
        targets = []

        with torch.no_grad():
            for batch in test_loader:
                encoder_input = batch['encoder_input'].to(self.device)
                decoder_input = batch['decoder_input'].to(self.device)
                decoder_input_diff = batch['decoder_input_diff'].to(self.device)
                target = batch['target'].to(self.device)

                prediction = self.model.predict(encoder_input, decoder_input, decoder_input_diff)

                predictions.append(prediction.cpu().numpy())
                targets.append(target.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        # 计算指标
        metrics = {'per_step': {}}
        for step in range(self.model.output_steps):
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

        metrics['overall'] = {
            'mse': float(np.mean((predictions - targets) ** 2)),
            'mae': float(np.mean(np.abs(predictions - targets))),
            'rmse': float(np.sqrt(np.mean((predictions - targets) ** 2))),
        }

        return metrics

    def get_model_state_dict(self) -> dict:
        return self.model.state_dict()


__all__ = [
    "ImprovedNARXLSTM",
    "ImprovedNARXLSTMTrainer",
    "ImprovedEncoder",
    "ImprovedDecoder",
    "ControlFusionLayer",
]