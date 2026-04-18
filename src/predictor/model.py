"""
预测模型模块

架构：简洁的 Encoder-Decoder LSTM

Encoder: 双向LSTM编码历史窗口
Decoder: LSTM解码 + 控制输入融合
Output: 逐步预测未来H步

输入：
    encoder_input: (batch, L, n_features) 历史窗口数据
    decoder_input: (batch, H, n_u) 未来控制输入

输出：
    prediction: (batch, H, n_y) 预测的未来目标值

使用方式：
    from src.predictor.model import BoilerPredictor

    model = BoilerPredictor(config)
    prediction = model(encoder_input, decoder_input)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .config import Config
from .utils import get_logger

logger = get_logger(__name__)


class Encoder(nn.Module):
    """编码器 - 双向LSTM

    功能：编码历史窗口数据，提取时序特征

    输入: (batch, L, n_features)
    输出: (batch, L, hidden_size * 2) 和 hidden state
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # 双向LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # 输出维度
        self.output_size = hidden_size * 2 if bidirectional else hidden_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: (batch, L, input_size)

        Returns:
            output: (batch, L, output_size)
            hidden: (h_n, c_n) LSTM隐藏状态
        """
        output, hidden = self.lstm(x)
        return output, hidden


class Decoder(nn.Module):
    """解码器 - LSTM + 控制融合

    功能：逐步解码，融合控制输入，输出预测

    输入:
        - initial_state: Encoder的隐藏状态
        - control_input: (batch, H, n_u) 未来控制

    输出: (batch, H, n_y)
    """

    def __init__(
        self,
        control_size: int,
        hidden_size: int,
        output_size: int,  # n_y
        num_layers: int = 2,
        dropout: float = 0.1,
        encoder_bidirectional: bool = True,
    ):
        super().__init__()
        self.control_size = control_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.encoder_bidirectional = encoder_bidirectional

        # 控制输入投影
        self.control_proj = nn.Linear(control_size, hidden_size)

        # LSTM输入维度 = hidden_size（来自控制投影或前一时刻输出）
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,  # Decoder不需要双向
        )

        # 输出投影（hidden_size -> n_y）
        self.output_proj = nn.Linear(hidden_size, output_size)

        # 如果Encoder是双向的，需要调整hidden state维度
        if encoder_bidirectional:
            # 将双向的hidden state (num_layers * 2, batch, hidden) 转换为
            # 单向的 (num_layers, batch, hidden)
            self.hidden_proj = nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.hidden_proj = None

    def forward(
        self,
        encoder_output: torch.Tensor,
        encoder_hidden: Tuple[torch.Tensor, torch.Tensor],
        control_input: torch.Tensor,
        teacher_forcing_target: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Args:
            encoder_output: (batch, L, hidden_size * 2) Encoder输出
            encoder_hidden: (h_n, c_n) Encoder隐藏状态
            control_input: (batch, H, n_u) 未来控制输入
            teacher_forcing_target: (batch, H, n_y) Teacher forcing目标（可选）
            teacher_forcing_ratio: Teacher forcing概率

        Returns:
            predictions: (batch, H, n_y)
        """
        batch_size = control_input.size(0)
        H = control_input.size(1)

        # 调整hidden state维度
        h_n, c_n = encoder_hidden
        if self.hidden_proj is not None:
            # h_n: (num_layers * 2, batch, hidden_size)
            # 需要将前向和后向合并
            h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
            h_n = h_n.permute(1, 2, 0, 3).contiguous()  # (2, batch, num_layers, hidden)
            h_n_forward = h_n[0].permute(1, 0, 2)  # (num_layers, batch, hidden)
            h_n_backward = h_n[1].permute(1, 0, 2)
            h_n_combined = torch.cat([h_n_forward, h_n_backward], dim=-1)  # (num_layers, batch, hidden*2)
            h_n = self.hidden_proj(h_n_combined)  # (num_layers, batch, hidden)

            c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)
            c_n = c_n.permute(1, 2, 0, 3).contiguous()
            c_n_forward = c_n[0].permute(1, 0, 2)
            c_n_backward = c_n[1].permute(1, 0, 2)
            c_n_combined = torch.cat([c_n_forward, c_n_backward], dim=-1)
            c_n = self.hidden_proj(c_n_combined)

        hidden = (h_n, c_n)

        # 初始输入：Encoder最后时刻的输出（取前向部分）
        if self.encoder_bidirectional:
            # encoder_output: (batch, L, hidden_size * 2)
            # 取最后时刻，并投影
            init_input = encoder_output[:, -1, :self.hidden_size]  # (batch, hidden_size)
        else:
            init_input = encoder_output[:, -1, :]

        # 逐步解码
        predictions = []
        decoder_input = init_input.unsqueeze(1)  # (batch, 1, hidden_size)

        for h in range(H):
            # LSTM解码一步
            decoder_output, hidden = self.lstm(decoder_input, hidden)
            # decoder_output: (batch, 1, hidden_size)

            # 输出预测
            pred = self.output_proj(decoder_output.squeeze(1))  # (batch, n_y)
            predictions.append(pred)

            # 下一步输入：融合控制
            control_embed = self.control_proj(control_input[:, h])  # (batch, hidden_size)

            # Teacher forcing：使用真实值或预测值
            if teacher_forcing_target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # 使用真实目标值（投影到hidden_size维度）
                # 这里简化处理：将预测值和控制融合作为输入
                next_input = control_embed.unsqueeze(1)
            else:
                # 使用预测值 + 控制融合
                next_input = control_embed.unsqueeze(1)

            decoder_input = next_input

        # 拼接所有预测
        predictions = torch.stack(predictions, dim=1)  # (batch, H, n_y)

        return predictions


class BoilerPredictor(nn.Module):
    """锅炉预测模型

    整体架构：Encoder-Decoder LSTM

    功能：给定历史数据和未来控制，预测未来目标值（负压/含氧）

    输入：
        - encoder_input: (batch, L, n_y + n_u + n_x) 历史窗口
        - decoder_input: (batch, H, n_u) 未来控制

    输出：
        - prediction: (batch, H, n_y) 预测的未来目标值
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.L = config.L
        self.H = config.H
        self.n_y = config.n_y
        self.n_u = config.n_u

        # 计算输入维度（假设n_x会动态确定）
        # 默认使用 config 中可能存在的 n_x 或通过参数传入
        self.n_x = getattr(config, 'n_x', 38)  # 状态变量维度

        # 模型参数
        hidden_size = config.model.hidden_size
        num_layers = config.model.num_layers
        dropout = config.model.dropout
        bidirectional = config.model.bidirectional

        # 编码器
        self.encoder = Encoder(
            input_size=self.n_y + self.n_u + self.n_x,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # 解码器
        self.decoder = Decoder(
            control_size=self.n_u,
            hidden_size=hidden_size,
            output_size=self.n_y,
            num_layers=num_layers,
            dropout=dropout,
            encoder_bidirectional=bidirectional,
        )

        logger.info(f"BoilerPredictor 创建完成:")
        logger.info(f"  输入维度: Y={self.n_y}, U={self.n_u}, X={self.n_x}")
        logger.info(f"  窗口参数: L={self.L}, H={self.H}")
        logger.info(f"  模型参数: hidden={hidden_size}, layers={num_layers}, bidirectional={bidirectional}")

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        teacher_forcing_ratio: float = 0.0,
        teacher_forcing_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            encoder_input: (batch, L, n_y + n_u + n_x)
            decoder_input: (batch, H, n_u)
            teacher_forcing_ratio: Teacher forcing概率
            teacher_forcing_target: (batch, H, n_y) 用于teacher forcing

        Returns:
            prediction: (batch, H, n_y)
        """
        # 编码
        encoder_output, encoder_hidden = self.encoder(encoder_input)

        # 解码
        prediction = self.decoder(
            encoder_output,
            encoder_hidden,
            decoder_input,
            teacher_forcing_target=teacher_forcing_target,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return prediction

    def predict(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> torch.Tensor:
        """推理模式（无teacher forcing）

        Args:
            encoder_input: (batch, L, n_features)
            decoder_input: (batch, H, n_u)

        Returns:
            prediction: (batch, H, n_y)
        """
        return self.forward(encoder_input, decoder_input, teacher_forcing_ratio=0.0)


def create_model(config: Config, n_x: Optional[int] = None) -> BoilerPredictor:
    """创建模型（工厂函数）

    Args:
        config: 配置对象
        n_x: 状态变量维度（可选，用于动态设置）

    Returns:
        model: BoilerPredictor模型
    """
    if n_x is not None:
        config.n_x = n_x

    model = BoilerPredictor(config)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"模型参数量:")
    logger.info(f"  总参数: {total_params:,}")
    logger.info(f"  可训练: {trainable_params:,}")

    return model


__all__ = [
    "Encoder",
    "Decoder",
    "BoilerPredictor",
    "create_model",
]