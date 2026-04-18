"""
概率NARX模型 - Dual-Head Probabilistic NARX

核心思想：
- 负压是随机振荡过程（燃烧脉动），不可用确定性模型预测点值
- 含氧量是平滑慢变过程，可以确定性预测
- 负压Head输出(μ, σ)概率分布，含氧Head输出点预测

架构：
    共享LSTM编码器 → 负压概率Head (μ, σ)
                    → 含氧确定性Head (ŷ)

损失函数：
    负压: Gaussian NLL（不惩罚平滑，惩罚分布不准）
    含氧: Huber Loss
    + 物理约束正则项
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ProbNARX(nn.Module):
    """概率NARX模型

    双头架构：
    - PressureHead: 输出(μ, σ) 概率预测
    - OxygenHead: 输出点预测
    """

    def __init__(
        self,
        input_dim: int,
        control_dim: int = 7,
        n_pressure: int = 4,
        n_oxygen: int = 3,
        horizon: int = 5,
        hidden_dim: int = 128,
        encoder_layers: int = 2,
        future_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_pressure = n_pressure
        self.n_oxygen = n_oxygen
        self.horizon = horizon
        self.hidden_dim = hidden_dim

        # === 共享编码器 ===
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=encoder_layers,
            batch_first=True,
            dropout=dropout if encoder_layers > 1 else 0,
        )
        self.enc_dropout = nn.Dropout(dropout)

        # === 未来控制编码器 ===
        self.future_proj = nn.Sequential(
            nn.Linear(control_dim, future_hidden),
            nn.GELU(),
            nn.Linear(future_hidden, future_hidden),
        )

        # === 负压概率预测头 ===
        head_in = hidden_dim + future_hidden
        self.pressure_net = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 128),
            nn.GELU(),
        )
        self.pressure_mu = nn.Linear(128, n_pressure)
        self.pressure_log_sigma = nn.Linear(128, n_pressure)
        # 初始化：让初始σ≈25Pa（原始空间的负压5步std）
        nn.init.constant_(self.pressure_log_sigma.bias, np.log(25.0))

        # === 含氧量确定性预测头 ===
        self.oxygen_net = nn.Sequential(
            nn.Linear(head_in, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_oxygen),
        )

    def forward(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            encoder_input: (B, L, input_dim) 历史窗口
            decoder_input: (B, H, control_dim) 未来控制序列

        Returns:
            dict:
                'pressure_mu': (B, H, 4) 负压均值
                'pressure_sigma': (B, H, 4) 负压标准差
                'oxygen': (B, H, 3) 含氧量点预测
        """
        B = encoder_input.shape[0]

        # 编码历史
        x = self.input_proj(encoder_input)
        _, (h_n, _) = self.lstm(x)
        h = self.enc_dropout(h_n[-1])  # (B, hidden_dim)

        # 编码未来控制
        u_enc = self.future_proj(decoder_input)  # (B, H, future_hidden)

        # 拼接：h扩展到每步 + 未来控制
        h_exp = h.unsqueeze(1).expand(-1, self.horizon, -1)  # (B, H, hidden_dim)
        combined = torch.cat([h_exp, u_enc], dim=-1)  # (B, H, head_in)
        combined_flat = combined.reshape(B * self.horizon, -1)

        # 负压概率头
        p_feat = self.pressure_net(combined_flat)
        mu = self.pressure_mu(p_feat).reshape(B, self.horizon, self.n_pressure)
        log_sigma = self.pressure_log_sigma(p_feat).reshape(
            B, self.horizon, self.n_pressure
        )
        sigma = torch.exp(log_sigma.clamp(-4, 4))

        # 含氧头
        o_hat = self.oxygen_net(combined_flat).reshape(
            B, self.horizon, self.n_oxygen
        )

        return {
            "pressure_mu": mu,
            "pressure_sigma": sigma,
            "oxygen": o_hat,
        }

    def predict(
        self,
        encoder_input: torch.Tensor,
        decoder_input: torch.Tensor,
        n_samples: int = 0,
    ) -> dict[str, torch.Tensor]:
        """推理接口

        Args:
            n_samples: 如果>0，从预测分布中采样n_samples条轨迹

        Returns:
            dict:
                'pressure_mu', 'pressure_sigma', 'oxygen'
                若n_samples>0还有'pressure_samples': (B, n_samples, H, 4)
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(encoder_input, decoder_input)
            if n_samples > 0:
                mu = out["pressure_mu"]
                sigma = out["pressure_sigma"]
                # (B, 1, H, 4) + (B, 1, H, 4) * (B, n_samples, H, 4)
                eps = torch.randn(
                    mu.shape[0], n_samples, *mu.shape[1:], device=mu.device
                )
                samples = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps
                out["pressure_samples"] = samples
        return out


class ProbNARXLoss(nn.Module):
    """概率NARX损失函数

    负压: Gaussian NLL — 鼓励准确的分布预测，不强求点精度
    含氧: Huber Loss — 对异常值鲁棒的确定性损失
    + σ正则 + 安全约束软惩罚 + 步权重衰减
    """

    def __init__(
        self,
        n_pressure: int = 4,
        n_oxygen: int = 3,
        horizon: int = 5,
        oxygen_weight: float = 1.0,
        sigma_reg_weight: float = 0.001,
        safety_weight: float = 0.01,
        pressure_safe_range: tuple[float, float] = (-200.0, -20.0),
        oxygen_safe_range: tuple[float, float] = (1.0, 6.0),
        step_decay: float = 0.9,
    ):
        super().__init__()
        self.n_pressure = n_pressure
        self.n_oxygen = n_oxygen
        self.oxygen_weight = oxygen_weight
        self.sigma_reg_weight = sigma_reg_weight
        self.safety_weight = safety_weight
        self.pressure_safe_range = pressure_safe_range
        self.oxygen_safe_range = oxygen_safe_range

        step_weights = torch.tensor(
            [step_decay**i for i in range(horizon)], dtype=torch.float32
        )
        self.register_buffer("step_weights", step_weights / step_weights.sum())

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            pred: 模型输出 dict
            target: (B, H, n_pressure + n_oxygen)

        Returns:
            dict: 'total' 及各分项
        """
        y_p = target[:, :, : self.n_pressure]
        y_o = target[:, :, self.n_pressure :]

        mu = pred["pressure_mu"]
        sigma = pred["pressure_sigma"]
        o_hat = pred["oxygen"]

        # --- 负压 Gaussian NLL ---
        var = sigma**2 + 1e-6
        nll = 0.5 * (torch.log(var) + (y_p - mu) ** 2 / var)
        nll_per_step = nll.mean(dim=-1)  # (B, H)
        loss_p = (nll_per_step * self.step_weights).sum(dim=-1).mean()

        # --- 含氧量 Huber Loss ---
        huber = F.huber_loss(o_hat, y_o, reduction="none", delta=0.5)
        huber_per_step = huber.mean(dim=-1)
        loss_o = (huber_per_step * self.step_weights).sum(dim=-1).mean()

        # --- σ正则：防止坍缩或爆炸 ---
        sigma_reg = torch.log(sigma).var(dim=(1, 2)).mean()

        # --- 安全约束软惩罚 ---
        p_lo, p_hi = self.pressure_safe_range
        o_lo, o_hi = self.oxygen_safe_range
        safety = (
            F.relu(mu - p_hi).mean()
            + F.relu(p_lo - mu).mean()
            + F.relu(o_hat - o_hi).mean()
            + F.relu(o_lo - o_hat).mean()
        )

        total = (
            loss_p
            + self.oxygen_weight * loss_o
            + self.sigma_reg_weight * sigma_reg
            + self.safety_weight * safety
        )

        return {
            "total": total,
            "pressure_nll": loss_p.detach(),
            "oxygen_huber": loss_o.detach(),
            "sigma_reg": sigma_reg.detach(),
            "safety": safety.detach(),
            "sigma_mean": sigma.mean().detach(),
        }
