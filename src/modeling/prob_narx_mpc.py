"""
基于概率NARX的MPC控制优化模块

核心思想：
1. 给定当前状态，预测不同控制序列下的目标分布
2. 寻找最优控制序列，使目标值接近理想范围
3. 考虑不确定性（σ）做风险-收益权衡

优化目标：
    min  |μ_pressure - target_pressure| + λ_oxygen * |ŷ_oxygen - target_oxygen|
         + λ_risk * σ_pressure  (风险项，σ大时保守)
    s.t. 控制变量在物理范围内，单步调整幅度受限

用法：
    from src.modeling.prob_narx_mpc import ProbNARXMPC
    mpc = ProbNARXMPC(model, scaler_params)
    control_seq = mpc.optimize(current_state, target_pressure=-115, target_oxygen=2.0)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from dataclasses import dataclass

from src.config.variables import CONTROL_VARIABLES, PRESSURE_VARIABLES, OXYGEN_VARIABLES
from src.config.constraints import (
    CONTROL_RANGES,
    MAX_SINGLE_ADJUSTMENT,
    PRESSURE_TARGET,
    OXYGEN_TARGET,
    PRESSURE_IDEAL_RANGE,
    OXYGEN_IDEAL_RANGE,
)


@dataclass
class MPCResult:
    """MPC优化结果"""
    control_sequence: np.ndarray  # (H, n_u) 最优控制序列
    predicted_pressure_mu: np.ndarray  # (H, 4) 预测负压均值
    predicted_pressure_sigma: np.ndarray  # (H, 4) 预测负压σ
    predicted_oxygen: np.ndarray  # (H, 3) 预测含氧量
    total_cost: float  # 总代价
    risk_level: float  # 风险等级 (σ平均值)
    first_step_adjustment: dict  # 第一步调整量


class ProbNARXMPC:
    """基于概率NARX的MPC控制器"""

    def __init__(
        self,
        model: nn.Module,
        scaler_params: dict,
        device: torch.device = None,
        horizon: int = 5,
        n_pressure: int = 4,
        n_oxygen: int = 3,
        # 代价权重
        pressure_weight: float = 1.0,
        oxygen_weight: float = 1.0,
        risk_weight: float = 0.5,  # σ风险权重
        control_change_weight: float = 0.1,  # 控制变化惩罚
        # 目标值
        target_pressure: float = PRESSURE_TARGET,  # -115 Pa
        target_oxygen: float = OXYGEN_TARGET,  # 2.0 %
        # 优化参数
        max_iterations: int = 100,
        learning_rate: float = 0.01,
    ):
        self.model = model
        self.scaler_params = scaler_params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.horizon = horizon
        self.n_pressure = n_pressure
        self.n_oxygen = n_oxygen

        self.pressure_weight = pressure_weight
        self.oxygen_weight = oxygen_weight
        self.risk_weight = risk_weight
        self.control_change_weight = control_change_weight

        self.target_pressure = target_pressure
        self.target_oxygen = target_oxygen

        self.max_iterations = max_iterations
        self.learning_rate = learning_rate

        # 控制变量范围（标准化空间）
        u_mean = scaler_params["u"]["mean"]
        u_scale = scaler_params["u"]["scale"]
        self.u_bounds = []
        for i, var in enumerate(CONTROL_VARIABLES):
            lo, hi = CONTROL_RANGES.get(var, (0, 100))
            lo_scaled = (lo - u_mean[i]) / u_scale[i]
            hi_scaled = (hi - u_mean[i]) / u_scale[i]
            self.u_bounds.append((lo_scaled, hi_scaled))

        # 目标值标准化
        y_mean = scaler_params["y"]["mean"]
        y_scale = scaler_params["y"]["scale"]
        self.target_p_scaled = (target_pressure - y_mean[0]) / y_scale[0]
        self.target_o_scaled = (target_oxygen - y_mean[4]) / y_scale[4]

    def optimize(
        self,
        encoder_input: np.ndarray,
        current_control: np.ndarray,
        target_pressure: Optional[float] = None,
        target_oxygen: Optional[float] = None,
        method: str = "gradient",  # "gradient" or "sampling"
    ) -> MPCResult:
        """
        优化控制序列

        Args:
            encoder_input: (L, input_dim) 当前历史窗口（已标准化）
            current_control: (n_u,) 当前控制值（已标准化）
            target_pressure: 目标负压（Pa），可选覆盖默认值
            target_oxygen: 目标含氧量（%），可选覆盖默认值
            method: 优化方法
                - "gradient": 梯度下降（快，适合在线）
                - "sampling": 随机采样（鲁棒，适合离线）

        Returns:
            MPCResult
        """
        if target_pressure is not None:
            y_mean = self.scaler_params["y"]["mean"]
            y_scale = self.scaler_params["y"]["scale"]
            self.target_p_scaled = (target_pressure - y_mean[0]) / y_scale[0]
        if target_oxygen is not None:
            y_mean = self.scaler_params["y"]["mean"]
            y_scale = self.scaler_params["y"]["scale"]
            self.target_o_scaled = (target_oxygen - y_mean[4]) / y_scale[4]

        if method == "gradient":
            return self._optimize_gradient(encoder_input, current_control)
        else:
            return self._optimize_sampling(encoder_input, current_control)

    def _optimize_gradient(
        self,
        encoder_input: np.ndarray,
        current_control: np.ndarray,
    ) -> MPCResult:
        """梯度下降优化（需要临时启用训练模式）"""
        # 初始化控制序列：从当前值平滑过渡
        u_init = np.tile(current_control, (self.horizon, 1))  # (H, n_u)

        # 转为可优化tensor
        u_seq = torch.tensor(u_init, dtype=torch.float32, requires_grad=True, device=self.device)
        enc_in = torch.tensor(encoder_input, dtype=torch.float32, device=self.device)
        enc_in = enc_in.unsqueeze(0)  # (1, L, input_dim)

        optimizer = torch.optim.Adam([u_seq], lr=self.learning_rate)

        best_cost = float("inf")
        best_u = u_init.copy()

        # 临时启用训练模式以支持LSTM反向传播（但保持dropout=0）
        self.model.train()

        for it in range(self.max_iterations):
            # 预测
            pred = self.model(enc_in, u_seq.unsqueeze(0))
            mu = pred["pressure_mu"][0]  # (H, 4)
            sigma = pred["pressure_sigma"][0]  # (H, 4)
            o_hat = pred["oxygen"][0]  # (H, 3)

            # 代价函数
            cost = self._compute_cost(u_seq, mu, sigma, o_hat, current_control)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # 约束投影
            with torch.no_grad():
                for i, (lo, hi) in enumerate(self.u_bounds):
                    u_seq[:, i] = torch.clamp(u_seq[:, i], lo, hi)

            if cost.item() < best_cost:
                best_cost = cost.item()
                best_u = u_seq.detach().cpu().numpy().copy()

        # 恢复eval模式
        self.model.eval()

        # 最终预测
        u_final = torch.tensor(best_u, dtype=torch.float32, device=self.device)
        pred = self.model(enc_in, u_final.unsqueeze(0))

        return self._build_result(best_u, pred, current_control)

    def _optimize_sampling(
        self,
        encoder_input: np.ndarray,
        current_control: np.ndarray,
        n_samples: int = 100,
    ) -> MPCResult:
        """随机采样优化（更鲁棒）"""
        enc_in = torch.tensor(encoder_input, dtype=torch.float32, device=self.device)
        enc_in = enc_in.unsqueeze(0)

        best_cost = float("inf")
        best_u = None

        # 生成候选控制序列
        for _ in range(n_samples):
            # 从当前值附近采样
            u_candidate = np.tile(current_control, (self.horizon, 1))
            # 添加随机扰动
            for i, (lo, hi) in enumerate(self.u_bounds):
                delta = np.random.uniform(-0.5, 0.5, self.horizon)  # 标准化空间的扰动
                u_candidate[:, i] = np.clip(u_candidate[:, i] + delta, lo, hi)

            # 预测
            u_tensor = torch.tensor(u_candidate, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                pred = self.model(enc_in, u_tensor.unsqueeze(0))

            mu = pred["pressure_mu"][0]
            sigma = pred["pressure_sigma"][0]
            o_hat = pred["oxygen"][0]

            cost = self._compute_cost_numpy(u_candidate, mu.cpu().numpy(),
                                            sigma.cpu().numpy(), o_hat.cpu().numpy(),
                                            current_control)

            if cost < best_cost:
                best_cost = cost
                best_u = u_candidate.copy()

        # 最终预测
        u_final = torch.tensor(best_u, dtype=torch.float32, device=self.device)
        pred = self.model(enc_in, u_final.unsqueeze(0))

        return self._build_result(best_u, pred, current_control)

    def _compute_cost(
        self,
        u_seq: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        o_hat: torch.Tensor,
        current_control: np.ndarray,
    ) -> torch.Tensor:
        """计算代价函数"""
        # 目标偏差
        p_deviation = (mu - self.target_p_scaled).abs().mean()
        o_deviation = (o_hat - self.target_o_scaled).abs().mean()

        # 风险项（σ）
        risk = sigma.mean()

        # 控制变化惩罚
        u_change = (u_seq[0] - torch.tensor(current_control, device=self.device)).abs().mean()

        total = (
            self.pressure_weight * p_deviation
            + self.oxygen_weight * o_deviation
            + self.risk_weight * risk
            + self.control_change_weight * u_change
        )
        return total

    def _compute_cost_numpy(
        self,
        u_seq: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        o_hat: np.ndarray,
        current_control: np.ndarray,
    ) -> float:
        """numpy版本代价计算"""
        p_deviation = np.abs(mu - self.target_p_scaled).mean()
        o_deviation = np.abs(o_hat - self.target_o_scaled).mean()
        risk = sigma.mean()
        u_change = np.abs(u_seq[0] - current_control).mean()

        return (
            self.pressure_weight * p_deviation
            + self.oxygen_weight * o_deviation
            + self.risk_weight * risk
            + self.control_change_weight * u_change
        )

    def _build_result(
        self,
        u_seq: np.ndarray,
        pred: dict,
        current_control: np.ndarray,
    ) -> MPCResult:
        """构建结果"""
        mu = pred["pressure_mu"][0].detach().cpu().numpy()
        sigma = pred["pressure_sigma"][0].detach().cpu().numpy()
        o_hat = pred["oxygen"][0].detach().cpu().numpy()

        # 反标准化
        y_mean = self.scaler_params["y"]["mean"]
        y_scale = self.scaler_params["y"]["scale"]
        u_mean = self.scaler_params["u"]["mean"]
        u_scale = self.scaler_params["u"]["scale"]

        mu_original = mu * y_scale[:4] + y_mean[:4]
        sigma_original = sigma * y_scale[:4]
        o_original = o_hat * y_scale[4:7] + y_mean[4:7]
        u_original = u_seq * u_scale + u_mean

        # 第一步调整量
        u_current_original = current_control * u_scale + u_mean
        adjustment = {var: u_original[0, i] - u_current_original[i]
                      for i, var in enumerate(CONTROL_VARIABLES)}

        cost = self._compute_cost_numpy(u_seq, mu, sigma, o_hat, current_control)

        return MPCResult(
            control_sequence=u_original,
            predicted_pressure_mu=mu_original,
            predicted_pressure_sigma=sigma_original,
            predicted_oxygen=o_original,
            total_cost=cost,
            risk_level=sigma_original.mean(),
            first_step_adjustment=adjustment,
        )


def run_mpc_demo():
    """MPC演示"""
    import json
    from src.modeling.prob_narx import ProbNARX

    # 加载模型
    model_dir = Path("output/models/prob_narx")
    ckpt = torch.load(model_dir / "best_model.pt", map_location="cuda", weights_only=False)
    with open(model_dir / "scaler_params.json") as f:
        scaler_params = json.load(f)

    args = ckpt["args"]
    info = ckpt["info"]

    model = ProbNARX(
        input_dim=info["encoder_input_shape"][2],
        control_dim=info["decoder_input_shape"][2],
        n_pressure=4,
        n_oxygen=3,
        horizon=args["horizon"],
        hidden_dim=args["hidden_dim"],
        encoder_layers=args["encoder_layers"],
        future_hidden=args["future_hidden"],
        dropout=0.0,  # 推理时不dropout
    ).to("cuda")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 创建MPC控制器
    mpc = ProbNARXMPC(
        model=model,
        scaler_params=scaler_params,
        device=torch.device("cuda"),
        horizon=args["horizon"],
        risk_weight=0.5,
        control_change_weight=0.2,
    )

    # 模拟当前状态（随机取一个测试样本）
    from src.data.window_sampler import MPCWindowSampler
    from src.data.mpc_dataset import split_narx_data
    sampler = MPCWindowSampler(
        data_path="output/all_data_cleaned.feather",
        history_length=args["history_length"],
        prediction_horizon=args["horizon"],
    )
    enc_in, dec_in, target, _ = sampler.build_samples_with_future_control(scale=True)
    _, _, test_data = split_narx_data(enc_in, dec_in, target)

    # 取一个样本
    sample_idx = 1000
    encoder_input = test_data["encoder_input"][sample_idx]  # (L, input_dim)
    current_control = test_data["decoder_input"][sample_idx, 0]  # 当前控制（第一步）

    # 优化
    print("=" * 60)
    print("MPC控制优化演示")
    print("=" * 60)
    print(f"目标负压: {mpc.target_pressure} Pa")
    print(f"目标含氧量: {mpc.target_oxygen} %")

    result = mpc.optimize(encoder_input, current_control, method="gradient")

    print("\n优化结果:")
    print(f"  总代价: {result.total_cost:.4f}")
    print(f"  风险等级(σ均值): {result.risk_level:.2f} Pa")
    print("\n第一步控制调整:")
    for var, delta in result.first_step_adjustment.items():
        print(f"    {var}: Δ={delta:+.2f}")

    print("\n预测负压均值 (Pa):")
    print(f"    步1: {result.predicted_pressure_mu[0]}")
    print(f"    步5: {result.predicted_pressure_mu[4]}")

    print("\n预测含氧量 (%):")
    print(f"    步1: {result.predicted_oxygen[0]}")
    print(f"    步5: {result.predicted_oxygen[4]}")

    return result


if __name__ == "__main__":
    run_mpc_demo()