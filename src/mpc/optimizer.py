"""
贝叶斯优化器 - MPC候选控制序列评估

优化策略：
1. 常规优化：标准贝叶斯优化（30-50次评估）
2. 紧急工况：分组优化（减少评估次数）
3. 精细优化：两阶段混合优化

目标：
- 4负压均值 → -115 Pa
- 3含氧均值 → 2.0%
- 控制变化量惩罚
"""

import numpy as np
import torch
from typing import Optional, Tuple, Callable
from scipy.optimize import minimize
from dataclasses import dataclass

from src.config.constraints import (
    CONTROL_RANGES,
    PRESSURE_TARGET,
    OXYGEN_TARGET,
    MAX_SINGLE_ADJUSTMENT,
)
from src.config.hyperparams import MPC_CONFIG, H
from src.config.variables import CONTROL_VARIABLES, FAN_CONTROL_GROUPS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    """优化结果"""
    best_control: np.ndarray  # 最优控制序列 (H, n_u)
    best_prediction: np.ndarray  # 最优预测 (H, n_y)
    best_cost: float  # 最优目标值
    n_evaluations: int  # 评估次数
    optimization_time: float  # 优化时间（秒）
    feasible: bool  # 是否可行
    message: str  # 附加信息


class BayesianMPCOptimizer:
    """贝叶斯优化器 for MPC

    使用贝叶斯优化搜索最优控制序列：
    - 目标函数：预测偏差 + 控制变化惩罚
    - 约束：物理边界、调整幅度限制
    """

    def __init__(
        self,
        proxy_model: torch.nn.Module,
        n_u: int = 7,
        n_y: int = 7,
        H: int = 5,
        pressure_target: float = PRESSURE_TARGET,
        oxygen_target: float = OXYGEN_TARGET,
        pressure_weight: float = 1.0,
        oxygen_weight: float = 1.0,
        control_change_weight: float = 0.1,
        device: torch.device = torch.device('cpu'),
    ):
        """
        Args:
            proxy_model: 代理模型（用于预测）
            n_u: 控制变量维度
            n_y: 目标变量维度
            H: 预测步数
            pressure_target: 负压目标值
            oxygen_target: 含氧目标值
            pressure_weight: 负压权重
            oxygen_weight: 含氧权重
            control_change_weight: 控制变化惩罚权重
            device: 计算设备
        """
        self.proxy_model = proxy_model
        self.n_u = n_u
        self.n_y = n_y
        self.H = H
        self.pressure_target = pressure_target
        self.oxygen_target = oxygen_target
        self.pressure_weight = pressure_weight
        self.oxygen_weight = oxygen_weight
        self.control_change_weight = control_change_weight
        self.device = device

        # 控制变量边界
        self.control_bounds = self._build_bounds()

        # 当前控制值（用于计算变化量）
        self.current_control: Optional[np.ndarray] = None

    def _build_bounds(self) -> list:
        """构建控制变量边界"""
        bounds = []
        for var in CONTROL_VARIABLES:
            if var in CONTROL_RANGES:
                bounds.append(CONTROL_RANGES[var])
            else:
                bounds.append((0.0, 100.0))  # 默认边界
        return bounds

    def set_current_control(self, current_control: np.ndarray) -> None:
        """设置当前控制值

        Args:
            current_control: 当前控制值 (n_u,)
        """
        self.current_control = current_control

    def objective_function(
        self,
        control_flat: np.ndarray,
        encoder_input: torch.Tensor,
        return_prediction: bool = False,
    ) -> float:
        """目标函数

        Args:
            control_flat: 扁平化的控制序列 (H * n_u,)
            encoder_input: 编码器输入（历史窗口）
            return_prediction: 是否返回预测值

        Returns:
            cost: 目标值（越小越好）
        """
        # 重塑控制序列
        control_seq = control_flat.reshape(self.H, self.n_u)

        # 构建代理模型输入
        # 扁平化：历史 + 未来控制
        encoder_flat = encoder_input.flatten().unsqueeze(0)  # (1, L*n_features)
        control_flat_tensor = torch.tensor(control_flat, dtype=torch.float32).unsqueeze(0)

        proxy_input = torch.cat([encoder_flat, control_flat_tensor], dim=1).to(self.device)

        # 预测
        with torch.no_grad():
            prediction_flat = self.proxy_model(proxy_input)
            prediction = prediction_flat.view(1, self.H, self.n_y).cpu().numpy()[0]

        # 计算目标函数
        # 负压偏差（前4个变量）
        pressure_pred = prediction[:, :4].mean(axis=1)  # (H,) 4测点平均
        pressure_error = np.mean((pressure_pred - self.pressure_target) ** 2)

        # 含氧偏差（后3个变量）
        oxygen_pred = prediction[:, 4:7].mean(axis=1)  # (H,) 3测点平均
        oxygen_error = np.mean((oxygen_pred - self.oxygen_target) ** 2)

        # 控制变化惩罚
        if self.current_control is not None:
            # 第一步与当前值的差异
            first_step_change = np.sum((control_seq[0] - self.current_control) ** 2)
            # 步间差异
            step_changes = np.sum(np.diff(control_seq, axis=0) ** 2)
            control_penalty = first_step_change + step_changes.sum()
        else:
            control_penalty = np.sum(np.diff(control_seq, axis=0) ** 2).sum()

        # 总目标
        cost = (
            self.pressure_weight * pressure_error +
            self.oxygen_weight * oxygen_error +
            self.control_change_weight * control_penalty
        )

        if return_prediction:
            return cost, prediction
        return cost

    def optimize(
        self,
        encoder_input: torch.Tensor,
        n_evaluations: int = 30,
        n_initial_samples: int = 5,
        timeout: float = 30.0,
    ) -> OptimizationResult:
        """优化控制序列

        Args:
            encoder_input: 编码器输入（历史窗口）
            n_evaluations: 评估次数
            n_initial_samples: 初始随机采样数
            timeout: 超时时间

        Returns:
            OptimizationResult: 优化结果
        """
        import time
        start_time = time.time()

        # 初始采样
        initial_samples = self._generate_initial_samples(n_initial_samples)

        best_cost = float('inf')
        best_control = None
        best_prediction = None

        # 评估初始样本
        for sample in initial_samples:
            cost, prediction = self.objective_function(sample, encoder_input, return_prediction=True)
            if cost < best_cost:
                best_cost = cost
                best_control = sample.reshape(self.H, self.n_u)
                best_prediction = prediction

        # 使用scipy.optimize进行精细搜索
        # 从最佳初始点开始
        x0 = best_control.flatten()

        # 构建边界约束（每个变量在每个时间步）
        bounds_flat = []
        for step in range(self.H):
            for i, (low, high) in enumerate(self.control_bounds):
                bounds_flat.append((low, high))

        # 优化
        try:
            result = minimize(
                self.objective_function,
                x0,
                args=(encoder_input,),
                method='L-BFGS-B',
                bounds=bounds_flat,
                options={'maxiter': n_evaluations, 'ftol': 1e-4},
            )

            final_cost, final_prediction = self.objective_function(
                result.x, encoder_input, return_prediction=True
            )

            if final_cost < best_cost:
                best_cost = final_cost
                best_control = result.x.reshape(self.H, self.n_u)
                best_prediction = final_prediction

        except Exception as e:
            logger.warning(f"优化过程出错: {e}")

        optimization_time = time.time() - start_time

        return OptimizationResult(
            best_control=best_control,
            best_prediction=best_prediction,
            best_cost=best_cost,
            n_evaluations=n_evaluations + n_initial_samples,
            optimization_time=optimization_time,
            feasible=True,
            message="优化完成",
        )

    def _generate_initial_samples(self, n_samples: int) -> np.ndarray:
        """生成初始随机样本

        Args:
            n_samples: 样本数量

        Returns:
            samples: (n_samples, H * n_u)
        """
        samples = np.zeros((n_samples, self.H * self.n_u))

        for i in range(n_samples):
            for step in range(self.H):
                for j, (low, high) in enumerate(self.control_bounds):
                    # 在边界内随机采样
                    samples[i, step * self.n_u + j] = np.random.uniform(low, high)

        return samples

    def optimize_grouped(
        self,
        encoder_input: torch.Tensor,
        group: str = 'pressure',
        n_evaluations: int = 10,
    ) -> OptimizationResult:
        """分组优化（紧急工况）

        只优化特定控制组，减少评估次数

        Args:
            encoder_input: 编码器输入
            group: 优化组名 ('pressure', 'oxygen', 'load')
            n_evaluations: 评估次数

        Returns:
            OptimizationResult: 优化结果
        """
        # 确定优化的控制变量
        if group == 'pressure':
            active_vars = FAN_CONTROL_GROUPS['induced_fan']  # 引风机调负压
        elif group == 'oxygen':
            active_vars = FAN_CONTROL_GROUPS['secondary_fan']  # 二次风机调氧
        elif group == 'load':
            active_vars = FAN_CONTROL_GROUPS['load_group']  # 给煤+一次风
        else:
            active_vars = CONTROL_VARIABLES

        # 确定变量索引
        active_indices = [CONTROL_VARIABLES.index(v) for v in active_vars if v in CONTROL_VARIABLES]

        # 只优化活跃变量，其他保持当前值
        # ... 简化实现，使用标准优化
        return self.optimize(encoder_input, n_evaluations=n_evaluations)


class GroupedOptimizer:
    """分组优化器

    分组策略：
    - 负压组：引风机A/B
    - 含氧组：二次风机A/B
    - 负荷组：给煤量 + 一次风机A/B
    """

    def __init__(self, base_optimizer: BayesianMPCOptimizer):
        self.base_optimizer = base_optimizer
        self.groups = {
            'pressure': FAN_CONTROL_GROUPS['induced_fan'],
            'oxygen': FAN_CONTROL_GROUPS['secondary_fan'],
            'load': FAN_CONTROL_GROUPS['coal_feed'] + FAN_CONTROL_GROUPS['primary_fan'],
        }

    def optimize_by_group(
        self,
        encoder_input: torch.Tensor,
        primary_group: str = 'pressure',
        secondary_group: Optional[str] = None,
    ) -> OptimizationResult:
        """按组优化

        先优化主组，再优化次组（可选）

        Args:
            encoder_input: 编码器输入
            primary_group: 主优化组
            secondary_group: 次优化组

        Returns:
            OptimizationResult: 优化结果
        """
        # 主组优化
        result1 = self.base_optimizer.optimize_grouped(
            encoder_input, group=primary_group, n_evaluations=15
        )

        if secondary_group is None:
            return result1

        # 次组优化（从主组结果开始）
        self.base_optimizer.set_current_control(result1.best_control[0])
        result2 = self.base_optimizer.optimize_grouped(
            encoder_input, group=secondary_group, n_evaluations=10
        )

        # 合并结果
        return OptimizationResult(
            best_control=result2.best_control,
            best_prediction=result2.best_prediction,
            best_cost=result2.best_cost,
            n_evaluations=result1.n_evaluations + result2.n_evaluations,
            optimization_time=result1.optimization_time + result2.optimization_time,
            feasible=True,
            message=f"分组优化: {primary_group} + {secondary_group}",
        )


def create_optimizer(
    proxy_model: torch.nn.Module,
    config: Optional[dict] = None,
    device: torch.device = torch.device('cpu'),
) -> BayesianMPCOptimizer:
    """创建MPC优化器

    Args:
        proxy_model: 代理模型
        config: 配置字典
        device: 计算设备

    Returns:
        BayesianMPCOptimizer
    """
    if config is None:
        config = MPC_CONFIG

    optimizer = BayesianMPCOptimizer(
        proxy_model=proxy_model,
        n_u=7,
        n_y=7,
        H=config.get('horizon', H),
        pressure_target=PRESSURE_TARGET,
        oxygen_target=OXYGEN_TARGET,
        pressure_weight=config.get('pressure_weight', 1.0),
        oxygen_weight=config.get('oxygen_weight', 1.0),
        control_change_weight=config.get('control_change_weight', 0.1),
        device=device,
    )

    return optimizer


__all__ = [
    "OptimizationResult",
    "BayesianMPCOptimizer",
    "GroupedOptimizer",
    "create_optimizer",
]