"""
MPC控制器 - 模型预测控制核心

核心流程：
1. 状态更新：采集当前运行数据
2. 滚动优化：搜索最优控制序列
3. 约束检查：验证控制安全性
4. 执行控制：只执行第一步
5. 滚动到下一时刻

特点：
- 滚动优化（Receding Horizon）
- 只执行第一步（One-Step Execution）
- 安全约束嵌入
"""

import numpy as np
import torch
from typing import Optional, Tuple
from dataclasses import dataclass
import time

from src.mpc.optimizer import BayesianMPCOptimizer, OptimizationResult
from src.mpc.safety_monitor import SafetyMonitor, SafetyLevel, SafetyCheckResult
from src.config.hyperparams import MPC_CONFIG, H, L
from src.config.constraints import PRESSURE_TARGET, OXYGEN_TARGET
from src.config.variables import CONTROL_VARIABLES, TARGET_VARIABLES
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MPCStepResult:
    """MPC单步执行结果"""
    executed_control: np.ndarray      # 执行的控制值 (n_u,)
    full_control_sequence: np.ndarray # 完整控制序列 (H, n_u)
    prediction: np.ndarray            # 预测值 (H, n_y)
    cost: float                       # 目标值
    safety_result: SafetyCheckResult  # 安全检查结果
    optimization_time: float          # 优化时间（秒）
    n_evaluations: int                # 评估次数
    message: str                      # 附加信息


class MPCController:
    """MPC控制器

    完整的模型预测控制流程
    """

    def __init__(
        self,
        proxy_model: torch.nn.Module,
        optimizer: Optional[BayesianMPCOptimizer] = None,
        safety_monitor: Optional[SafetyMonitor] = None,
        H: int = H,
        L: int = L,
        n_y: int = 7,
        n_u: int = 7,
        device: torch.device = torch.device('cpu'),
        config: Optional[dict] = None,
    ):
        """
        Args:
            proxy_model: 代理预测模型
            optimizer: MPC优化器（可选，自动创建）
            safety_monitor: 安全监控器（可选，自动创建）
            H: 预测步长
            L: 历史窗口长度
            n_y: 目标变量维度
            n_u: 控制变量维度
            device: 计算设备
            config: MPC配置
        """
        self.proxy_model = proxy_model.to(device)
        self.device = device
        self.H = H
        self.L = L
        self.n_y = n_y
        self.n_u = n_u
        self.config = config or MPC_CONFIG

        # 创建优化器
        if optimizer is None:
            from src.mpc.optimizer import create_optimizer
            self.optimizer = create_optimizer(proxy_model, self.config, device)
        else:
            self.optimizer = optimizer

        # 创建安全监控器
        if safety_monitor is None:
            from src.mpc.safety_monitor import create_safety_monitor
            self.safety_monitor = SafetyMonitor()
        else:
            self.safety_monitor = safety_monitor

        # 当前状态
        self.current_control: Optional[np.ndarray] = None
        self.history_window: Optional[np.ndarray] = None  # (L, n_features)

        # 目标值
        self.pressure_target = PRESSURE_TARGET
        self.oxygen_target = OXYGEN_TARGET

        # 执行历史
        self.execution_history: list[MPCStepResult] = []

    def update_state(
        self,
        current_y: np.ndarray,
        current_u: np.ndarray,
        current_x: np.ndarray,
        history_y: Optional[np.ndarray] = None,
        history_u: Optional[np.ndarray] = None,
        history_x: Optional[np.ndarray] = None,
    ) -> None:
        """更新当前状态

        Args:
            current_y: 当前目标值 (n_y,)
            current_u: 当前控制值 (n_u,)
            current_x: 当前状态值 (n_x,)
            history_y: 历史目标值 (L, n_y)（可选）
            history_u: 历史控制值 (L, n_u)（可选）
            history_x: 历史状态值 (L, n_x)（可选）
        """
        # 更新当前控制
        self.current_control = current_u.copy()

        # 构建历史窗口
        if history_y is not None and history_u is not None and history_x is not None:
            # 使用提供的完整历史
            self.history_window = np.concatenate([history_y, history_u, history_x], axis=1)
        else:
            # 使用当前值构建简单历史（实际应用中应从数据系统获取）
            if self.history_window is None:
                # 初始化历史窗口（用当前值填充）
                current_features = np.concatenate([current_y, current_u, current_x])
                self.history_window = np.tile(current_features, (self.L, 1))
            else:
                # 滑动窗口：移除最早时刻，添加当前时刻
                current_features = np.concatenate([current_y, current_u, current_x])
                self.history_window = np.vstack([
                    self.history_window[1:],  # 移除最早
                    current_features           # 添加当前
                ])

        logger.info(f"状态更新: 当前控制 = {current_u[:3]}... (前3个)")

    def compute_control(
        self,
        emergency_mode: bool = False,
        n_evaluations: Optional[int] = None,
    ) -> MPCStepResult:
        """计算最优控制

        Args:
            emergency_mode: 是否紧急模式（减少评估次数）
            n_evaluations: 自定义评估次数

        Returns:
            MPCStepResult: 控制计算结果
        """
        start_time = time.time()

        # 检查历史窗口是否存在
        if self.history_window is None:
            raise ValueError("历史窗口未初始化，请先调用 update_state()")

        # 转换为张量
        encoder_input = torch.tensor(self.history_window, dtype=torch.float32)

        # 设置当前控制（用于变化量计算）
        if self.current_control is not None:
            self.optimizer.set_current_control(self.current_control)

        # 优化参数
        if emergency_mode:
            n_eval = self.config.get('emergency_evaluations', 10)
            group = 'pressure'  # 紧急模式优先调负压
        else:
            n_eval = n_evaluations or self.config.get('n_evaluations', 30)

        # 执行优化
        if emergency_mode:
            opt_result = self.optimizer.optimize_grouped(encoder_input, group=group, n_evaluations=n_eval)
        else:
            opt_result = self.optimizer.optimize(encoder_input, n_evaluations=n_eval)

        # 安全检查（控制序列）
        safety_result = self.safety_monitor.check_control_sequence(
            opt_result.best_control,
            self.current_control
        )

        # 安全检查（预测值）
        prediction_safety = self.safety_monitor.check_prediction(opt_result.best_prediction)

        # 合并安全检查结果
        combined_issues = safety_result.issues + prediction_safety.issues
        if combined_issues:
            safety_result.issues = combined_issues
            safety_result.level = max(safety_result.level, prediction_safety.level)

        # 确定执行的控制
        if safety_result.level == SafetyLevel.ALARM:
            # 严重告警时，使用保守控制（保持当前或小幅调整）
            if self.current_control is not None:
                executed_control = self._conservative_control(self.current_control)
            else:
                executed_control = opt_result.best_control[0]
            message = "安全告警，执行保守控制"
        else:
            # 正常执行第一步
            executed_control = opt_result.best_control[0]
            message = "正常执行"

        optimization_time = time.time() - start_time

        result = MPCStepResult(
            executed_control=executed_control,
            full_control_sequence=opt_result.best_control,
            prediction=opt_result.best_prediction,
            cost=opt_result.best_cost,
            safety_result=safety_result,
            optimization_time=optimization_time,
            n_evaluations=opt_result.n_evaluations,
            message=message,
        )

        # 记录执行历史
        self.execution_history.append(result)

        logger.info(f"MPC控制计算完成:")
        logger.info(f"  优化时间: {optimization_time:.2f}s")
        logger.info(f"  评估次数: {opt_result.n_evaluations}")
        logger.info(f"  目标值: {opt_result.best_cost:.4f}")
        logger.info(f"  安全等级: {safety_result.level.name}")

        return result

    def _conservative_control(self, current: np.ndarray) -> np.ndarray:
        """计算保守控制（紧急情况）

        小幅调整以稳定系统

        Args:
            current: 当前控制值

        Returns:
            conservative: 保守控制值
        """
        conservative = current.copy()

        # 小幅调整引风机（降低负压风险）
        # 增加引风机频率（更负的负压）
        induced_fan_indices = [CONTROL_VARIABLES.index(v) for v in ['DPU61AX107', 'DPU61AX108'] if v in CONTROL_VARIABLES]
        for idx in induced_fan_indices:
            conservative[idx] = min(current[idx] + 2, 50)  # 小幅增加，不超过上限

        # 小幅增加二次风机（增加含氧）
        secondary_fan_indices = [CONTROL_VARIABLES.index(v) for v in ['2LA30A12C11', '2LA40A12C11'] if v in CONTROL_VARIABLES]
        for idx in secondary_fan_indices:
            conservative[idx] = min(current[idx] + 1, 50)

        return conservative

    def run_step(
        self,
        current_y: np.ndarray,
        current_u: np.ndarray,
        current_x: np.ndarray,
        history_y: Optional[np.ndarray] = None,
        history_u: Optional[np.ndarray] = None,
        history_x: Optional[np.ndarray] = None,
    ) -> MPCStepResult:
        """执行完整的MPC单步流程

        Args:
            current_y: 当前目标值
            current_u: 当前控制值
            current_x: 当前状态值
            history_y: 历史目标值（可选）
            history_u: 历史控制值（可选）
            history_x: 历史状态值（可选）

        Returns:
            MPCStepResult: 执行结果
        """
        # 1. 状态更新
        self.update_state(current_y, current_u, current_x, history_y, history_u, history_x)

        # 2. 检查当前安全状态
        safety_check = self.safety_monitor.check_current_state(
            current_y[:4],  # 负压
            current_y[4:7], # 含氧
            None            # 床温（可选）
        )

        # 3. 确定优化模式
        emergency_mode = safety_check.level == SafetyLevel.ALARM

        # 4. 计算控制
        result = self.compute_control(emergency_mode=emergency_mode)

        return result

    def get_execution_summary(self) -> dict:
        """获取执行历史摘要

        Returns:
            summary: 执行摘要字典
        """
        if not self.execution_history:
            return {'n_steps': 0}

        # 统计
        n_steps = len(self.execution_history)
        avg_cost = np.mean([r.cost for r in self.execution_history])
        avg_time = np.mean([r.optimization_time for r in self.execution_history])
        alarm_count = sum(1 for r in self.execution_history if r.safety_result.level == SafetyLevel.ALARM)

        return {
            'n_steps': n_steps,
            'avg_cost': avg_cost,
            'avg_optimization_time': avg_time,
            'alarm_count': alarm_count,
            'alarm_ratio': alarm_count / n_steps if n_steps > 0 else 0,
        }


def create_mpc_controller(
    proxy_model: torch.nn.Module,
    device: torch.device = torch.device('cpu'),
    config: Optional[dict] = None,
) -> MPCController:
    """创建MPC控制器

    Args:
        proxy_model: 代理模型
        device: 计算设备
        config: MPC配置

    Returns:
        MPCController
    """
    return MPCController(
        proxy_model=proxy_model,
        device=device,
        config=config,
    )


__all__ = [
    "MPCStepResult",
    "MPCController",
    "create_mpc_controller",
]