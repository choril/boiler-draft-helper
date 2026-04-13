"""
风机控制参数优化模块

基于已训练的 LSTM 多步预测模型，实现风机控制参数的最优配置。

优化算法:
1. 贝叶斯优化 (Optuna TPE) - 高效黑盒优化，适合有限评估预算
2. 多目标优化 (NSGA-II) - 同时优化负压和含氧量，生成 Pareto 前沿
3. 梯度下降优化 - 基于 TensorFlow 自动微分
4. CMA-ES 进化策略 - 连续域全局优化
5. 滚动时域优化 (MPC-like) - 实时动态控制

目标:
1. 负压稳定在理想范围 (-150 ~ -80 Pa)，目标值 -115 Pa
2. 含氧量稳定在理想范围 (1.5% ~ 2.5%)，目标值 2.0%
3. 减少控制参数波动，提高系统稳定性
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from scipy.optimize import minimize, Bounds

from src.utils.config import CONTROL_PARAMS, PRESSURE_VARIABLES, OXYGEN_VARIABLES, EXPERT_RANGES
from src.utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# 配置和结果数据类
# =============================================================================

@dataclass
class OptimizationConfig:
    """优化配置参数"""

    # 目标范围
    pressure_target: float = -115.0
    pressure_min: float = -150.0
    pressure_max: float = -80.0
    oxygen_target: float = 2.0
    oxygen_min: float = 1.5
    oxygen_max: float = 2.5

    # 损失函数权重
    pressure_weight: float = 1.0
    oxygen_weight: float = 0.5
    stability_weight: float = 0.2
    smoothness_weight: float = 0.001

    # 控制参数约束
    max_adjustment_ratio: float = 0.3
    min_adjustment: float = 5.0

    # 优化参数
    optimization_horizon: int = 10  # 只考虑前N步预测
    max_iterations: int = 200
    tolerance: float = 1e-6
    n_restarts: int = 5

    # 贝叶斯优化参数
    n_bayesian_trials: int = 50
    bayesian_sampler: str = "TPETSampler"  # TPE 或 CMA-ES

    # 多目标优化参数
    n_pop_size: int = 50  # NSGA-II 种群大小
    n_generations: int = 30  # NSGA-II 代数

    # CMA-ES 参数
    cmaes_max_evals: int = 1000

    # 滚动优化参数
    mpc_horizon: int = 10
    mpc_control_horizon: int = 3

    # 控制参数物理边界
    """TODO: 待补充其他参数物理边界"""
    control_bounds: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "2LB10CS001": (200, 980),      # 一次风机A转速
        "2LB20CS001": (200, 980),      # 一次风机B转速
        "2LB30CS901": (200, 980),      # 二次风机A转速
        "2LB40CS901": (200, 980),      # 二次风机B转速
        "2NC10CS901": (200, 980),      # 引风机A转速
        "2NC2CS901": (200, 980),       # 引风机B转速
        "D62AX002": (30, 100),         # 给煤量
        "APAFCDMD": (0, 100),          # 一次风机A阀门开度
        "BPAFCDMD": (0, 100),          # 一次风机B阀门开度
        "2LA30A11C01": (0, 100),       # 二次风机A阀门开度
        "2LA40A11C01": (0, 100),       # 二次风机B阀门开度
        "2NC10A11C01": (0, 100),       # 引风机A阀开度
        "2NC20A11C01": (0, 100),       # 引风机B阀门开度
    })


@dataclass
class OptimizationResult:
    """优化结果"""
    optimal_values: Dict[str, float]
    adjustments: Dict[str, float]
    predicted_pressure: np.ndarray
    predicted_oxygen: np.ndarray
    loss_before: float
    loss_after: float
    improvement_ratio: float
    method: str
    converged: bool
    n_evaluations: int = 0
    elapsed_time: float = 0.0
    predictions_before: Optional[np.ndarray] = None
    predictions_after: Optional[np.ndarray] = None
    pareto_front: Optional[List[Dict]] = None  # 多目标优化的 Pareto 前沿


@dataclass
class MultiObjectiveResult:
    """多目标优化结果"""
    pareto_front: List[Dict]  # Pareto 前沿解集
    optimal_solution: Dict    # 推荐最优解
    objectives_tradeoff: Dict # 目标权衡分析


# =============================================================================
# 场景检测器
# =============================================================================

class SceneDetector:
    """场景检测器 - 识别不同运行状态"""

    PARAM_NAMES_CN = {
        "pressure_mean": "负压均值",
        "pressure_std": "负压波动",
        "oxygen_mean": "含氧量均值",
        "oxygen_std": "含氧量波动",
        "stable": "稳定工况",
        "volatile": "波动工况",
        "transition": "过渡工况",
    }

    def __init__(
        self,
        pressure_col: str = PRESSURE_VARIABLES[0],
        oxygen_col: str = OXYGEN_VARIABLES[0],
        window_size: int = 60,
    ):
        self.pressure_col = pressure_col
        self.oxygen_col = oxygen_col
        self.window_size = window_size

    def detect_scenes(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """检测不同运行场景 - 综合负压和含氧量波动"""
        scenes = {"stable": [], "volatile": [], "transition": []}

        if self.pressure_col not in df.columns or self.oxygen_col not in df.columns:
            logger.warning("数据中缺少压力或含氧量列，无法进行场景检测！")
            return scenes

        pressure = df[self.pressure_col].values
        oxygen = df[self.oxygen_col].values

        # 计算滚动标准差
        pressure_std = pd.Series(pressure).rolling(
            self.window_size, min_periods=self.window_size
        ).std().values
        oxygen_std = pd.Series(oxygen).rolling(
            self.window_size, min_periods=self.window_size
        ).std().values

        valid_p_std = pressure_std[~np.isnan(pressure_std)]
        valid_o_std = oxygen_std[~np.isnan(oxygen_std)]

        if len(valid_p_std) == 0 or len(valid_o_std) == 0:
            logger.warning("没有有效的波动数据，无法进行场景检测！")
            return scenes

        # 计算分位数阈值
        p_std_low = np.percentile(valid_p_std, 25)
        p_std_high = np.percentile(valid_p_std, 75)
        o_std_low = np.percentile(valid_o_std, 25)
        o_std_high = np.percentile(valid_o_std, 75)

        i = 0
        while i < len(pressure) - self.window_size:
            # 检查窗口末尾的标准差是否有效
            idx_end = i + self.window_size - 1
            p_window_std = pressure_std[idx_end]
            o_window_std = oxygen_std[idx_end]

            if np.isnan(p_window_std) or np.isnan(o_window_std):
                i += 1
                continue

            start_idx, end_idx = i, i + self.window_size
            scene_stats = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "pressure_mean": float(np.mean(pressure[start_idx:end_idx])),
                "oxygen_mean": float(np.mean(oxygen[start_idx:end_idx])),
                "pressure_window_std": float(p_window_std),
                "oxygen_window_std": float(o_window_std),
                "pressure_std": float(np.std(pressure[start_idx:end_idx], ddof=1)),
                "oxygen_std": float(np.std(oxygen[start_idx:end_idx], ddof=1)),
            }

            # 综合判断：基于相对波动程度（归一化后加权）
            p_volatility = 0.0
            o_volatility = 0.0
            if p_std_high > p_std_low:
                p_volatility = np.clip((p_window_std - p_std_low) / (p_std_high - p_std_low), 0, 1)
            if o_std_high > o_std_low:
                o_volatility = np.clip((o_window_std - o_std_low) / (o_std_high - o_std_low), 0, 1)
            # 综合波动指数（负压权重 0.7，含氧量权重 0.3）
            combined_volatility = 0.7 * p_volatility + 0.3 * o_volatility

            if combined_volatility < 0.25:  # 低波动区间
                scene_stats["score"] = float(1.0 - combined_volatility)
                scene_stats["type"] = "stable"
                scene_stats["combined_volatility"] = float(combined_volatility)
                scenes["stable"].append(scene_stats)
                i += self.window_size
            elif combined_volatility > 0.75:  # 高波动区间
                scene_stats["score"] = float(combined_volatility)
                scene_stats["type"] = "volatile"
                scene_stats["combined_volatility"] = float(combined_volatility)
                scenes["volatile"].append(scene_stats)
                i += self.window_size // 2  # 波动场景密集采样
            else:  # 过渡区间
                scene_stats["score"] = float(combined_volatility)
                scene_stats["type"] = "transition"
                scene_stats["combined_volatility"] = float(combined_volatility)
                scenes["transition"].append(scene_stats)
                i += self.window_size

        return scenes

    def get_top_scenes(
        self,
        df: pd.DataFrame,
        scene_type: str = "volatile",
        top_k: int = 5,
    ) -> List[Dict]:
        """获取前K个特定类型的场景"""
        scenes = self.detect_scenes(df)
        if scene_type not in scenes or not scenes[scene_type]:
            logger.warning(f"没有检测到 {scene_type} 场景！")
            return []
        return sorted(scenes[scene_type], key=lambda x: x["score"], reverse=True)[:top_k]

    def analyze_current_state(
        self,
        df: pd.DataFrame,
        window: int = 60,
    ) -> Dict:
        """分析当前运行状态"""
        if len(df) < window:
            return {"status": "unknown"}

        recent = df.iloc[-window:]

        pressure = recent[self.pressure_col].values
        oxygen = recent[self.oxygen_col].values

        pressure_mean = np.mean(pressure)
        pressure_std = np.std(pressure)
        oxygen_mean = np.mean(oxygen)
        oxygen_std = np.std(oxygen)

        # 判断状态
        ideal_p_min, ideal_p_max = EXPERT_RANGES["pressure_ideal"]
        ideal_o_min, ideal_o_max = EXPERT_RANGES["oxygen_ideal"]

        p_in_range = (pressure_mean >= ideal_p_min) and (pressure_mean <= ideal_p_max)
        o_in_range = (oxygen_mean >= ideal_o_min) and (oxygen_mean <= ideal_o_max)

        if p_in_range and o_in_range and pressure_std < 10:
            status = "optimal"
        elif pressure_std > 20 or oxygen_std > 0.3:
            status = "volatile"
        elif not p_in_range or not o_in_range:
            status = "deviation"
        else:
            status = "normal"

        return {
            "status": status,
            "pressure_mean": pressure_mean,
            "pressure_std": pressure_std,
            "oxygen_mean": oxygen_mean,
            "oxygen_std": oxygen_std,
            "pressure_in_range": p_in_range,
            "oxygen_in_range": o_in_range,
        }


# =============================================================================
# 基础优化器
# =============================================================================

class BaseOptimizer:
    """基础优化器 - 提供通用预测和损失计算接口"""

    # 控制参数名称映射
    PARAM_NAMES_CN = {
        # 给煤量
        "D62AX002": "给煤量",
        # 二次风机A（调氧量）
        "2LB30CS901": "二次风机A转速",
        "2BBA13Q11": "二次风机A电流",
        "2LA30A12C11": "二次风机A输出频率",
        "2HLA30CP01": "二次风机A出口压力",
        "2LA30A11C01": "二次风机A阀门开度",
        # 二次风机B
        "2LB40CS901": "二次风机B转速",
        "2BBB11Q11": "二次风机B电流",
        "2LA40A12C11": "二次风机B输出频率",
        "2LA40CP01": "二次风机B出口压力",
        "2LA40A11C01": "二次风机B阀门开度",
        # 引风机A（调负压）
        "2NC10CS901": "引风机A转速",
        "2BBA15Q11": "引风机A电流",
        "DPU61AX107": "引风机A输出频率",
        "2NA10CP004": "引风机A入口压力",
        "2NC10A11C01": "引风机A阀门开度",
        # 引风机B
        "2NC2CS901": "引风机B转速",
        "2BBB13Q11": "引风机B电流",
        "DPU61AX108": "引风机B输出频率",
        "2NA2CP004": "引风机B入口压力",
        "2NC20A11C01": "引风机B阀门开度",
        # 一次风机A（快速提升负荷）
        "2LB10CS001": "一次风机A转速",
        "2BBA14Q11": "一次风机A电流",
        "2LA10A12C11": "一次风机A输出频率",
        "2LA10CP01": "一次风机A出口压力",
        "APAFCDMD": "一次风机A阀门开度",
        # 一次风机B
        "2LB20CS001": "一次风机B转速",
        "2BBB12Q11": "一次风机B电流",
        "2LA20A12C11": "一次风机B输出频率",
        "2HLA2CP001": "一次风机B出口压力",
        "BPAFCDMD": "一次风机B阀门开度",
    }

    # 监测参数名称映射
    MONITOR_NAMES_CN = {
        "MSFLOW": "主蒸汽流量",
        "D66P53A10": "床温",
        "D61AX023": "一次风风量",
        "D61AX024": "二次风风量",
        "2LA10CT11": "出风温1",
        "2LA2CT11": "出风温2",
        "2BK10CP004": "炉膛压力",
        "2BK10CQ1": "含氧量",
    }

    # 目标变量名称映射
    TARGET_NAMES_CN = {
        "2BK10CP004": "炉膛压力1",
        "2BK2CP004": "炉膛压力2",
        "2BK10CP005": "炉膛压力3",
        "2BK2CP005": "炉膛压力4",
        "2BK10CQ1": "含氧量1",
        "2BK2CQ1": "含氧量2",
        "2BK2CQ2": "含氧量3",
    }

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        self.model = model
        self.scaler = scaler
        self.target_scaler = target_scaler
        self.feature_names = list(feature_names)
        self.config = config or OptimizationConfig()

        # 建立特征索引映射
        self.feature_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        self.control_param_names = []
        self.control_indices = []
        for ctrl in CONTROL_PARAMS:
            if ctrl in self.feature_to_idx:
                self.control_param_names.append(ctrl)
                self.control_indices.append(self.feature_to_idx[ctrl])
        logger.info(f"优化器初始化: 找到 {len(self.control_param_names)} 个控制参数")
        if self.control_param_names:
            logger.info(f"控制参数: {self.control_param_names} : {[self.PARAM_NAMES_CN.get(ctrl, ctrl) for ctrl in self.control_param_names]}")

    def _predict(self, features_orig: np.ndarray) -> np.ndarray:
        """预测 - 输入原始特征，输出原始尺度的预测"""
        features_scaled = self.scaler.transform(features_orig)
        features_batch = features_scaled[np.newaxis, :, :]

        # 兼容 LSTM 类和 keras 模型
        if hasattr(self.model, 'model'):
            pred_scaled = self.model.model(features_batch, training=False).numpy()
        else:
            pred_scaled = self.model(features_batch, training=False).numpy()

        pred_orig = self.target_scaler.inverse_transform(pred_scaled[0])

        horizon = min(self.config.optimization_horizon, len(pred_orig))
        return pred_orig[:horizon]

    def _predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """批量预测"""
        # features_batch: (n_samples, seq_length, n_features)
        features_scaled = np.zeros_like(features_batch)
        for i in range(len(features_batch)):
            features_scaled[i] = self.scaler.transform(features_batch[i])

        # 兼容 LSTM 类和 keras 模型
        if hasattr(self.model, 'model'):
            pred_scaled = self.model.model(features_scaled, training=False).numpy()
        else:
            pred_scaled = self.model(features_scaled, training=False).numpy()

        pred_orig = np.zeros((len(features_batch), pred_scaled.shape[1], 2))
        for i in range(len(features_batch)):
            pred_orig[i] = self.target_scaler.inverse_transform(pred_scaled[i])

        horizon = min(self.config.optimization_horizon, pred_orig.shape[1])
        return pred_orig[:, :horizon, :]

    def _compute_loss(
        self,
        control_values: np.ndarray,
        base_features: np.ndarray,
        current_values: np.ndarray,
    ) -> float:
        """计算损失函数"""
        features = base_features.copy()

        # 更新控制参数（修改最后一个时间步）
        for i, idx in enumerate(self.control_indices):
            features[-1, idx] = control_values[i]

        predictions = self._predict(features)
        pressure_pred = predictions[:, 0]
        oxygen_pred = predictions[:, 1]

        # 负压损失（MSE到目标值）
        pressure_loss = np.mean(
            (pressure_pred - self.config.pressure_target) ** 2
        ) * self.config.pressure_weight

        # 含氧量损失
        oxygen_loss = np.mean(
            (oxygen_pred - self.config.oxygen_target) ** 2
        ) * self.config.oxygen_weight

        # 稳定性损失（预测方差）
        stability_loss = (
            np.var(pressure_pred) + np.var(oxygen_pred)
        ) * self.config.stability_weight

        # 平滑性损失（控制参数变化）
        smoothness_loss = np.sum(
            (control_values - current_values) ** 2
        ) * self.config.smoothness_weight

        return pressure_loss + oxygen_loss + stability_loss + smoothness_loss

    def _compute_multi_objective(
        self,
        control_values: np.ndarray,
        base_features: np.ndarray,
    ) -> Tuple[float, float]:
        """计算多目标函数值（用于 NSGA-II）"""
        features = base_features.copy()

        for i, idx in enumerate(self.control_indices):
            features[-1, idx] = control_values[i]

        predictions = self._predict(features)
        pressure_pred = predictions[:, 0]
        oxygen_pred = predictions[:, 1]

        # 目标1: 负压 MSE
        obj1 = np.mean((pressure_pred - self.config.pressure_target) ** 2)

        # 目标2: 含氧量 MSE
        obj2 = np.mean((oxygen_pred - self.config.oxygen_target) ** 2)

        return obj1, obj2

    def _build_bounds(self, current_values: np.ndarray) -> List[Tuple[float, float]]:
        """构建优化边界 - 基于当前值和最大调整比例"""
        bounds = []
        for i, ctrl in enumerate(self.control_param_names):
            current = current_values[i]

            # 硬物理边界
            lower_hard, upper_hard = self.config.control_bounds.get(ctrl, (0, 1000))

            # 计算允许的调整范围
            max_adj = max(abs(current) * self.config.max_adjustment_ratio, self.config.min_adjustment)

            # 软约束：基于当前值的变化范围
            lower_soft = current - max_adj
            upper_soft = current + max_adj

            # 应用硬边界约束
            lower = max(lower_hard, lower_soft)
            upper = min(upper_hard, upper_soft)

            # 确保 lower <= upper（当当前值超出硬边界时）
            if lower > upper:
                if current < lower_hard:
                    # 当前值低于下限：下界强制为硬下限，上界给一个小步长让其回归
                    lower = lower_hard
                    upper = min(upper_hard, lower_hard + self.config.min_adjustment)
                elif current > upper_hard:
                    # 当前值高于上限：上界强制为硬上限，下界给一个小步长让其回归
                    upper = upper_hard
                    lower = max(lower_hard, upper_hard - self.config.min_adjustment)
                else:
                    # 理论上不会走到这里（在界内时 lower <= upper 必然成立）
                    # 除非 min_adjustment 大于整个硬边界区间，此时退化为硬边界
                    lower = lower_hard
                    upper = upper_hard

            bounds.append((float(lower), float(upper)))

        return bounds

    def _build_scipy_bounds(self, current_values: np.ndarray) -> Bounds:
        """构建 scipy Bounds 对象"""
        bounds_list = self._build_bounds(current_values)
        lower = np.array([b[0] for b in bounds_list])
        upper = np.array([b[1] for b in bounds_list])
        return Bounds(lower, upper)

    def get_current_control_values(self, features_orig: np.ndarray) -> np.ndarray:
        """获取当前控制参数值"""
        return np.array([
            features_orig[-1, idx] for idx in self.control_indices
        ])


# =============================================================================
# 贝叶斯优化器 (Optuna-based)
# =============================================================================

class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器 - 使用 Optuna TPE/CMA-ES 采样器"""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        super().__init__(model, scaler, target_scaler, feature_names, config)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna = optuna
        except ImportError:
            raise ImportError("请安装 optuna: pip install optuna")

    def optimize(
        self,
        features_orig: np.ndarray,
        method: str = "TPE",
    ) -> OptimizationResult:
        """贝叶斯优化"""
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)
        bounds = self._build_bounds(current_values)

        # 计算初始损失
        loss_before = self._compute_loss(current_values, features_orig, current_values)
        predictions_before = self._predict(features_orig)

        # 创建 Optuna Study
        sampler = self._create_sampler(method)
        study = self.optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        n_evaluations = [0]  # 使用列表以便在闭包中修改

        def objective(trial):
            nonlocal n_evaluations
            # 采样控制参数
            control_values = []
            for i, (ctrl, bound) in enumerate(zip(self.control_param_names, bounds)):
                value = trial.suggest_float(ctrl, bound[0], bound[1])
                control_values.append(value)

            control_values = np.array(control_values)
            n_evaluations[0] += 1

            return self._compute_loss(control_values, features_orig, current_values)

        # 运行优化
        study.optimize(
            objective,
            n_trials=self.config.n_bayesian_trials,
            show_progress_bar=False,
        )

        # 获取最优结果
        best_trial = study.best_trial
        optimal_values = np.array([
            best_trial.params[ctrl] for ctrl in self.control_param_names
        ])

        # 计算优化后的预测
        features_optimized = features_orig.copy()
        for i, idx in enumerate(self.control_indices):
            features_optimized[-1, idx] = optimal_values[i]

        predictions_after = self._predict(features_optimized)
        loss_after = best_trial.value

        elapsed_time = time.time() - start_time
        improvement = max(0, (loss_before - loss_after) / (loss_before + 1e-8))

        return OptimizationResult(
            optimal_values={ctrl: float(optimal_values[i]) for i, ctrl in enumerate(self.control_param_names)},
            adjustments={ctrl: float(optimal_values[i] - current_values[i]) for i, ctrl in enumerate(self.control_param_names)},
            predicted_pressure=predictions_after[:, 0],
            predicted_oxygen=predictions_after[:, 1],
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            improvement_ratio=float(improvement),
            method=f"Bayesian-{method}",
            converged=True,
            n_evaluations=n_evaluations[0],
            elapsed_time=elapsed_time,
            predictions_before=predictions_before,
            predictions_after=predictions_after,
        )

    def _create_sampler(self, method: str):
        """创建 Optuna 采样器"""
        if method == "TPE":
            return self.optuna.samplers.TPESampler(multivariate=True)
        elif method == "CMA-ES":
            return self.optuna.samplers.CmaEsSampler(n_startup_trials=10)
        elif method == "GPSampler":
            # 高斯过程采样器
            try:
                return self.optuna.samplers.GPSampler()
            except AttributeError:
                logger.warning("当前 Optuna 版本不支持 GPSampler，使用 TPE 替代")
                return self.optuna.samplers.TPESampler()
        else:
            return self.optuna.samplers.TPESampler()


# =============================================================================
# 多目标优化器 (NSGA-II)
# =============================================================================

class MultiObjectiveOptimizer(BaseOptimizer):
    """多目标优化器 - 使用 NSGA-II 算法生成 Pareto 前沿"""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        super().__init__(model, scaler, target_scaler, feature_names, config)

    def optimize(
        self,
        features_orig: np.ndarray,
        return_pareto: bool = True,
    ) -> OptimizationResult:
        """多目标优化 - 返回 Pareto 前沿"""
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)
        bounds = self._build_bounds(current_values)

        # 初始预测
        predictions_before = self._predict(features_orig)
        loss_before = self._compute_loss(current_values, features_orig, current_values)

        # NSGA-II 优化
        pareto_front = self._run_nsga2(features_orig, bounds)

        # 从 Pareto 前沿选择最优解（根据加权偏好）
        optimal_solution = self._select_from_pareto(pareto_front)

        optimal_values = np.array([
            optimal_solution["controls"][ctrl]
            for ctrl in self.control_param_names
        ])

        # 计算优化后预测
        features_optimized = features_orig.copy()
        for i, idx in enumerate(self.control_indices):
            features_optimized[-1, idx] = optimal_values[i]

        predictions_after = self._predict(features_optimized)
        loss_after = optimal_solution["weighted_loss"]

        elapsed_time = time.time() - start_time
        improvement = max(0, (loss_before - loss_after) / (loss_before + 1e-8))

        return OptimizationResult(
            optimal_values={ctrl: float(optimal_values[i]) for i, ctrl in enumerate(self.control_param_names)},
            adjustments={ctrl: float(optimal_values[i] - current_values[i]) for i, ctrl in enumerate(self.control_param_names)},
            predicted_pressure=predictions_after[:, 0],
            predicted_oxygen=predictions_after[:, 1],
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            improvement_ratio=float(improvement),
            method="NSGA-II",
            converged=True,
            n_evaluations=self.config.n_pop_size * self.config.n_generations,
            elapsed_time=elapsed_time,
            predictions_before=predictions_before,
            predictions_after=predictions_after,
            pareto_front=pareto_front if return_pareto else None,
        )

    def _run_nsga2(
        self,
        features_orig: np.ndarray,
        bounds: List[Tuple[float, float]],
    ) -> List[Dict]:
        """运行简化版 NSGA-II"""
        n_pop = self.config.n_pop_size
        n_gen = self.config.n_generations
        n_controls = len(self.control_param_names)

        # 初始化种群
        population = []
        for _ in range(n_pop):
            individual = np.array([
                np.random.uniform(b[0], b[1]) for b in bounds
            ])
            obj1, obj2 = self._compute_multi_objective(individual, features_orig)
            population.append({
                "controls": {ctrl: individual[i] for i, ctrl in enumerate(self.control_param_names)},
                "obj1": obj1,
                "obj2": obj2,
            })

        # 运行进化
        for gen in range(n_gen):
            # 选择（锦标赛）
            new_population = []
            for _ in range(n_pop):
                # 随机选择两个个体
                idx1, idx2 = np.random.choice(len(population), 2, replace=False)
                p1, p2 = population[idx1], population[idx2]

                # Pareto 比较
                if self._dominates(p1, p2):
                    winner = p1
                elif self._dominates(p2, p1):
                    winner = p2
                else:
                    winner = p1 if np.random.random() < 0.5 else p2

                new_population.append(winner.copy())

            # 变异
            for i in range(n_pop):
                for j, ctrl in enumerate(self.control_param_names):
                    if np.random.random() < 0.3:  # 变异概率
                        current = new_population[i]["controls"][ctrl]
                        bound = bounds[j]
                        # 高斯变异
                        mutation = np.random.normal(0, (bound[1] - bound[0]) * 0.1)
                        new_val = np.clip(current + mutation, bound[0], bound[1])
                        new_population[i]["controls"][ctrl] = new_val

                # 重新计算目标值
                control_array = np.array([
                    new_population[i]["controls"][ctrl] for ctrl in self.control_param_names
                ])
                obj1, obj2 = self._compute_multi_objective(control_array, features_orig)
                new_population[i]["obj1"] = obj1
                new_population[i]["obj2"] = obj2

            population = new_population

        # 提取 Pareto 前沿
        pareto_front = []
        for p in population:
            dominated = False
            for q in population:
                if self._dominates(q, p):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(p)

        # 计算加权损失
        for p in pareto_front:
            p["weighted_loss"] = (
                p["obj1"] * self.config.pressure_weight +
                p["obj2"] * self.config.oxygen_weight
            )

        return pareto_front

    def _dominates(self, p1: Dict, p2: Dict) -> bool:
        """Pareto 占优判断"""
        return (p1["obj1"] <= p2["obj1"] and p1["obj2"] <= p2["obj2"]) and \
               (p1["obj1"] < p2["obj1"] or p1["obj2"] < p2["obj2"])

    def _select_from_pareto(self, pareto_front: List[Dict]) -> Dict:
        """从 Pareto 前沿选择最优解"""
        if not pareto_front:
            raise ValueError("Pareto 前沿为空")

        # 根据加权损失选择
        return min(pareto_front, key=lambda x: x["weighted_loss"])

# =============================================================================
# 控制推荐器
# =============================================================================

class ControlRecommender:
    """控制建议生成器 - 将优化结果转换为可执行建议"""

    PARAM_NAMES_CN = BaseOptimizer.PARAM_NAMES_CN

    def __init__(
        self,
        optimizer: BaseOptimizer,
        config: Optional[OptimizationConfig] = None,
    ):
        self.optimizer = optimizer
        self.config = config or OptimizationConfig()

    def generate_recommendation(
        self,
        result: OptimizationResult,
        safety_margin: float = 0.8,
    ) -> Dict:
        """生成控制建议"""
        recommendations = []

        for param, opt_value in result.optimal_values.items():
            adj = result.adjustments.get(param, 0)
            safe_adj = adj * safety_margin
            safe_value = opt_value - adj + safe_adj

            if abs(safe_adj) > 1.0:
                name_cn = self.PARAM_NAMES_CN.get(param, param)
                recommendations.append({
                    "parameter": param,
                    "parameter_cn": name_cn,
                    "current_value": round(opt_value - adj, 2),
                    "recommended_value": round(safe_value, 2),
                    "adjustment": round(safe_adj, 2),
                    "direction": "增加" if safe_adj > 0 else "减少",
                    "significance": "高" if abs(safe_adj) > 20 else "中" if abs(safe_adj) > 5 else "低",
                    "physical_bound": self.config.control_bounds.get(param, (None, None)),
                })

        # 预期效果
        expected = {
            "pressure_target": self.config.pressure_target,
            "pressure_predicted_mean": float(np.mean(result.predicted_pressure)),
            "pressure_predicted_range": (
                float(np.min(result.predicted_pressure)),
                float(np.max(result.predicted_pressure)),
            ),
            "oxygen_target": self.config.oxygen_target,
            "oxygen_predicted_mean": float(np.mean(result.predicted_oxygen)),
            "oxygen_predicted_range": (
                float(np.min(result.predicted_oxygen)),
                float(np.max(result.predicted_oxygen)),
            ),
        }

        # 状态评估
        p_in_range = all(
            self.config.pressure_min <= p <= self.config.pressure_max
            for p in result.predicted_pressure
        )
        o_in_range = all(
            self.config.oxygen_min <= o <= self.config.oxygen_max
            for o in result.predicted_oxygen
        )

        status = {
            "pressure_in_ideal_range": p_in_range,
            "oxygen_in_ideal_range": o_in_range,
            "improvement_ratio": result.improvement_ratio,
            "overall_quality": self._assess_quality(p_in_range, o_in_range, result.improvement_ratio),
        }

        return {
            "recommendations": recommendations,
            "expected_effects": expected,
            "status": status,
            "optimization_method": result.method,
            "elapsed_time": result.elapsed_time,
        }

    def _assess_quality(
        self,
        p_in_range: bool,
        o_in_range: bool,
        improvement: float,
    ) -> str:
        """评估优化质量"""
        if p_in_range and o_in_range:
            return "优秀"
        elif improvement > 0.2:
            return "良好"
        elif improvement > 0.1:
            return "一般"
        else:
            return "需改进"

    def format_report(self, recommendation: Dict) -> str:
        """格式化报告"""
        lines = [
            "=" * 70,
            "风机控制参数优化建议报告",
            "=" * 70,
            "",
            f"优化方法: {recommendation['optimization_method']}",
            f"计算耗时: {recommendation['elapsed_time']:.3f}s",
            "",
            "【控制参数调整建议】",
            "-" * 60,
        ]

        for rec in recommendation["recommendations"]:
            bound = rec.get("physical_bound", (None, None))
            bound_str = f" [{bound[0]:.1f}, {bound[1]:.1f}]" if bound[0] is not None else ""

            lines.extend([
                f"\n{rec['parameter_cn']} ({rec['parameter']}){bound_str}:",
                f"  当前值: {rec['current_value']:.2f}",
                f"  建议值: {rec['recommended_value']:.2f}",
                f"  调整量: {rec['adjustment']:+.2f} ({rec['direction']})",
                f"  重要度: {rec['significance']}",
            ])

        effects = recommendation["expected_effects"]
        status = recommendation["status"]

        lines.extend([
            "",
            "【预期控制效果】",
            "-" * 60,
            f"负压目标: {effects['pressure_target']:.1f} Pa",
            f"  预测均值: {effects['pressure_predicted_mean']:.2f} Pa",
            f"  预测范围: [{effects['pressure_predicted_range'][0]:.2f}, {effects['pressure_predicted_range'][1]:.2f}] Pa",
            "",
            f"含氧量目标: {effects['oxygen_target']:.1f} %",
            f"  预测均值: {effects['oxygen_predicted_mean']:.2f} %",
            f"  预测范围: [{effects['oxygen_predicted_range'][0]:.2f}, {effects['oxygen_predicted_range'][1]:.2f}] %",
            "",
            "【状态评估】",
            "-" * 60,
            f"  负压在理想范围: {'是' if status['pressure_in_ideal_range'] else '否'}",
            f"  含氧量在理想范围: {'是' if status['oxygen_in_ideal_range'] else '否'}",
            f"  改善比例: {status['improvement_ratio']:.1%}",
            f"  整体评价: {status['overall_quality']}",
            "",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# 分组贝叶斯优化器 (Hierarchical Optimization)
# =============================================================================

class HierarchicalBayesianOptimizer(BaseOptimizer):
    """
    分组贝叶斯优化器 - 基于专家经验的分层优化策略

    核心思想:
    1. 氧含量调节 → 仅优化二次风机参数 (二次风机A/B转速和阀门)
    2. 负压调节 → 仅优化引风机参数 (引风机A/B转速和阀门)

    优势:
    - 降低参数维度 (13维 → 4维/4维)，提高优化效率
    - 符合实际操作逻辑，便于执行
    - 减少参数耦合干扰
    """

    # 控制参数分组定义
    CONTROL_GROUPS = {
        "oxygen": {
            "description": "氧含量调节组 - 二次风机参数",
            "params": [
                "2LB30CS901",   # 二次风机A转速
                "2LA30A11C01",  # 二次风机A阀门开度
                "2LB40CS901",   # 二次风机B转速
                "2LA40A11C01",  # 二次风机B阀门开度
            ],
            "target": "oxygen",
            "target_value": 2.0,
            "weight": 0.5,
        },
        "pressure": {
            "description": "负压调节组 - 引风机参数",
            "params": [
                "2NC10CS901",   # 引风机A转速
                "2NC10A11C01",  # 引风机A阀开度
                "2NC2CS901",    # 引风机B转速
                "2NC20A11C01",  # 引风机B阀门开度
            ],
            "target": "pressure",
            "target_value": -115.0,
            "weight": 1.0,
        },
        "load": {
            "description": "负荷调节组 - 一次风机和给煤量",
            "params": [
                "2LB10CS001",   # 一次风机A转速
                "APAFCDMD",     # 一次风机A阀门开度
                "2LB20CS001",   # 一次风机B转速
                "BPAFCDMD",     # 一次风机B阀门开度
                "D62AX002",     # 给煤量
            ],
            "target": "both",
            "target_value": None,
            "weight": 0.3,
        },
    }

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        super().__init__(model, scaler, target_scaler, feature_names, config)

        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna = optuna
        except ImportError:
            raise ImportError("请安装 optuna: pip install optuna")

        # 构建参数组索引映射
        self._build_group_indices()

    def _build_group_indices(self):
        """构建参数组的特征索引映射"""
        self.group_indices = {}

        for group_name, group_info in self.CONTROL_GROUPS.items():
            indices = []
            params_in_features = []
            for param in group_info["params"]:
                if param in self.feature_to_idx:
                    indices.append(self.feature_to_idx[param])
                    params_in_features.append(param)

            if indices:
                self.group_indices[group_name] = {
                    "indices": indices,
                    "params": params_in_features,
                    "info": group_info,
                }

        logger.info(f"分组优化器初始化: 建立了 {len(self.group_indices)} 个参数组")
        for name, info in self.group_indices.items():
            logger.info(f"  {name}组: {len(info['params'])} 个参数 - {info['params']}")

    def optimize(
        self,
        features_orig: np.ndarray,
        strategy: str = "sequential",
        n_trials_per_group: int = 30,
    ) -> OptimizationResult:
        """
        分组优化

        Args:
            features_orig: 原始特征数组
            strategy: 优化策略
                - "sequential": 序贯优化，先氧量后负压
                - "parallel": 并行优化两组参数
                - "pressure_first": 先负压后氧量
            n_trials_per_group: 每组参数的优化试验次数

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)
        predictions_before = self._predict(features_orig)
        loss_before = self._compute_loss(current_values, features_orig, current_values)

        # 根据策略确定优化顺序
        if strategy == "pressure_first":
            order = ["pressure", "oxygen"]
        elif strategy == "parallel":
            order = ["pressure", "oxygen"]  # 并行执行
        else:  # sequential
            order = ["oxygen", "pressure"]

        # 执行分组优化
        features_current = features_orig.copy()
        group_results = {}

        if strategy == "parallel":
            # 并行优化两组
            group_results = self._parallel_group_optimize(
                features_orig, n_trials_per_group
            )
            # 合并结果
            features_optimized = features_orig.copy()
            for group_name, result in group_results.items():
                for param, value in result["optimal_values"].items():
                    if param in self.feature_to_idx:
                        features_optimized[-1, self.feature_to_idx[param]] = value
        else:
            # 序贯优化
            for group_name in order:
                if group_name not in self.group_indices:
                    continue

                result = self._optimize_single_group(
                    features_current, group_name, n_trials_per_group
                )
                group_results[group_name] = result

                # 更新特征用于下一组优化
                for param, value in result["optimal_values"].items():
                    if param in self.feature_to_idx:
                        features_current[-1, self.feature_to_idx[param]] = value

            features_optimized = features_current

        # 计算最终结果
        predictions_after = self._predict(features_optimized)
        optimal_values = {}
        adjustments = {}

        for group_name, result in group_results.items():
            for param, value in result["optimal_values"].items():
                optimal_values[param] = value
                adjustments[param] = result["adjustments"].get(param, 0.0)

        # 补充未优化的参数（保持原值）
        for ctrl in self.control_param_names:
            if ctrl not in optimal_values:
                optimal_values[ctrl] = float(features_orig[-1, self.feature_to_idx.get(ctrl, 0)])
                adjustments[ctrl] = 0.0

        loss_after = self._compute_loss(
            np.array([optimal_values.get(ctrl, 0) for ctrl in self.control_param_names]),
            features_orig, current_values
        )

        elapsed_time = time.time() - start_time
        improvement = max(0, (loss_before - loss_after) / (loss_before + 1e-8))

        return OptimizationResult(
            optimal_values=optimal_values,
            adjustments=adjustments,
            predicted_pressure=predictions_after[:, 0],
            predicted_oxygen=predictions_after[:, 1],
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            improvement_ratio=float(improvement),
            method=f"Hierarchical-{strategy}",
            converged=True,
            n_evaluations=n_trials_per_group * len(group_results),
            elapsed_time=elapsed_time,
            predictions_before=predictions_before,
            predictions_after=predictions_after,
        )

    def _optimize_single_group(
        self,
        features_orig: np.ndarray,
        group_name: str,
        n_trials: int,
    ) -> Dict:
        """优化单个参数组"""
        group_info = self.group_indices[group_name]
        indices = group_info["indices"]
        params = group_info["params"]

        # 获取当前值和边界
        current_values = np.array([features_orig[-1, idx] for idx in indices])
        bounds = self._build_group_bounds(group_name, current_values)

        # 创建优化器
        sampler = self.optuna.samplers.TPESampler(multivariate=True)
        study = self.optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        # 目标函数：针对该组的特定目标
        target_type = group_info["info"]["target"]

        def objective(trial):
            # 采样该组参数
            control_values = []
            for i, (param, bound) in enumerate(zip(params, bounds)):
                value = trial.suggest_float(param, bound[0], bound[1])
                control_values.append(value)

            # 更新特征
            features = features_orig.copy()
            for i, idx in enumerate(indices):
                features[-1, idx] = control_values[i]

            # 预测
            predictions = self._predict(features)
            pressure_pred = predictions[:, 0]
            oxygen_pred = predictions[:, 1]

            # 根据组目标计算损失
            if target_type == "oxygen":
                # 氧含量优化组
                loss = np.mean((oxygen_pred - self.config.oxygen_target) ** 2)
                # 加入稳定性约束
                loss += np.var(oxygen_pred) * self.config.stability_weight
                # 加入平滑性约束
                loss += np.sum((np.array(control_values) - current_values) ** 2) * self.config.smoothness_weight
            elif target_type == "pressure":
                # 负压优化组
                loss = np.mean((pressure_pred - self.config.pressure_target) ** 2)
                # 加入稳定性约束
                loss += np.var(pressure_pred) * self.config.stability_weight
                # 加入平滑性约束
                loss += np.sum((np.array(control_values) - current_values) ** 2) * self.config.smoothness_weight
            else:
                # 综合目标
                loss = self._compute_group_loss(
                    np.array(control_values), features_orig, current_values
                )

            return loss

        # 运行优化
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # 提取最优结果
        best_trial = study.best_trial
        optimal_values = {param: best_trial.params[param] for param in params}
        adjustments = {param: best_trial.params[param] - features_orig[-1, indices[i]]
                      for i, param in enumerate(params)}

        return {
            "optimal_values": optimal_values,
            "adjustments": adjustments,
            "loss": best_trial.value,
            "n_trials": n_trials,
        }

    def _parallel_group_optimize(
        self,
        features_orig: np.ndarray,
        n_trials: int,
    ) -> Dict:
        """并行优化多个参数组"""
        group_results = {}

        # 对每个组独立优化（基于原始特征）
        for group_name in ["pressure", "oxygen"]:
            if group_name not in self.group_indices:
                continue

            result = self._optimize_single_group(features_orig, group_name, n_trials)
            group_results[group_name] = result

        return group_results

    def _build_group_bounds(
        self,
        group_name: str,
        current_values: np.ndarray,
    ) -> List[Tuple[float, float]]:
        """构建参数组的优化边界"""
        bounds = []
        group_info = self.group_indices[group_name]

        for i, param in enumerate(group_info["params"]):
            current = current_values[i]
            # 硬物理边界
            lower_hard, upper_hard = self.config.control_bounds.get(param, (0, 1000))

            # 软约束
            max_adj = max(abs(current) * self.config.max_adjustment_ratio,
                         self.config.min_adjustment)

            lower_soft = current - max_adj
            upper_soft = current + max_adj

            lower = max(lower_hard, lower_soft)
            upper = min(upper_hard, upper_soft)

            if lower > upper:
                lower = max(lower_hard, current - self.config.min_adjustment)
                upper = min(upper_hard, current + self.config.min_adjustment)
                if lower > upper:
                    lower = lower_hard
                    upper = upper_hard

            bounds.append((float(lower), float(upper)))

        return bounds

    def _compute_group_loss(
        self,
        control_values: np.ndarray,
        base_features: np.ndarray,
        current_values: np.ndarray,
    ) -> float:
        """计算综合损失"""
        features = base_features.copy()

        # 更新控制参数
        for i, idx in enumerate(self.control_indices):
            features[-1, idx] = control_values[i]

        predictions = self._predict(features)
        pressure_pred = predictions[:, 0]
        oxygen_pred = predictions[:, 1]

        # 综合损失
        pressure_loss = np.mean(
            (pressure_pred - self.config.pressure_target) ** 2
        ) * self.config.pressure_weight

        oxygen_loss = np.mean(
            (oxygen_pred - self.config.oxygen_target) ** 2
        ) * self.config.oxygen_weight

        stability_loss = (
            np.var(pressure_pred) + np.var(oxygen_pred)
        ) * self.config.stability_weight

        smoothness_loss = np.sum(
            (control_values - current_values) ** 2
        ) * self.config.smoothness_weight

        return pressure_loss + oxygen_loss + stability_loss + smoothness_loss

    def get_group_info(self) -> Dict:
        """获取参数分组信息"""
        return {
            "defined_groups": self.CONTROL_GROUPS,
            "available_groups": self.group_indices,
            "feature_names": self.feature_names,
        }


# =============================================================================
# 两阶段混合优化器 (Hierarchical + Bayesian)
# =============================================================================

class HybridTwoStageOptimizer(BaseOptimizer):
    """
    两阶段混合优化器 - 结合分组优化和全参数优化的优势

    策略:
    1. 第一阶段（粗调）：分组优化快速找到大致优化方向
    2. 第二阶段（精调）：基于粗调结果进行全参数精细调整

    优势:
    - 粗调阶段：低维搜索，快速收敛到好的区域
    - 精调阶段：捕捉参数耦合效应，精细优化
    - 兼顾效率和效果
    """

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        super().__init__(model, scaler, target_scaler, feature_names, config)

        # 导入 optuna
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            self.optuna = optuna
        except ImportError:
            raise ImportError("请安装 optuna: pip install optuna")

        # 初始化子优化器
        self.hierarchical_optimizer = HierarchicalBayesianOptimizer(
            model, scaler, target_scaler, feature_names, config
        )
        self.bayesian_optimizer = BayesianOptimizer(
            model, scaler, target_scaler, feature_names, config
        )

    def optimize(
        self,
        features_orig: np.ndarray,
        coarse_trials: int = 20,
        fine_trials: int = 30,
        coarse_strategy: str = "sequential",
    ) -> OptimizationResult:
        """
        两阶段混合优化

        Args:
            features_orig: 原始特征数组
            coarse_trials: 粗调阶段每组试验次数
            fine_trials: 精调阶段试验次数
            coarse_strategy: 粗调阶段策略 (sequential/parallel/pressure_first)

        Returns:
            OptimizationResult: 优化结果
        """
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)
        predictions_before = self._predict(features_orig)
        loss_before = self._compute_loss(current_values, features_orig, current_values)

        logger.info("=" * 50)
        logger.info("两阶段混合优化开始")
        logger.info("=" * 50)

        # ================== 第一阶段：粗调 ==================
        logger.info(f"【第一阶段-粗调】分组优化 (每组{coarse_trials}次试验)")
        coarse_result = self.hierarchical_optimizer.optimize(
            features_orig,
            strategy=coarse_strategy,
            n_trials_per_group=coarse_trials,
        )

        logger.info(f"  粗调损失改善: {coarse_result.improvement_ratio:.2%}")
        logger.info(f"  粗调耗时: {coarse_result.elapsed_time:.2f}s")

        # 基于粗调结果更新特征
        features_coarse = features_orig.copy()
        for param, value in coarse_result.optimal_values.items():
            if param in self.feature_to_idx:
                features_coarse[-1, self.feature_to_idx[param]] = value

        # ================== 第二阶段：精调 ==================
        logger.info(f"【第二阶段-精调】全参数优化 ({fine_trials}次试验)")

        # 获取粗调后的控制参数值作为精调的初始点
        coarse_values = np.array([
            coarse_result.optimal_values.get(ctrl, features_orig[-1, self.feature_to_idx.get(ctrl, 0)])
            for ctrl in self.control_param_names
        ])

        # 构建精调的边界（基于粗调结果，缩小搜索范围）
        fine_bounds = self._build_fine_bounds(coarse_values, shrink_ratio=0.5)

        # 运行精调优化
        fine_result = self._run_fine_optimization(
            features_coarse, fine_bounds, fine_trials, coarse_values
        )

        logger.info(f"  精调损失改善: {fine_result['improvement']:.2%}")
        logger.info(f"  精调耗时: {fine_result['elapsed_time']:.2f}s")

        # ================== 合并结果 ==================
        optimal_values = fine_result["optimal_values"]
        adjustments = {
            ctrl: optimal_values[ctrl] - current_values[i]
            for i, ctrl in enumerate(self.control_param_names)
        }

        # 计算最终预测
        features_optimized = features_orig.copy()
        for i, idx in enumerate(self.control_indices):
            features_optimized[-1, idx] = optimal_values[self.control_param_names[i]]

        predictions_after = self._predict(features_optimized)
        loss_after = fine_result["loss"]

        elapsed_time = time.time() - start_time
        improvement = max(0, (loss_before - loss_after) / (loss_before + 1e-8))

        logger.info("=" * 50)
        logger.info(f"两阶段混合优化完成")
        logger.info(f"  总损失改善: {improvement:.2%}")
        logger.info(f"  总耗时: {elapsed_time:.2f}s")
        logger.info("=" * 50)

        return OptimizationResult(
            optimal_values=optimal_values,
            adjustments=adjustments,
            predicted_pressure=predictions_after[:, 0],
            predicted_oxygen=predictions_after[:, 1],
            loss_before=float(loss_before),
            loss_after=float(loss_after),
            improvement_ratio=float(improvement),
            method=f"Hybrid-TwoStage",
            converged=True,
            n_evaluations=coarse_result.n_evaluations + fine_result["n_evaluations"],
            elapsed_time=elapsed_time,
            predictions_before=predictions_before,
            predictions_after=predictions_after,
        )

    def _build_fine_bounds(
        self,
        coarse_values: np.ndarray,
        shrink_ratio: float = 0.5,
    ) -> List[Tuple[float, float]]:
        """
        构建精调阶段的边界

        基于粗调结果，缩小搜索范围以提高精调效率
        """
        bounds = []
        for i, ctrl in enumerate(self.control_param_names):
            coarse_val = coarse_values[i]

            # 硬物理边界
            lower_hard, upper_hard = self.config.control_bounds.get(ctrl, (0, 1000))

            # 基于粗调结果缩小的范围
            # 使用原始允许调整范围的 shrink_ratio
            max_adj = max(abs(coarse_val) * self.config.max_adjustment_ratio,
                         self.config.min_adjustment)
            fine_range = max_adj * shrink_ratio

            lower = max(lower_hard, coarse_val - fine_range)
            upper = min(upper_hard, coarse_val + fine_range)

            # 确保边界有效
            if lower > upper:
                lower = lower_hard
                upper = upper_hard

            bounds.append((float(lower), float(upper)))

        return bounds

    def _run_fine_optimization(
        self,
        features_coarse: np.ndarray,
        bounds: List[Tuple[float, float]],
        n_trials: int,
        coarse_values: np.ndarray,
    ) -> Dict:
        """运行精调阶段的全参数优化"""
        start_time = time.time()

        # 创建优化器
        sampler = self.optuna.samplers.TPESampler(multivariate=True)
        study = self.optuna.create_study(
            direction="minimize",
            sampler=sampler,
        )

        # 计算粗调后的损失作为基准
        loss_coarse = self._compute_loss(coarse_values, features_coarse, coarse_values)

        def objective(trial):
            # 采样控制参数
            control_values = []
            for i, (ctrl, bound) in enumerate(zip(self.control_param_names, bounds)):
                value = trial.suggest_float(ctrl, bound[0], bound[1])
                control_values.append(value)

            return self._compute_loss(
                np.array(control_values), features_coarse, coarse_values
            )

        # 运行优化
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # 提取最优结果
        best_trial = study.best_trial
        optimal_values = {
            ctrl: best_trial.params[ctrl]
            for ctrl in self.control_param_names
        }
        loss_after = best_trial.value

        elapsed_time = time.time() - start_time
        improvement = max(0, (loss_coarse - loss_after) / (loss_coarse + 1e-8))

        return {
            "optimal_values": optimal_values,
            "loss": loss_after,
            "improvement": improvement,
            "n_evaluations": n_trials,
            "elapsed_time": elapsed_time,
        }

    def get_optimization_info(self) -> Dict:
        """获取优化器信息"""
        return {
            "type": "Two-Stage Hybrid",
            "stage1": {
                "name": "Hierarchical Optimization",
                "purpose": "快速粗调，定位优化区域",
                "groups": self.hierarchical_optimizer.CONTROL_GROUPS,
            },
            "stage2": {
                "name": "Bayesian Optimization",
                "purpose": "精细调整，捕捉参数耦合",
            },
        }


# =============================================================================
# 工厂函数
# =============================================================================

def create_optimizer(
    method: str,
    model: Any,
    scaler: Any,
    target_scaler: Any,
    feature_names: List[str],
    config: Optional[OptimizationConfig] = None,
) -> BaseOptimizer:
    """创建优化器"""
    config = config or OptimizationConfig()

    if method == "bayesian" or method == "TPE":
        return BayesianOptimizer(model, scaler, target_scaler, feature_names, config)
    elif method == "hierarchical" or method == "grouped":
        return HierarchicalBayesianOptimizer(model, scaler, target_scaler, feature_names, config)
    elif method == "hybrid_two_stage" or method == "two_stage":
        return HybridTwoStageOptimizer(model, scaler, target_scaler, feature_names, config)
    elif method == "multi-objective" or method == "NSGA-II":
        return MultiObjectiveOptimizer(model, scaler, target_scaler, feature_names, config)
    else:
        raise ValueError(f"未知的优化方法: {method}")


__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "MultiObjectiveResult",
    "SceneDetector",
    "BaseOptimizer",
    "BayesianOptimizer",
    "HierarchicalBayesianOptimizer",
    "HybridTwoStageOptimizer",
    "MultiObjectiveOptimizer",
    "ControlRecommender",
    "create_optimizer",
]