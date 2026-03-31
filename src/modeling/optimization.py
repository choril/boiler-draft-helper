"""
风机控制参数优化模块 - 高级优化算法实现

基于已训练的 LSTM 多步预测模型，实现风机控制参数的最优配置。

优化算法:
1. 贝叶斯优化 (Optuna TPE) - 高效黑盒优化，适合有限评估预算
2. 多目标优化 (NSGA-II) - 同时优化负压和含氧量，生成 Pareto 前沿
3. 梯度下降优化 - 基于 TensorFlow 自动微分
4. CMA-ES 进化策略 - 连续域全局优化
5. 滚动时域优化 (MPC-like) - 实时动态控制

目标:
1. 负压稳定在理想范围 (-150 ~ -80 Pa)，目标值 -115 Pa
2. 含氧量稳定在理想范围 (1.7% ~ 2.3%)，目标值 2.0%
3. 减少控制参数波动，提高系统稳定性
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings

warnings.filterwarnings("ignore")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.stats import norm

from src.utils.config import CONTROL_PARAMS, PRESSURE_MAIN, OXYGEN_MAIN, EXPERT_RANGES


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
    oxygen_min: float = 1.7
    oxygen_max: float = 2.3

    # 损失函数权重
    pressure_weight: float = 1.0
    oxygen_weight: float = 0.5
    stability_weight: float = 0.2
    smoothness_weight: float = 0.001

    # 控制参数约束
    max_adjustment_ratio: float = 0.3
    min_adjustment: float = 5.0

    # 优化参数
    optimization_horizon: int = 5  # 只考虑前N步预测
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
        pressure_col: str = PRESSURE_MAIN,
        oxygen_col: str = OXYGEN_MAIN,
        window_size: int = 60,
    ):
        self.pressure_col = pressure_col
        self.oxygen_col = oxygen_col
        self.window_size = window_size

    def detect_scenes(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """检测不同运行场景"""
        scenes = {"stable": [], "volatile": [], "transition": []}

        if self.pressure_col not in df.columns or self.oxygen_col not in df.columns:
            return scenes

        pressure = df[self.pressure_col].values
        oxygen = df[self.oxygen_col].values

        # 计算滚动标准差
        pressure_std = pd.Series(pressure).rolling(
            self.window_size, min_periods=1
        ).std().values

        valid_std = pressure_std[~np.isnan(pressure_std)]
        if len(valid_std) == 0:
            return scenes

        std_low = np.percentile(valid_std, 25)
        std_high = np.percentile(valid_std, 75)

        i = 0
        while i < len(pressure) - self.window_size:
            window_std = pressure_std[i + self.window_size - 1]
            if np.isnan(window_std):
                i += 1
                continue

            start_idx, end_idx = i, i + self.window_size
            scene_stats = {
                "start_idx": start_idx,
                "end_idx": end_idx,
                "pressure_mean": float(np.mean(pressure[start_idx:end_idx])),
                "pressure_std": float(np.std(pressure[start_idx:end_idx])),
                "oxygen_mean": float(np.mean(oxygen[start_idx:end_idx])),
                "oxygen_std": float(np.std(oxygen[start_idx:end_idx])),
            }

            # 根据波动程度分类
            if window_std < std_low:
                scene_stats["score"] = float(1.0 - window_std / std_low)
                scene_stats["type"] = "stable"
                scenes["stable"].append(scene_stats)
                i += self.window_size
            elif window_std > std_high:
                scene_stats["score"] = float(window_std / std_high)
                scene_stats["type"] = "volatile"
                scenes["volatile"].append(scene_stats)
                i += self.window_size // 2
            else:
                scene_stats["score"] = 0.5
                scene_stats["type"] = "transition"
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

    PARAM_NAMES_CN = {
        "2LB10CS001": "一次风机A转速",
        "2LB20CS001": "一次风机B转速",
        "2LB30CS901": "二次风机A转速",
        "2LB40CS901": "二次风机B转速",
        "2NC10CS901": "引风机A转速",
        "2NC2CS901": "引风机B转速",
        "D62AX002": "给煤量",
        "APAFCDMD": "一次风机A阀门开度",
        "BPAFCDMD": "一次风机B阀门开度",
        "2LA30A11C01": "二次风机A阀门开度",
        "2LA40A11C01": "二次风机B阀门开度",
        "2NC10A11C01": "引风机A阀开度",
        "2NC20A11C01": "引风机B阀门开度",
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

        # 找出模型特征中存在的控制参数
        self.control_param_names = []
        self.control_indices = []

        for ctrl in CONTROL_PARAMS:
            if ctrl in self.feature_to_idx:
                self.control_param_names.append(ctrl)
                self.control_indices.append(self.feature_to_idx[ctrl])

        print(f"优化器初始化: 找到 {len(self.control_param_names)} 个控制参数")
        if self.control_param_names:
            print(f"  控制参数: {self.control_param_names}")

    def _predict(self, features_orig: np.ndarray) -> np.ndarray:
        """预测 - 输入原始特征，输出原始尺度的预测"""
        features_scaled = self.scaler.transform(features_orig)
        features_batch = features_scaled[np.newaxis, :, :]

        pred_scaled = self.model.model(features_batch, training=False).numpy()
        pred_orig = self.target_scaler.inverse_transform(pred_scaled[0])

        horizon = min(self.config.optimization_horizon, len(pred_orig))
        return pred_orig[:horizon]

    def _predict_batch(self, features_batch: np.ndarray) -> np.ndarray:
        """批量预测"""
        # features_batch: (n_samples, seq_length, n_features)
        features_scaled = np.zeros_like(features_batch)
        for i in range(len(features_batch)):
            features_scaled[i] = self.scaler.transform(features_batch[i])

        pred_scaled = self.model.model(features_scaled, training=False).numpy()
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
                # 使用硬边界或当前值附近的范围
                lower = max(lower_hard, current - self.config.min_adjustment)
                upper = min(upper_hard, current + self.config.min_adjustment)
                # 如果仍然无效，直接使用硬边界
                if lower > upper:
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
            return self.optuna.samplers.TPESampler(
                n_startup_trials=10,
                multivariate=True,
            )
        elif method == "CMA-ES":
            return self.optuna.samplers.CmaEsSampler(
                n_startup_trials=10,
            )
        elif method == "GPSampler":
            # 高斯过程采样器（需要 Optuna >= 3.5）
            try:
                return self.optuna.samplers.GPSampler()
            except AttributeError:
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
# 梯度优化器 (TensorFlow AutoGrad)
# =============================================================================

class GradientOptimizer(BaseOptimizer):
    """梯度优化器 - 利用 TensorFlow 自动微分"""

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
        learning_rate: float = 0.01,
        max_iterations: int = 100,
    ) -> OptimizationResult:
        """梯度下降优化"""
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)
        bounds = self._build_bounds(current_values)

        predictions_before = self._predict(features_orig)
        loss_before = self._compute_loss(current_values, features_orig, current_values)

        # 转换为 TensorFlow 变量
        control_vars = tf.Variable(current_values, dtype=tf.float32)
        features_tf = tf.constant(features_orig, dtype=tf.float32)
        bounds_tf = tf.constant(bounds, dtype=tf.float32)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def compute_loss_tf(control_vals, features):
            # 更新特征
            features_updated = tf.identity(features)
            for i, idx in enumerate(self.control_indices):
                # 使用 tensor_array 或 scatter_update
                indices = tf.stack([tf.constant(len(features) - 1), tf.constant(idx)])
                features_updated = tf.tensor_scatter_nd_update(
                    features_updated,
                    [indices],
                    [control_vals[i]]
                )

            # 标准化
            features_scaled = self.scaler.transform(features_updated.numpy())
            features_batch = tf.constant(features_scaled[np.newaxis, :, :], dtype=tf.float32)

            # 预测
            pred_scaled = self.model.model(features_batch, training=False)
            pred_orig = self.target_scaler.inverse_transform(pred_scaled.numpy()[0])

            horizon = min(self.config.optimization_horizon, len(pred_orig))
            pred = tf.constant(pred_orig[:horizon], dtype=tf.float32)

            # 计算损失
            pressure_pred = pred[:, 0]
            oxygen_pred = pred[:, 1]

            pressure_loss = tf.reduce_mean(
                (pressure_pred - self.config.pressure_target) ** 2
            ) * self.config.pressure_weight

            oxygen_loss = tf.reduce_mean(
                (oxygen_pred - self.config.oxygen_target) ** 2
            ) * self.config.oxygen_weight

            stability_loss = (
                tf.math.reduce_variance(pressure_pred) +
                tf.math.reduce_variance(oxygen_pred)
            ) * self.config.stability_weight

            smoothness_loss = tf.reduce_sum(
                (control_vals - tf.constant(current_values, dtype=tf.float32)) ** 2
            ) * self.config.smoothness_weight

            return pressure_loss + oxygen_loss + stability_loss + smoothness_loss

        # 优化循环
        n_evaluations = 0
        best_loss = float('inf')
        best_controls = current_values.copy()

        for iteration in range(max_iterations):
            with tf.GradientTape() as tape:
                loss = compute_loss_tf(control_vars, features_tf)

            gradients = tape.gradient(loss, control_vars)
            optimizer.apply_gradients([(gradients, control_vars)])

            # 应用边界约束
            control_clipped = tf.clip_by_value(
                control_vars,
                tf.constant([b[0] for b in bounds], dtype=tf.float32),
                tf.constant([b[1] for b in bounds], dtype=tf.float32),
            )
            control_vars.assign(control_clipped)

            n_evaluations += 1
            current_loss = float(loss.numpy())

            if current_loss < best_loss:
                best_loss = current_loss
                best_controls = control_vars.numpy().copy()

            # 早停
            if iteration > 10 and current_loss < self.config.tolerance:
                break

        optimal_values = best_controls

        # 计算优化后预测
        features_optimized = features_orig.copy()
        for i, idx in enumerate(self.control_indices):
            features_optimized[-1, idx] = optimal_values[i]

        predictions_after = self._predict(features_optimized)
        loss_after = best_loss

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
            method="Gradient-Adam",
            converged=True,
            n_evaluations=n_evaluations,
            elapsed_time=elapsed_time,
            predictions_before=predictions_before,
            predictions_after=predictions_after,
        )


# =============================================================================
# 混合优化器 (自动选择最优算法)
# =============================================================================

class HybridOptimizer(BaseOptimizer):
    """混合优化器 - 自动选择最优算法组合"""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        super().__init__(model, scaler, target_scaler, feature_names, config)

        # 初始化子优化器
        self.bayesian_optimizer = BayesianOptimizer(
            model, scaler, target_scaler, feature_names, config
        )
        self.gradient_optimizer = GradientOptimizer(
            model, scaler, target_scaler, feature_names, config
        )

    def optimize(
        self,
        features_orig: np.ndarray,
        strategy: str = "auto",
    ) -> OptimizationResult:
        """混合优化 - 自动选择或组合算法"""
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)

        if strategy == "auto":
            # 根据控制参数数量自动选择策略
            n_controls = len(self.control_param_names)
            if n_controls <= 3:
                strategy = "bayesian"
            elif n_controls <= 7:
                strategy = "hybrid"
            else:
                strategy = "gradient"

        results = []

        if strategy in ["bayesian", "hybrid"]:
            try:
                result_bayes = self.bayesian_optimizer.optimize(features_orig, "TPE")
                results.append(result_bayes)
            except Exception as e:
                print(f"贝叶斯优化失败: {e}")

        if strategy in ["scipy", "hybrid"]:
            result_scipy = self._scipy_optimize(features_orig)
            results.append(result_scipy)

        if strategy in ["gradient", "hybrid"]:
            try:
                result_grad = self.gradient_optimizer.optimize(features_orig)
                results.append(result_grad)
            except Exception as e:
                print(f"梯度优化失败: {e}")

        # 选择最优结果
        if not results:
            # 使用当前值作为默认结果
            return self._create_default_result(features_orig, current_values)

        best_result = min(results, key=lambda x: x.loss_after)

        # 标记策略
        best_result.method = f"Hybrid-{strategy}"
        best_result.elapsed_time = time.time() - start_time

        return best_result

    def _scipy_optimize(self, features_orig: np.ndarray) -> OptimizationResult:
        """Scipy L-BFGS-B 优化"""
        start_time = time.time()

        current_values = self.get_current_control_values(features_orig)
        bounds = self._build_scipy_bounds(current_values)

        predictions_before = self._predict(features_orig)
        loss_before = self._compute_loss(current_values, features_orig, current_values)

        best_result = None
        best_loss = float('inf')

        for restart in range(self.config.n_restarts):
            x0 = current_values if restart == 0 else np.array([
                np.random.uniform(bounds.lb[i], bounds.ub[i])
                for i in range(len(current_values))
            ])

            try:
                result = minimize(
                    self._compute_loss,
                    x0,
                    args=(features_orig, current_values),
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": self.config.max_iterations,
                        "ftol": self.config.tolerance,
                    },
                )

                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result
            except Exception:
                continue

        if best_result is None:
            best_result = type('obj', (object,), {
                'x': current_values, 'fun': loss_before, 'success': False
            })()

        optimal_values = best_result.x

        features_optimized = features_orig.copy()
        for i, idx in enumerate(self.control_indices):
            features_optimized[-1, idx] = optimal_values[i]

        predictions_after = self._predict(features_optimized)
        loss_after = best_result.fun

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
            method="L-BFGS-B",
            converged=bool(best_result.success),
            n_evaluations=best_result.nfev if hasattr(best_result, 'nfev') else 0,
            elapsed_time=elapsed_time,
            predictions_before=predictions_before,
            predictions_after=predictions_after,
        )

    def _create_default_result(
        self,
        features_orig: np.ndarray,
        current_values: np.ndarray,
    ) -> OptimizationResult:
        """创建默认结果"""
        predictions = self._predict(features_orig)
        loss = self._compute_loss(current_values, features_orig, current_values)

        return OptimizationResult(
            optimal_values={ctrl: float(current_values[i]) for i, ctrl in enumerate(self.control_param_names)},
            adjustments={ctrl: 0.0 for ctrl in self.control_param_names},
            predicted_pressure=predictions[:, 0],
            predicted_oxygen=predictions[:, 1],
            loss_before=float(loss),
            loss_after=float(loss),
            improvement_ratio=0.0,
            method="Default",
            converged=False,
            predictions_before=predictions,
            predictions_after=predictions,
        )


# =============================================================================
# 滚动时域优化器 (MPC-like)
# =============================================================================

class RollingHorizonOptimizer(BaseOptimizer):
    """滚动时域优化器 - 类 MPC 实时控制"""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        target_scaler: Any,
        feature_names: List[str],
        config: Optional[OptimizationConfig] = None,
    ):
        super().__init__(model, scaler, target_scaler, feature_names, config)
        self.hybrid_optimizer = None  # 按需初始化

    def optimize(
        self,
        features_orig: np.ndarray,
        n_steps: int = 3,
    ) -> List[OptimizationResult]:
        """滚动优化 - 逐步调整控制参数"""
        if self.hybrid_optimizer is None:
            self.hybrid_optimizer = HybridOptimizer(
                self.model, self.scaler, self.target_scaler,
                self.feature_names, self.config
            )

        results = []
        features_current = features_orig.copy()

        for step in range(n_steps):
            result = self.hybrid_optimizer.optimize(features_current, "hybrid")
            result.method = f"MPC-Step{step+1}"
            results.append(result)

            # 更新特征（应用最优控制）
            for ctrl, value in result.optimal_values.items():
                if ctrl in self.feature_to_idx:
                    features_current[-1, self.feature_to_idx[ctrl]] = value

        return results

    def get_final_recommendation(
        self,
        rolling_results: List[OptimizationResult],
    ) -> OptimizationResult:
        """从滚动优化结果中提取最终推荐"""
        if not rolling_results:
            raise ValueError("滚动优化结果为空")

        # 取最后一步的结果
        final = rolling_results[-1]

        # 计算累计改善
        total_improvement = 0.0
        for r in rolling_results:
            total_improvement += r.improvement_ratio

        final.improvement_ratio = min(total_improvement, 1.0)
        final.method = "MPC-Final"

        return final


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
    elif method == "gradient":
        return GradientOptimizer(model, scaler, target_scaler, feature_names, config)
    elif method == "multi-objective" or method == "NSGA-II":
        return MultiObjectiveOptimizer(model, scaler, target_scaler, feature_names, config)
    elif method == "hybrid":
        return HybridOptimizer(model, scaler, target_scaler, feature_names, config)
    elif method == "mpc":
        return RollingHorizonOptimizer(model, scaler, target_scaler, feature_names, config)
    else:
        return HybridOptimizer(model, scaler, target_scaler, feature_names, config)


__all__ = [
    "OptimizationConfig",
    "OptimizationResult",
    "MultiObjectiveResult",
    "SceneDetector",
    "BaseOptimizer",
    "BayesianOptimizer",
    "MultiObjectiveOptimizer",
    "GradientOptimizer",
    "HybridOptimizer",
    "RollingHorizonOptimizer",
    "ControlRecommender",
    "create_optimizer",
]