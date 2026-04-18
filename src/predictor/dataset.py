"""
数据处理模块

功能：
1. 加载清洗后的数据
2. 特征提取（可选，调用 src/features/extractor.py）
3. 特征选择（可选，调用 src/features/selector.py）
4. 动态计算控制增益（引风机→负压，二次风机→含氧）
5. 构建滑动窗口样本
6. RevIN标准化处理
7. 划分数据集
8. 返回PyTorch DataLoader

使用方式：
    from src.predictor.dataset import BoilerDataset

    # 基础用法（不使用特征工程）
    dataset = BoilerDataset(config)
    train_loader, val_loader, test_loader = dataset.get_loaders()

    # 集成特征工程
    dataset = BoilerDataset(config, use_feature_extraction=True, use_feature_selection=True)
    train_loader, val_loader, test_loader = dataset.get_loaders()
    gains = dataset.get_control_gains()  # 获取动态计算的增益
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from typing import Tuple, Dict, List, Optional
from pathlib import Path

from .config import (
    Config,
    TARGET_VARS,
    CONTROL_VARS,
    PRESSURE_VARS,
    OXYGEN_VARS,
    INDUCED_FAN_VARS,
    SECONDARY_FAN_VARS,
    MAIN_PRESSURE_VAR,
    MAIN_OXYGEN_VAR,
    get_state_vars,
)
from .utils import get_logger

logger = get_logger(__name__)


class GainEstimator:
    """控制增益动态估计器

    从实际数据中计算：
    - 引风机频率变化 → 负压变化的增益（Pa/Hz）
    - 二次风机频率变化 → 含氧变化的增益（%/Hz）

    方法：
    1. correlation: 基于变化率的相关性分析
    2. regression: 简单线性回归 ΔY vs ΔU
    3. steady_state: 稳态段识别后计算增益
    """

    def __init__(self, config: Config):
        self.config = config
        self.window = config.gain.estimation_window
        self.method = config.gain.method
        self.min_corr = config.gain.min_correlation

        # 结果存储
        self.induced_fan_pressure_gain = config.gain.induced_fan_pressure_gain_default
        self.secondary_fan_oxygen_gain = config.gain.secondary_fan_oxygen_gain_default
        self.gain_stats = {}

    def estimate(self, df: pd.DataFrame) -> Dict[str, float]:
        """估计控制增益

        Args:
            df: 原始数据DataFrame

        Returns:
            gains: {'induced_fan_pressure_gain': float, 'secondary_fan_oxygen_gain': float}
        """
        logger.info(f"开始估计控制增益（方法: {self.method}, 窗口: {self.window}分钟）")

        # 计算变化率
        df_diff = self._compute_derivatives(df)

        # 估计引风机→负压增益
        self.induced_fan_pressure_gain = self._estimate_pressure_gain(df_diff)

        # 估计二次风机→含氧增益
        self.secondary_fan_oxygen_gain = self._estimate_oxygen_gain(df_diff)

        # 更新config中的增益值
        self.config.gain.induced_fan_pressure_gain = self.induced_fan_pressure_gain
        self.config.gain.secondary_fan_oxygen_gain = self.secondary_fan_oxygen_gain

        logger.info(f"增益估计完成:")
        logger.info(f"  引风机→负压增益: {self.induced_fan_pressure_gain:.2f} Pa/Hz")
        logger.info(f"  二次风机→含氧增益: {self.secondary_fan_oxygen_gain:.3f} %/Hz")

        return {
            'induced_fan_pressure_gain': self.induced_fan_pressure_gain,
            'secondary_fan_oxygen_gain': self.secondary_fan_oxygen_gain,
        }

    def _compute_derivatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算变化率（差分）

        Args:
            df: 原始数据

        Returns:
            df_diff: 包含变化率的数据
        """
        # 计算各变量的变化率（每分钟变化）
        df_diff = df.copy()

        for var in TARGET_VARS + CONTROL_VARS:
            if var in df.columns:
                df_diff[f'{var}_diff'] = df[var].diff()

        # 删除第一行（diff产生NaN）
        df_diff = df_diff.dropna(subset=[f'{TARGET_VARS[0]}_diff'])

        return df_diff

    def _estimate_pressure_gain(self, df_diff: pd.DataFrame) -> float:
        """估计引风机→负压增益（考虑响应延迟）

        正确方法：
        1. 计算滞后互相关，找到最佳滞后时间k
        2. 在时刻t的控制变化Δu[t]与时刻t+k的响应变化Δy[t+k]之间计算相关性
        """
        # 引风机频率变化（两台取平均）
        induced_fan_diff_cols = [f'{v}_diff' for v in INDUCED_FAN_VARS if f'{v}_diff' in df_diff.columns]

        if not induced_fan_diff_cols:
            logger.warning("未找到引风机差分数据，使用默认增益")
            return self.config.gain.induced_fan_pressure_gain_default

        induced_fan_diff = df_diff[induced_fan_diff_cols].mean(axis=1)

        # 主负压变化
        pressure_diff_col = f'{MAIN_PRESSURE_VAR}_diff'
        if pressure_diff_col not in df_diff.columns:
            logger.warning("未找到负压差分数据，使用默认增益")
            return self.config.gain.induced_fan_pressure_gain_default

        pressure_diff = df_diff[pressure_diff_col]

        # 筛选有效样本（引风机有显著变化）
        significant_change = induced_fan_diff.abs() > 0.5  # Hz
        valid_mask = significant_change & pressure_diff.notna() & induced_fan_diff.notna()

        if valid_mask.sum() < 100:
            logger.warning(f"有效增益样本不足（{valid_mask.sum()}个），使用默认增益")
            return self.config.gain.induced_fan_pressure_gain_default

        # ===== 新增：滞后互相关分析 =====
        # 搜索最佳滞后时间（0-5分钟）
        max_lag = 5
        best_corr = 0
        best_lag = 0

        # 获取有效索引
        valid_indices = np.where(valid_mask)[0]

        for lag in range(max_lag + 1):
            if lag == 0:
                u = induced_fan_diff[valid_mask].values
                y = pressure_diff[valid_mask].values
            else:
                # Δu[t] 与 Δy[t+lag]，确保长度一致
                u_indices = valid_indices[valid_indices < len(induced_fan_diff) - lag]
                y_indices = u_indices + lag

                # 确保y_indices在有效范围内
                y_indices = y_indices[y_indices < len(pressure_diff)]

                # 再次对齐u_indices
                u_indices = y_indices - lag

                if len(u_indices) > 50:
                    u = induced_fan_diff.iloc[u_indices].values
                    y = pressure_diff.iloc[y_indices].values
                else:
                    continue

            if len(u) > 50 and len(u) == len(y):
                corr = np.corrcoef(u, y)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        logger.info(f"  引风机-负压滞后互相关分析:")
        logger.info(f"    最佳滞后: {best_lag} 分钟")
        logger.info(f"    最佳相关性: {best_corr:.3f}")

        self.gain_stats['pressure_best_lag'] = best_lag
        self.gain_stats['pressure_best_corr'] = best_corr

        # 使用最佳滞后计算增益
        if best_lag == 0:
            u_valid = induced_fan_diff[valid_mask].values
            y_valid = pressure_diff[valid_mask].values
        else:
            u_indices = valid_indices[valid_indices < len(induced_fan_diff) - best_lag]
            y_indices = u_indices + best_lag
            y_indices = y_indices[y_indices < len(pressure_diff)]
            u_indices = y_indices - best_lag

            u_valid = induced_fan_diff.iloc[u_indices].values
            y_valid = pressure_diff.iloc[y_indices].values

        # 增益估计：基于变化率的比值
        gain_samples = y_valid / u_valid
        gain = np.median(gain_samples)  # 使用中位数更稳健

        self.gain_stats['pressure_gain_median'] = gain
        self.gain_stats['pressure_gain_std'] = np.std(gain_samples)
        self.gain_stats['pressure_valid_samples'] = len(u_valid)

        if abs(best_corr) < self.min_corr:
            logger.warning(f"相关性低于阈值({self.min_corr})，使用默认增益")
            return self.config.gain.induced_fan_pressure_gain_default

        return gain

    def _estimate_oxygen_gain(self, df_diff: pd.DataFrame) -> float:
        """估计二次风机→含氧增益（考虑响应延迟）

        正确方法：
        1. 计算滞后互相关，找到最佳滞后时间k
        2. 在时刻t的控制变化Δu[t]与时刻t+k的响应变化Δy[t+k]之间计算相关性
        """
        # 二次风机频率变化（两台取平均）
        secondary_fan_diff_cols = [f'{v}_diff' for v in SECONDARY_FAN_VARS if f'{v}_diff' in df_diff.columns]

        if not secondary_fan_diff_cols:
            logger.warning("未找到二次风机差分数据，使用默认增益")
            return self.config.gain.secondary_fan_oxygen_gain_default

        secondary_fan_diff = df_diff[secondary_fan_diff_cols].mean(axis=1)

        # 主含氧变化
        oxygen_diff_col = f'{MAIN_OXYGEN_VAR}_diff'
        if oxygen_diff_col not in df_diff.columns:
            logger.warning("未找到含氧差分数据，使用默认增益")
            return self.config.gain.secondary_fan_oxygen_gain_default

        oxygen_diff = df_diff[oxygen_diff_col]

        # 筛选有效样本
        significant_change = secondary_fan_diff.abs() > 0.5  # Hz
        valid_mask = significant_change & oxygen_diff.notna() & secondary_fan_diff.notna()

        if valid_mask.sum() < 100:
            logger.warning(f"有效增益样本不足（{valid_mask.sum()}个），使用默认增益")
            return self.config.gain.secondary_fan_oxygen_gain_default

        # ===== 新增：滞后互相关分析 =====
        # 搜索最佳滞后时间（0-5分钟）
        max_lag = 5
        best_corr = 0
        best_lag = 0

        # 获取有效索引
        valid_indices = np.where(valid_mask)[0]

        for lag in range(max_lag + 1):
            if lag == 0:
                u = secondary_fan_diff[valid_mask].values
                y = oxygen_diff[valid_mask].values
            else:
                # Δu[t] 与 Δy[t+lag]，确保长度一致
                u_indices = valid_indices[valid_indices < len(secondary_fan_diff) - lag]
                y_indices = u_indices + lag

                # 确保y_indices在有效范围内
                y_indices = y_indices[y_indices < len(oxygen_diff)]

                # 再次对齐u_indices
                u_indices = y_indices - lag

                if len(u_indices) > 50:
                    u = secondary_fan_diff.iloc[u_indices].values
                    y = oxygen_diff.iloc[y_indices].values
                else:
                    continue

            if len(u) > 50 and len(u) == len(y):
                corr = np.corrcoef(u, y)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag

        logger.info(f"  二次风机-含氧滞后互相关分析:")
        logger.info(f"    最佳滞后: {best_lag} 分钟")
        logger.info(f"    最佳相关性: {best_corr:.3f}")

        self.gain_stats['oxygen_best_lag'] = best_lag
        self.gain_stats['oxygen_best_corr'] = best_corr

        # 使用最佳滞后计算增益
        if best_lag == 0:
            u_valid = secondary_fan_diff[valid_mask].values
            y_valid = oxygen_diff[valid_mask].values
        else:
            u_indices = valid_indices[valid_indices < len(secondary_fan_diff) - best_lag]
            y_indices = u_indices + best_lag
            y_indices = y_indices[y_indices < len(oxygen_diff)]
            u_indices = y_indices - best_lag

            u_valid = secondary_fan_diff.iloc[u_indices].values
            y_valid = oxygen_diff.iloc[y_indices].values

        # 增益估计
        gain_samples = y_valid / u_valid
        gain = np.median(gain_samples)

        self.gain_stats['oxygen_gain_median'] = gain
        self.gain_stats['oxygen_gain_std'] = np.std(gain_samples)
        self.gain_stats['oxygen_valid_samples'] = len(u_valid)

        if abs(best_corr) < self.min_corr:
            logger.warning(f"相关性低于阈值({self.min_corr})，使用默认增益")
            return self.config.gain.secondary_fan_oxygen_gain_default

        return gain

    def get_stats(self) -> Dict:
        """获取增益估计的统计信息"""
        return self.gain_stats


class RevINNormalizer:
    """RevIN实例归一化器

    对每个样本窗口独立标准化：
    - 计算窗口均值和标准差
    - 标准化：Y_normalized = (Y - mean) / std
    - 反标准化：Y_original = Y_normalized * std + mean

    支持最小标准差阈值，防止窗口内数据过于稳定导致标准化爆炸
    """

    def __init__(self, config: Config):
        self.config = config
        self.pressure_min_std = config.normalize.pressure_min_std
        self.oxygen_min_std = config.normalize.oxygen_min_std
        self.eps = 1e-5

    def normalize(
        self,
        data: np.ndarray,
        feature_type: str = 'mixed',
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """标准化数据

        Args:
            data: 输入数据 (N, T, n_features)
            feature_type: 特征类型
                - 'y_only': 全部是目标变量Y
                - 'mixed': 混合特征（前n_y是Y，后面是其他）

        Returns:
            normalized: 标准化后的数据
            mean: 窗口均值 (N, n_features)
            std: 窗口标准差 (N, n_features)
        """
        N, T, n_features = data.shape

        # 计算窗口统计量（沿时间轴）
        mean = np.mean(data, axis=1)  # (N, n_features)
        std = np.std(data, axis=1)    # (N, n_features)

        # 设置最小标准差阈值
        if feature_type == 'y_only' or feature_type == 'mixed':
            n_y = self.config.n_y
            # 负压最小标准差（前4个）
            std[:, :4] = np.maximum(std[:, :4], self.pressure_min_std)
            # 含氧最小标准差（后3个）
            std[:, 4:n_y] = np.maximum(std[:, 4:n_y], self.oxygen_min_std)

        # 标准化
        normalized = (data - mean[:, np.newaxis, :]) / (std[:, np.newaxis, :] + self.eps)

        return normalized, mean, std

    def normalize_y_only(
        self,
        y_data: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """仅标准化目标变量Y

        Args:
            y_data: Y数据 (N, T, n_y)

        Returns:
            normalized: 标准化后的Y
            y_mean: 窗口均值 (N, n_y)
            y_std: 窗口标准差 (N, n_y)
        """
        return self.normalize(y_data, feature_type='y_only')

    def inverse_normalize(
        self,
        normalized: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        """反标准化

        Args:
            normalized: 标准化后的数据 (N, T, n_features)
            mean: 窗口均值 (N, n_features)
            std: 窗口标准差 (N, n_features)

        Returns:
            original: 原始尺度的数据
        """
        return normalized * std[:, np.newaxis, :] + mean[:, np.newaxis, :]


class BoilerDataset:
    """锅炉预测数据集

    功能：
    1. 加载清洗后的数据
    2. 特征提取（可选）
    3. 特征选择（可选）
    4. 估计控制增益
    5. 构建滑动窗口样本
    6. 标准化处理（RevIN）
    7. 划分数据集
    8. 返回DataLoader

    数据格式：
    - encoder_input: (N, L, n_features) 历史窗口
    - decoder_input: (N, H, n_u) 未来控制输入
    - target: (N, H, n_y) 未来预测目标
    - window_stats: (N, n_y) 窗口统计量（用于RevIN反标准化）
    """

    def __init__(
        self,
        config: Config,
        data_path: Optional[Path] = None,
        use_feature_extraction: bool = False,
        use_feature_selection: bool = False,
        feature_selection_k: int = 80,
    ):
        """
        Args:
            config: 配置对象
            data_path: 数据文件路径（默认使用config中的路径）
            use_feature_extraction: 是否使用特征提取（调用 src/features/extractor.py）
            use_feature_selection: 是否使用特征选择（调用 src/features/selector.py）
            feature_selection_k: 特征选择的目标特征数
        """
        self.config = config
        self.data_path = Path(data_path) if data_path else Path(config.data_path) if config.data_path else None
        self.use_feature_extraction = use_feature_extraction
        self.use_feature_selection = use_feature_selection
        self.feature_selection_k = feature_selection_k

        # 参数简写
        self.L = config.L
        self.H = config.H
        self.n_y = config.n_y
        self.n_u = config.n_u

        # 数据
        self.df: Optional[pd.DataFrame] = None
        self.df_raw: Optional[pd.DataFrame] = None  # 原始数据（特征提取前）
        self.state_vars: List[str] = []
        self.n_x: int = 0
        self.feature_cols: List[str] = []  # 最终用于模型的特征列
        self.selected_features: List[str] = []  # 特征选择结果

        # 特征提取器（可选）
        self.feature_extractor: Optional[any] = None
        self.feature_selector: Optional[any] = None

        # 样本
        self.encoder_input: Optional[np.ndarray] = None
        self.decoder_input: Optional[np.ndarray] = None
        self.target: Optional[np.ndarray] = None
        self.window_stats: Optional[Dict] = None

        # 增益估计器
        self.gain_estimator = GainEstimator(config)
        self.gains: Dict[str, float] = {}

        # 标准化器
        self.normalizer = RevINNormalizer(config)

        # 全局标准化器（用于U和X）
        self.u_scaler_mean: Optional[np.ndarray] = None
        self.u_scaler_std: Optional[np.ndarray] = None
        self.x_scaler_mean: Optional[np.ndarray] = None
        self.x_scaler_std: Optional[np.ndarray] = None

    def load_data(self) -> pd.DataFrame:
        """加载清洗后的数据"""
        logger.info(f"加载数据: {self.data_path}")

        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

        # 支持feather和csv格式
        if self.data_path.suffix == '.feather':
            self.df = pd.read_feather(self.data_path)
        else:
            self.df = pd.read_csv(self.data_path)

        # 确认关键变量存在
        missing_vars = [v for v in TARGET_VARS + CONTROL_VARS if v not in self.df.columns]
        if missing_vars:
            raise ValueError(f"数据中缺少关键变量: {missing_vars}")

        # 获取状态变量
        self.state_vars = get_state_vars(self.df.columns.tolist())
        self.n_x = len(self.state_vars)

        logger.info(f"数据维度: {len(self.df)}行, {len(self.df.columns)}列")
        logger.info(f"目标变量(Y): {self.n_y}维")
        logger.info(f"控制变量(U): {self.n_u}维")
        logger.info(f"状态变量(X): {self.n_x}维")

        return self.df

    def extract_features(self) -> pd.DataFrame:
        """特征提取（调用 src/features/extractor.py）

        提取滞后、趋势、变化、响应、约束、窗口统计等特征
        """
        if self.df is None:
            self.load_data()

        logger.info("开始特征提取...")

        # 导入特征提取模块
        try:
            from src.features.extractor import create_extractor_pipeline
        except ImportError:
            logger.warning("特征提取模块未安装，跳过特征提取")
            return self.df

        # 创建特征提取管道
        pipeline = create_extractor_pipeline()

        # 执行特征提取（使用正确的方法名）
        feature_df = pipeline.extract_all(self.df, include_original=False)

        # 合并特征（保留原始目标变量）
        self.df_raw = self.df.copy()

        # 只保留提取的特征 + 目标变量
        extracted_features = [col for col in feature_df.columns if col not in TARGET_VARS]
        self.df = pd.concat([
            feature_df[extracted_features],
            self.df[TARGET_VARS],
        ], axis=1)

        logger.info(f"特征提取完成:")
        logger.info(f"  提取特征数: {len(extracted_features)}")
        logger.info(f"  总列数: {len(self.df.columns)}")

        return self.df

    def select_features(self) -> pd.DataFrame:
        """特征选择（调用 src/features/selector.py）

        使用滞后互信息 + Granger因果检验选择重要特征
        """
        if self.df is None:
            self.load_data()

        logger.info("开始特征选择...")

        # 导入特征选择模块
        try:
            from src.features.selector import FeatureSelector
        except ImportError:
            logger.warning("特征选择模块未安装，跳过特征选择")
            self.feature_cols = [col for col in self.df.columns if col not in TARGET_VARS]
            return self.df

        # 创建特征选择器
        selector = FeatureSelector(
            self.df,
            target_vars=TARGET_VARS,
        )

        # 执行特征选择
        self.selected_features = selector.select(
            k=self.feature_selection_k,
            use_granger=False,  # Granger因果检验耗时较长，默认关闭
            must_have_features=CONTROL_VARS.copy(),
        )

        # 保存特征选择器（用于后续保存/加载）
        self.feature_selector = selector

        # 更新DataFrame
        self.df_raw = self.df.copy()
        self.df = self.df[self.selected_features + TARGET_VARS]

        self.feature_cols = self.selected_features

        logger.info(f"特征选择完成:")
        logger.info(f"  选择特征数: {len(self.selected_features)}")
        logger.info(f"  控制参数保留: {len([f for f in CONTROL_VARS if f in self.selected_features])}")

        return self.df

    def estimate_gains(self) -> Dict[str, float]:
        """估计控制增益"""
        if self.df is None:
            self.load_data()

        self.gains = self.gain_estimator.estimate(self.df)
        return self.gains

    def build_samples(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
        """构建滑动窗口样本

        Returns:
            encoder_input: (N, L, n_features) 其中 n_features = n_y + n_other
            decoder_input: (N, H, n_u)
            target: (N, H, n_y)
            window_stats: {'y_mean': (N, n_y), 'y_std': (N, n_y)}
        """
        if self.df is None:
            self.load_data()

        # 确定特征列
        if not self.feature_cols:
            # 如果没有使用特征工程，使用原始状态变量
            if self.use_feature_extraction or self.use_feature_selection:
                # 应该已经通过 extract_features/select_features 设置了 feature_cols
                self.feature_cols = [col for col in self.df.columns if col not in TARGET_VARS]
            else:
                # 使用原始数据：控制变量 + 状态变量
                self.feature_cols = CONTROL_VARS + self.state_vars

        n_features = len(self.feature_cols)
        logger.info(f"构建滑动窗口样本（L={self.L}, H={self.H}, features={n_features}）")

        # 提取数据
        y_data = self.df[TARGET_VARS].values.astype(np.float32)

        # 提取特征数据
        available_features = [f for f in self.feature_cols if f in self.df.columns]
        if len(available_features) < len(self.feature_cols):
            missing = [f for f in self.feature_cols if f not in self.df.columns]
            logger.warning(f"部分特征不存在: {missing}")

        feature_data = self.df[available_features].values.astype(np.float32)
        n_features_actual = feature_data.shape[1]

        # 处理NaN
        y_data = np.nan_to_num(y_data, nan=0.0)
        feature_data = np.nan_to_num(feature_data, nan=0.0)

        n_total = len(self.df)

        # 计算全局标准化参数（用于特征）
        train_end = int(n_total * self.config.train.train_ratio)

        self.feature_scaler_mean = feature_data[:train_end].mean(axis=0)
        self.feature_scaler_std = feature_data[:train_end].std(axis=0) + 1e-5

        # 全局标准化特征
        feature_scaled = (feature_data - self.feature_scaler_mean) / self.feature_scaler_std

        # 构建样本索引
        valid_starts = np.arange(n_total - self.L - self.H + 1)
        n_samples = len(valid_starts)

        logger.info(f"总样本数: {n_samples}")

        # 初始化数组
        encoder_input = np.zeros((n_samples, self.L, self.n_y + n_features_actual), dtype=np.float32)
        decoder_input = np.zeros((n_samples, self.H, self.n_u), dtype=np.float32)
        target = np.zeros((n_samples, self.H, self.n_y), dtype=np.float32)
        y_mean_windows = np.zeros((n_samples, self.n_y), dtype=np.float32)
        y_std_windows = np.zeros((n_samples, self.n_y), dtype=np.float32)

        # 提取控制变量位置（用于decoder_input）
        control_indices = []
        for i, f in enumerate(available_features):
            if f in CONTROL_VARS:
                control_indices.append(i)

        # 构建每个样本
        for i, t in enumerate(valid_starts):
            # 历史Y（原始值，用于RevIN）
            y_hist = y_data[t:t + self.L]  # (L, n_y)

            # RevIN标准化历史Y
            y_hist_mean = y_hist.mean(axis=0)
            y_hist_std = y_hist.std(axis=0)

            # 应用最小标准差阈值
            y_hist_std[:4] = np.maximum(y_hist_std[:4], self.config.normalize.pressure_min_std)
            y_hist_std[4:] = np.maximum(y_hist_std[4:], self.config.normalize.oxygen_min_std)

            y_mean_windows[i] = y_hist_mean
            y_std_windows[i] = y_hist_std

            y_hist_normalized = (y_hist - y_hist_mean) / (y_hist_std + 1e-5)

            # 历史特征（全局标准化）
            feature_hist = feature_scaled[t:t + self.L]

            # 拼接encoder_input: Y_normalized + features
            encoder_input[i] = np.concatenate([y_hist_normalized, feature_hist], axis=1)

            # 未来控制U（从特征中提取控制变量）
            if control_indices:
                decoder_feature = feature_scaled[t + self.L:t + self.L + self.H]
                decoder_input[i] = decoder_feature[:, control_indices]

            # 未来目标Y（使用历史窗口统计量标准化 - RevIN的核心）
            # 问题：如果未来数据和历史差异大，标准化值可能极端
            # 解决：增大最小标准差阈值，或检查数据质量
            y_future = y_data[t + self.L:t + self.L + self.H]

            # 检查极端值：如果未来数据偏离历史均值太多，记录警告
            deviation = np.abs(y_future - y_hist_mean) / (y_hist_std + 1e-5)
            max_deviation = deviation.max()

            # 标准化target
            target[i] = (y_future - y_hist_mean) / (y_hist_std + 1e-5)

        self.encoder_input = encoder_input
        self.decoder_input = decoder_input
        self.target = target
        self.window_stats = {
            'y_mean': y_mean_windows,
            'y_std': y_std_windows,
        }

        # 更新n_x（用于模型创建）
        self.n_x = n_features_actual - self.n_u  # 除了控制变量的特征数

        # 检查标准化后数据范围
        y_norm_range = encoder_input[:, :, :self.n_y]
        logger.info(f"RevIN标准化后数据范围:")
        logger.info(f"  encoder Y范围: [{y_norm_range.min():.2f}, {y_norm_range.max():.2f}]")
        logger.info(f"  target范围: [{target.min():.2f}, {target.max():.2f}]")

        # 分变量统计y_std
        pressure_std = y_std_windows[:, :4]  # 负压变量
        oxygen_std = y_std_windows[:, 4:7]   # 含氧变量
        logger.info(f"  y_std统计:")
        logger.info(f"    负压: min={pressure_std.min():.2f}, max={pressure_std.max():.2f}, mean={pressure_std.mean():.2f}")
        logger.info(f"    含氧: min={oxygen_std.min():.2f}, max={oxygen_std.max():.2f}, mean={oxygen_std.mean():.2f}")

        logger.info(f"  encoder_input形状: {encoder_input.shape}")
        logger.info(f"  decoder_input形状: {decoder_input.shape}")

        # 分析target极端值的来源
        target_abs = np.abs(target)
        extreme_mask = target_abs > 5.0  # 超过±5σ的视为极端值
        n_extreme = extreme_mask.sum()
        extreme_ratio = n_extreme / target.size

        logger.info(f"  target极端值分析:")
        logger.info(f"    超过±5σ的数量: {n_extreme} ({extreme_ratio*100:.2f}%)")
        logger.info(f"    超过±10σ的数量: {(target_abs > 10).sum()} ({(target_abs > 10).sum()/target.size*100:.2f}%)")

        # 如果极端值比例高，给出建议
        if extreme_ratio > 0.01:  # 超过1%
            logger.warning(f"  ⚠ target极端值比例较高 ({extreme_ratio*100:.2f}%)")
            logger.warning(f"  建议: 1) 检查数据是否存在异常跳变; 2) 增大最小标准差阈值")

        return encoder_input, decoder_input, target, self.window_stats

    def split(
        self,
        encoder_input: np.ndarray,
        decoder_input: np.ndarray,
        target: np.ndarray,
        window_stats: Dict,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        shuffle_train: bool = True,
    ) -> Dict[str, Dict]:
        """划分数据集

        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            shuffle_train: 是否打乱训练集

        Returns:
            splits: {'train': {...}, 'val': {...}, 'test': {...}}
        """
        n_samples = len(encoder_input)

        # 按时间顺序划分
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))

        # 训练集
        train_idx = np.arange(train_end)
        if shuffle_train:
            np.random.shuffle(train_idx)

        splits = {
            'train': {
                'encoder_input': encoder_input[train_idx],
                'decoder_input': decoder_input[train_idx],
                'target': target[train_idx],
                'y_mean': window_stats['y_mean'][train_idx],
                'y_std': window_stats['y_std'][train_idx],
            },
            'val': {
                'encoder_input': encoder_input[train_end:val_end],
                'decoder_input': decoder_input[train_end:val_end],
                'target': target[train_end:val_end],
                'y_mean': window_stats['y_mean'][train_end:val_end],
                'y_std': window_stats['y_std'][train_end:val_end],
            },
            'test': {
                'encoder_input': encoder_input[val_end:],
                'decoder_input': decoder_input[val_end:],
                'target': target[val_end:],
                'y_mean': window_stats['y_mean'][val_end:],
                'y_std': window_stats['y_std'][val_end:],
            },
        }

        logger.info(f"数据集划分:")
        logger.info(f"  训练集: {len(splits['train']['encoder_input'])}样本")
        logger.info(f"  验证集: {len(splits['val']['encoder_input'])}样本")
        logger.info(f"  测试集: {len(splits['test']['encoder_input'])}样本")

        return splits

    def create_loaders(
        self,
        splits: Dict[str, Dict],
        batch_size: int = 64,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建DataLoader"""

        train_loader = self._create_loader(splits['train'], batch_size, shuffle=True)
        val_loader = self._create_loader(splits['val'], batch_size, shuffle=False)
        test_loader = self._create_loader(splits['test'], batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def _create_loader(
        self,
        data: Dict,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        """创建单个DataLoader"""
        encoder_tensor = torch.tensor(data['encoder_input'], dtype=torch.float32)
        decoder_tensor = torch.tensor(data['decoder_input'], dtype=torch.float32)
        target_tensor = torch.tensor(data['target'], dtype=torch.float32)
        y_mean_tensor = torch.tensor(data['y_mean'], dtype=torch.float32)
        y_std_tensor = torch.tensor(data['y_std'], dtype=torch.float32)

        dataset = TensorDataset(encoder_tensor, decoder_tensor, target_tensor, y_mean_tensor, y_std_tensor)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def get_loaders(
        self,
        batch_size: Optional[int] = None,
        val_ratio: Optional[float] = None,
        test_ratio: Optional[float] = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """一站式获取DataLoader

        流程：
        1. 加载数据
        2. 特征提取（可选）
        3. 特征选择（可选）
        4. 估计增益
        5. 构建样本
        6. 划分数据集
        7. 创建DataLoader

        Args:
            batch_size: 批大小（默认使用config中的值）
            val_ratio: 验证集比例（默认使用config中的值）
            test_ratio: 测试集比例（用于计算验证集结束位置）

        Returns:
            train_loader, val_loader, test_loader
        """
        batch_size = batch_size or self.config.train.batch_size
        val_ratio = val_ratio or self.config.train.val_ratio
        test_ratio = test_ratio or self.config.train.test_ratio
        train_ratio = 1.0 - val_ratio - test_ratio

        # 1. 加载数据
        if self.df is None:
            self.load_data()

        # 2. 特征提取（可选）
        if self.use_feature_extraction:
            self.extract_features()

        # 3. 特征选择（可选）
        if self.use_feature_selection:
            self.select_features()

        # 4. 估计增益
        if not self.gains:
            self.estimate_gains()

        # 5. 构建样本
        if self.encoder_input is None:
            self.build_samples()

        # 6. 划分数据集
        splits = self.split(
            self.encoder_input,
            self.decoder_input,
            self.target,
            self.window_stats,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        # 7. 创建DataLoader
        return self.create_loaders(splits, batch_size)

    def get_control_gains(self) -> Dict[str, float]:
        """获取控制增益"""
        return self.gains

    def get_scaler_params(self) -> Dict:
        """获取全局标准化参数"""
        return {
            'feature_mean': self.feature_scaler_mean,
            'feature_std': self.feature_scaler_std,
            'u_mean': self.u_scaler_mean,
            'u_std': self.u_scaler_std,
            'x_mean': self.x_scaler_mean,
            'x_std': self.x_scaler_std,
            'feature_cols': self.feature_cols,
            'selected_features': self.selected_features,
        }

    def get_feature_info(self) -> Dict:
        """获取特征信息"""
        return {
            'use_feature_extraction': self.use_feature_extraction,
            'use_feature_selection': self.use_feature_selection,
            'n_features': len(self.feature_cols),
            'feature_cols': self.feature_cols,
            'selected_features': self.selected_features,
        }

    def inverse_transform_predictions(
        self,
        predictions: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
    ) -> np.ndarray:
        """反标准化预测结果

        Args:
            predictions: 标准化后的预测 (N, H, n_y)
            y_mean: 窗口均值 (N, n_y)
            y_std: 窗口标准差 (N, n_y)

        Returns:
            predictions_original: 原始尺度的预测
        """
        return predictions * y_std[:, np.newaxis, :] + y_mean[:, np.newaxis, :]


__all__ = [
    "GainEstimator",
    "RevINNormalizer",
    "BoilerDataset",
]