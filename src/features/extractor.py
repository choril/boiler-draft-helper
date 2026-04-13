import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import (
    CONTROL_PARAMS,
    EXPERT_RANGES,
    OXYGEN_VARIABLES,
    PARAMS_FOR_STATS,
    PRESSURE_VARIABLES,
    TARGET_VARIABLES,
)
from src.features.base import BaseFeatureExtractor
from src.utils.utils import safe_divide


class LagFeatureExtractor(BaseFeatureExtractor):
    """滞后特征提取器

    提取目标变量和控制参数的滞后值
    """

    def __init__(
        self,
        target_vars: list[str] | None = None,
        control_vars: list[str] | None = None,
        lags: list[int] | None = None,
    ):
        super().__init__("lag_features")
        # 目标变量（预测对象）
        self.target_vars = target_vars or TARGET_VARIABLES
        self.control_vars = control_vars or CONTROL_PARAMS
        self.lags = lags or [1, 2, 3, 5, 10]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        all_vars = self.target_vars + self.control_vars
        for var in all_vars:
            for lag in self.lags:
                self.feature_names.append(f"{var}_lag_{lag}")

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        all_vars = self._filter_columns(df, self.target_vars + self.control_vars)

        for var in all_vars:
            data = df[var]
            for lag in self.lags:
                # 使用前向填充，避免用未来数据
                features[f"{var}_lag_{lag}"] = data.shift(lag).ffill().fillna(0)

        return pd.DataFrame(features)


class TrendFeatureExtractor(BaseFeatureExtractor):
    """趋势特征提取器

    提取变量的趋势斜率和加速度
    """

    def __init__(
        self,
        params: list[str] | None = None,
        windows: list[int] | None = None,
    ):
        super().__init__("trend_features")
        self.params = params or TARGET_VARIABLES + CONTROL_PARAMS
        self.windows = windows or [5, 10, 20, 30]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        for param in self.params:
            for window in self.windows:
                self.feature_names.extend([
                    f"{param}_trend_slope_{window}",
                    f"{param}_trend_accel_{window}",
                ])

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        params = self._filter_columns(df, self.params)

        for param in params:
            data = df[param]
            for window in self.windows:
                rolling = data.rolling(window, min_periods=window)

                # 趋势斜率：线性回归的一次项系数
                features[f"{param}_trend_slope_{window}"] = rolling.apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == window else np.nan,
                    raw=True,
                )

                # 趋势加速度：二次回归的二阶导数 = 2 * 二次项系数
                features[f"{param}_trend_accel_{window}"] = rolling.apply(
                    lambda x: np.polyfit(np.arange(len(x)), x, 2)[0] * 2 if len(x) == window else np.nan,
                    raw=True,
                )

        return pd.DataFrame(features)


class ChangeFeatureExtractor(BaseFeatureExtractor):
    """变化特征提取器

    提取控制参数的变化量和变化统计
    """

    def __init__(
        self,
        control_params: list[str] | None = None,
        windows: list[int] | None = None,
    ):
        super().__init__("change_features")
        self.control_params = control_params or CONTROL_PARAMS
        self.windows = windows or [10]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        for param in self.control_params:
            self.feature_names.append(f"{param}_change")
            for window in self.windows:
                self.feature_names.extend([
                    f"{param}_change_mean_{window}",
                    f"{param}_change_std_{window}",
                ])

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        params = self._filter_columns(df, self.control_params)

        for param in params:
            diff = df[param].diff()
            features[f"{param}_change"] = diff.fillna(0)

            for window in self.windows:
                # 使用完整窗口计算统计量
                rolling = diff.rolling(window, min_periods=window)
                features[f"{param}_change_mean_{window}"] = rolling.mean()
                features[f"{param}_change_std_{window}"] = rolling.std()

        return pd.DataFrame(features)


class ResponseFeatureExtractor(BaseFeatureExtractor):
    """响应特征提取器

    提取执行机构对过程变量的响应增益
    """

    def __init__(self):
        super().__init__("response_features")
        self.feature_names = [
            # 转速响应增益
            "id_fan_speed_pressure_gain",
            "sa_fan_speed_oxygen_gain",
            "pa_fan_speed_load_gain",
            # 电流响应增益（反映风机实际负载变化）
            "id_fan_current_pressure_gain",
            "sa_fan_current_oxygen_gain",
            "pa_fan_current_load_gain",
            # 风量响应增益
            "secondary_air_oxygen_gain",
            "primary_air_load_gain",
            # 给煤量响应增益
            "coal_load_gain",
            "coal_oxygen_gain",
            "coal_pressure_gain",
            # 煤负荷比
            "coal_load_ratio",
        ]

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}

        # ========== 转速响应增益 ==========
        # 引风机转速→炉膛负压（调负压的主要手段）
        if self._check_columns(df, ["2NC10CS901", "2NC2CS901", "2BK10CP004"]):
            id_fan_speed_change = (df["2NC10CS901"].diff() + df["2NC2CS901"].diff()) / 2
            pressure_change = df["2BK10CP004"].diff()
            features["id_fan_speed_pressure_gain"] = safe_divide(
                id_fan_speed_change, pressure_change.abs()
            ).fillna(0)

        # 二次风机转速→氧量（调氧量的主要手段）
        if self._check_columns(df, ["2LB30CS901", "2LB40CS901", "2BK10CQ1"]):
            sa_fan_speed_change = (df["2LB30CS901"].diff() + df["2LB40CS901"].diff()) / 2
            oxygen_change = df["2BK10CQ1"].diff()
            features["sa_fan_speed_oxygen_gain"] = safe_divide(
                sa_fan_speed_change, oxygen_change.abs()
            ).fillna(0)

        # 一次风机转速→主蒸汽流量（锅炉负荷）（快速提升负荷时的响应）
        if self._check_columns(df, ["2LB10CS001", "2LB20CS001", "MSFLOW"]):
            pa_fan_speed_change = (df["2LB10CS001"].diff() + df["2LB20CS001"].diff()) / 2
            load_change = df["MSFLOW"].diff()
            features["pa_fan_speed_load_gain"] = safe_divide(
                pa_fan_speed_change, load_change.abs()
            ).fillna(0)

        # ========== 电流响应增益 ==========
        # 引风机电流→炉膛负压（电流反映实际负载）
        if self._check_columns(df, ["2BBA15Q11", "2BBB13Q11", "2BK10CP004"]):
            id_fan_current_change = (df["2BBA15Q11"].diff() + df["2BBB13Q11"].diff()) / 2
            pressure_change = df["2BK10CP004"].diff()
            features["id_fan_current_pressure_gain"] = safe_divide(
                id_fan_current_change, pressure_change.abs()
            ).fillna(0)

        # 二次风机电流→氧量
        if self._check_columns(df, ["2BBA13Q11", "2BBB11Q11", "2BK10CQ1"]):
            sa_fan_current_change = (df["2BBA13Q11"].diff() + df["2BBB11Q11"].diff()) / 2
            oxygen_change = df["2BK10CQ1"].diff()
            features["sa_fan_current_oxygen_gain"] = safe_divide(
                sa_fan_current_change, oxygen_change.abs()
            ).fillna(0)

        # 一次风机电流→主蒸汽流量（锅炉负荷）
        if self._check_columns(df, ["2BBA14Q11", "2BBB12Q11", "MSFLOW"]):
            pa_fan_current_change = (df["2BBA14Q11"].diff() + df["2BBB12Q11"].diff()) / 2
            load_change = df["MSFLOW"].diff()
            features["pa_fan_current_load_gain"] = safe_divide(
                pa_fan_current_change, load_change.abs()
            ).fillna(0)

        # ========== 风量响应增益 ==========
        # 二次风量→氧量（最直接的因果关系）
        if self._check_columns(df, ["D61AX024", "2BK10CQ1"]):
            secondary_air_change = df["D61AX024"].diff()
            oxygen_change = df["2BK10CQ1"].diff()
            features["secondary_air_oxygen_gain"] = safe_divide(
                secondary_air_change, oxygen_change.abs()
            ).fillna(0)

        # 一次风量→主蒸汽流量（负荷提升时的响应）
        if self._check_columns(df, ["D61AX023", "MSFLOW"]):
            primary_air_change = df["D61AX023"].diff()
            load_change = df["MSFLOW"].diff()
            features["primary_air_load_gain"] = safe_divide(
                primary_air_change, load_change.abs()
            ).fillna(0)

        # ========== 给煤量响应增益 ==========
        # 给煤量→锅炉负荷
        if self._check_columns(df, ["D62AX002", "MSFLOW"]):
            coal_change = df["D62AX002"].diff()
            load_change = df["MSFLOW"].diff()
            features["coal_load_gain"] = safe_divide(
                coal_change, load_change.abs()
            ).fillna(0)
            # 煤负荷比（燃烧效率指标）
            features["coal_load_ratio"] = safe_divide(
                df["D62AX002"], df["MSFLOW"]
            ).fillna(0)

        # 给煤量→氧量
        if self._check_columns(df, ["D62AX002", "2BK10CQ1"]):
            coal_change = df["D62AX002"].diff()
            oxygen_change = df["2BK10CQ1"].diff()
            features["coal_oxygen_gain"] = safe_divide(
                coal_change, oxygen_change.abs()
            ).fillna(0)

        # 给煤量→炉膛压力
        if self._check_columns(df, ["D62AX002", "2BK10CP004"]):
            coal_change = df["D62AX002"].diff()
            pressure_change = df["2BK10CP004"].diff()
            features["coal_pressure_gain"] = safe_divide(
                coal_change, pressure_change.abs()
            ).fillna(0)

        return pd.DataFrame(features)


class ConstraintFeatureExtractor(BaseFeatureExtractor):
    """约束特征提取器

    提取基于专家知识的工艺约束和安全边界特征
    """

    def __init__(self):
        super().__init__("constraint_features")
        self.expert_ranges = EXPERT_RANGES
        # 使用第一个变量作为理想区间特征的参考
        self.pressure_ref = PRESSURE_VARIABLES[0]  # "2BK10CP004"
        self.oxygen_ref = OXYGEN_VARIABLES[0]     # "2BK10CQ1"
        self.pressure_vars = PRESSURE_VARIABLES
        self.oxygen_vars = OXYGEN_VARIABLES
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = [
            # 理想区间特征
            "pressure_deviation_ideal",
            "pressure_in_ideal_range",
            "oxygen_deviation_ideal",
            "oxygen_in_ideal_range",
            # 一致性特征（多点测量标准差）
            "pressure_consistency",
            "oxygen_consistency",
            # 测点偏差特征（其他测点与主目标的偏差）
            "pressure2_deviation",
            "pressure3_deviation",
            "pressure4_deviation",
            "oxygen2_deviation",
            "oxygen3_deviation",
            # 变化率特征
            "load_change_rate",
            "bed_temp_stability",
        ]

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        available_pressure = self._filter_columns(df, self.pressure_vars)
        available_oxygen = self._filter_columns(df, self.oxygen_vars)

        # ========== 理想区间特征 ==========
        # 床压理想区间特征
        if self.pressure_ref in df.columns:
            ideal_low, ideal_high = self.expert_ranges["pressure_ideal"]
            pressure_target = (ideal_low + ideal_high) / 2
            features["pressure_deviation_ideal"] = np.abs(
                df[self.pressure_ref] - pressure_target
            )
            features["pressure_in_ideal_range"] = (
                (df[self.pressure_ref] >= ideal_low)
                & (df[self.pressure_ref] <= ideal_high)
            ).astype(int)

        # 氧量理想区间特征
        if self.oxygen_ref in df.columns:
            oxygen_target = self.expert_ranges["oxygen_target"]
            oxygen_low, oxygen_high = self.expert_ranges["oxygen_ideal"]
            features["oxygen_deviation_ideal"] = np.abs(
                df[self.oxygen_ref] - oxygen_target
            )
            features["oxygen_in_ideal_range"] = (
                (df[self.oxygen_ref] >= oxygen_low)
                & (df[self.oxygen_ref] <= oxygen_high)
            ).astype(int)

        # ========== 一致性特征（多点测量标准差）==========
        if len(available_pressure) >= 2:
            pressure_values = df[available_pressure].values
            features["pressure_consistency"] = np.std(pressure_values, axis=1)

        if len(available_oxygen) >= 2:
            oxygen_values = df[available_oxygen].values
            features["oxygen_consistency"] = np.std(oxygen_values, axis=1)

        # ========== 测点偏差特征（其他测点与参考点的偏差）==========
        # 压力测点偏差
        if self.pressure_ref in df.columns:
            other_pressure = [v for v in self.pressure_vars if v != self.pressure_ref]
            for i, var in enumerate(other_pressure):
                if var in df.columns:
                    features[f"pressure{i+2}_deviation"] = df[var] - df[self.pressure_ref]

        # 氧量测点偏差
        if self.oxygen_ref in df.columns:
            other_oxygen = [v for v in self.oxygen_vars if v != self.oxygen_ref]
            for i, var in enumerate(other_oxygen):
                if var in df.columns:
                    features[f"oxygen{i+2}_deviation"] = df[var] - df[self.oxygen_ref]

        # ========== 变化率特征 ==========
        # 负荷变化率
        if "MSFLOW" in df.columns:
            features["load_change_rate"] = df["MSFLOW"].diff().fillna(0)

        # 床温稳定性
        if "D66P53A10" in df.columns:
            features["bed_temp_stability"] = (
                df["D66P53A10"].rolling(30, min_periods=30).std()
            )

        return pd.DataFrame(features)


class WindowStatsExtractor(BaseFeatureExtractor):
    """窗口统计特征提取器

    提取滑动窗口内的均值和标准差
    """

    def __init__(
        self,
        params: list[str] | None = None,
        window_sizes: list[int] | None = None,
    ):
        super().__init__("window_stats")
        self.params = params or PARAMS_FOR_STATS
        self.window_sizes = window_sizes or [10, 30]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        for param in self.params:
            for window in self.window_sizes:
                self.feature_names.extend([
                    f"{param}_mean_{window}",
                    f"{param}_std_{window}",
                ])

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}

        for param in self._filter_columns(df, self.params):
            for window in self.window_sizes:
                rolling = df[param].rolling(window=window, min_periods=window)
                features[f"{param}_mean_{window}"] = rolling.mean()
                features[f"{param}_std_{window}"] = rolling.std()

        return pd.DataFrame(features)


def create_extractor_pipeline():
    """创建特征提取管道"""
    from src.features.base import FeatureExtractorPipeline

    pipeline = FeatureExtractorPipeline()
    pipeline.add_extractor(LagFeatureExtractor())
    pipeline.add_extractor(TrendFeatureExtractor())
    pipeline.add_extractor(ChangeFeatureExtractor())
    pipeline.add_extractor(ResponseFeatureExtractor())
    pipeline.add_extractor(ConstraintFeatureExtractor())
    pipeline.add_extractor(WindowStatsExtractor())
    return pipeline