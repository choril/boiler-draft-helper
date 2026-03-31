import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import (
    CONTROL_PARAMS,
    EXPERT_RANGES,
    KEY_PARAMS,
    OXYGEN_MAIN,
    OXYGEN_VARIABLES,
    PRESSURE_MAIN,
    PRESSURE_VARIABLES,
)
from src.features.base import BaseFeatureExtractor
from src.utils.utils import safe_divide


class TargetHistoryExtractor(BaseFeatureExtractor):
    def __init__(
        self, target_vars: list[str] | None = None, lags: list[int] | None = None
    ):
        super().__init__("target_history")
        self.target_vars = target_vars or [PRESSURE_MAIN, OXYGEN_MAIN]
        self.lags = lags or [1, 2, 3, 5, 10]
        self.trend_windows = [5, 10, 20, 30]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        for var in self.target_vars:
            for lag in self.lags:
                self.feature_names.append(f"{var}_lag_{lag}")
            for window in self.trend_windows:
                self.feature_names.extend(
                    [
                        f"{var}_trend_slope_{window}",
                        f"{var}_trend_accel_{window}",
                    ]
                )

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        for var in self._filter_columns(df, self.target_vars):
            data = df[var]
            for lag in self.lags:
                features[f"{var}_lag_{lag}"] = data.shift(lag).fillna(data.iloc[0])
            for window in self.trend_windows:
                rolling = data.rolling(window, min_periods=2)
                features[f"{var}_trend_slope_{window}"] = rolling.apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True,
                ).fillna(0)
                features[f"{var}_trend_accel_{window}"] = rolling.apply(
                    lambda x: np.polyfit(range(len(x)), x, 2)[0] if len(x) > 2 else 0,
                    raw=True,
                ).fillna(0)
        return pd.DataFrame(features)


class ControlParamsExtractor(BaseFeatureExtractor):
    def __init__(self, control_params: list[str] | None = None):
        super().__init__("control_params")
        self.control_params = control_params or CONTROL_PARAMS
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        for param in self.control_params:
            self.feature_names.extend(
                [
                    f"{param}_change",
                    f"{param}_change_mean_10",
                    f"{param}_change_std_10",
                    f"{param}_lag_1",
                ]
            )

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        for param in self._filter_columns(df, self.control_params):
            diff = df[param].diff()
            features[f"{param}_change"] = diff.fillna(0)
            features[f"{param}_change_mean_10"] = diff.rolling(10, min_periods=1).mean().fillna(0)
            features[f"{param}_change_std_10"] = diff.rolling(10, min_periods=1).std().fillna(0)
            features[f"{param}_lag_1"] = df[param].shift(1).fillna(df[param].iloc[0])
        return pd.DataFrame(features)


class ControlResponseExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__("control_response")
        self.feature_names = [
            "id_fan_pressure_gain",
            "pa_fan_oxygen_gain",
            "sa_fan_oxygen_gain",
            "coal_load_ratio",
            "load_coal_ratio",
        ]

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        if self._check_columns(df, ["2NC10CS901", "2NC2CS901", "2BK10CP004"]):
            id_fan_change = (df["2NC10CS901"].diff() + df["2NC2CS901"].diff()) / 2
            pressure_change = df["2BK10CP004"].diff()
            features["id_fan_pressure_gain"] = safe_divide(
                id_fan_change, pressure_change.abs()
            )
        if self._check_columns(df, ["2LB10CS001", "2LB20CS001", "2BK10CQ1"]):
            pa_fan_change = (df["2LB10CS001"].diff() + df["2LB20CS001"].diff()) / 2
            oxygen_change = df["2BK10CQ1"].diff()
            features["pa_fan_oxygen_gain"] = safe_divide(
                pa_fan_change, oxygen_change.abs()
            )
        if self._check_columns(df, ["2LB30CS901", "2LB40CS901", "2BK10CQ1"]):
            sa_fan_change = (df["2LB30CS901"].diff() + df["2LB40CS901"].diff()) / 2
            oxygen_change = df["2BK10CQ1"].diff()
            features["sa_fan_oxygen_gain"] = safe_divide(
                sa_fan_change, oxygen_change.abs()
            )
        if self._check_columns(df, ["D62AX002", "MSFLOW"]):
            features["coal_load_ratio"] = safe_divide(df["D62AX002"], df["MSFLOW"])
            features["load_coal_ratio"] = safe_divide(df["MSFLOW"], df["D62AX002"])
        return pd.DataFrame(features)


class PhysicsOptimizationExtractor(BaseFeatureExtractor):
    def __init__(self):
        super().__init__("physics_optimization")
        self.expert_ranges = EXPERT_RANGES
        self.pressure_main = PRESSURE_MAIN
        self.oxygen_main = OXYGEN_MAIN
        self.pressure_vars = PRESSURE_VARIABLES
        self.oxygen_vars = OXYGEN_VARIABLES
        self.feature_names = [
            "pressure_deviation_ideal",
            "pressure_in_ideal_range",
            "oxygen_deviation_ideal",
            "oxygen_in_ideal_range",
            "pressure_consistency",
            "oxygen_consistency",
            "coal_air_ratio",
            "primary_secondary_air_ratio",
            "total_air_flow",
            "load_change_rate",
            "id_fan_change_rate",
            "bed_temp_stability",
        ]

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        available_pressure = self._filter_columns(df, self.pressure_vars)
        available_oxygen = self._filter_columns(df, self.oxygen_vars)

        if self.pressure_main in df.columns:
            ideal_low, ideal_high = self.expert_ranges["pressure_ideal"]
            pressure_target = (ideal_low + ideal_high) / 2
            features["pressure_deviation_ideal"] = np.abs(
                df[self.pressure_main] - pressure_target
            )
            features["pressure_in_ideal_range"] = (
                (df[self.pressure_main] >= ideal_low)
                & (df[self.pressure_main] <= ideal_high)
            ).astype(int)

        if self.oxygen_main in df.columns:
            oxygen_target = self.expert_ranges["oxygen_target"]
            oxygen_low, oxygen_high = self.expert_ranges["oxygen_ideal"]
            features["oxygen_deviation_ideal"] = np.abs(
                df[self.oxygen_main] - oxygen_target
            )
            features["oxygen_in_ideal_range"] = (
                (df[self.oxygen_main] >= oxygen_low)
                & (df[self.oxygen_main] <= oxygen_high)
            ).astype(int)

        if len(available_pressure) >= 2:
            pressure_values = df[available_pressure].values
            features["pressure_consistency"] = np.std(pressure_values, axis=1)

        if len(available_oxygen) >= 2:
            oxygen_values = df[available_oxygen].values
            features["oxygen_consistency"] = np.std(oxygen_values, axis=1)

        if self._check_columns(df, ["D62AX002", "D61AX023", "D61AX024"]):
            total_air = df["D61AX023"] + df["D61AX024"]
            features["coal_air_ratio"] = safe_divide(df["D62AX002"], total_air / 1000)
            features["primary_secondary_air_ratio"] = safe_divide(
                df["D61AX023"], df["D61AX024"]
            )
            features["total_air_flow"] = total_air
        if "MSFLOW" in df.columns:
            features["load_change_rate"] = df["MSFLOW"].diff()
        if self._check_columns(df, ["2NC10CS901", "2NC2CS901"]):
            id_fan_speed = (df["2NC10CS901"] + df["2NC2CS901"]) / 2
            features["id_fan_change_rate"] = id_fan_speed.diff()
        if "D66P53A10" in df.columns:
            features["bed_temp_stability"] = (
                df["D66P53A10"].rolling(30, min_periods=1).std()
            )
        return pd.DataFrame(features)


class SlidingExtractor(BaseFeatureExtractor):
    def __init__(
        self, params: list[str] | None = None, window_sizes: list[int] | None = None
    ):
        super().__init__("balanced_sliding")
        self.params = params or KEY_PARAMS + CONTROL_PARAMS
        self.window_sizes = window_sizes or [10, 30]
        self.statistics = ["mean", "std"]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = [
            f"{param}_{stat}_{window}"
            for param in self.params
            for stat in self.statistics
            for window in self.window_sizes
        ]

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        params = self._filter_columns(df, self.params)
        for param in tqdm(params, desc=f"[{self.name}] 参数处理", leave=False):
            for window in self.window_sizes:
                rolling = df[param].rolling(window=window, min_periods=1)
                features[f"{param}_mean_{window}"] = rolling.mean()
                features[f"{param}_std_{window}"] = rolling.std()
        return pd.DataFrame(features)


class TrendExtractor(BaseFeatureExtractor):
    def __init__(self, params: list[str] | None = None):
        super().__init__("trend")
        self.params = params or [PRESSURE_MAIN, OXYGEN_MAIN] + CONTROL_PARAMS[:3]
        self._build_feature_names()

    def _build_feature_names(self):
        self.feature_names = []
        for p in self.params:
            self.feature_names.extend([f"{p}_diff", f"{p}_trend_30"])

    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        features: dict[str, pd.Series] = {}
        params = self._filter_columns(df, self.params)
        for param in tqdm(params, desc=f"[{self.name}] 参数处理", leave=False):
            features[f"{param}_diff"] = df[param].diff()
            features[f"{param}_trend_30"] = (
                df[param]
                .rolling(30, min_periods=2)
                .apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0,
                    raw=True,
                )
            )
        return pd.DataFrame(features)


def create_extractor_pipeline():
    from src.features.base import FeatureExtractorPipeline

    pipeline = FeatureExtractorPipeline()
    pipeline.add_extractor(TargetHistoryExtractor())
    pipeline.add_extractor(ControlParamsExtractor())
    pipeline.add_extractor(ControlResponseExtractor())
    pipeline.add_extractor(PhysicsOptimizationExtractor())
    pipeline.add_extractor(SlidingExtractor())
    pipeline.add_extractor(TrendExtractor())
    return pipeline
