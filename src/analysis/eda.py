from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import EXPERT_RANGES, KEY_PARAMS, KEY_PARAMS_FOR_STATS, TARGET_VARIABLES
from src.utils.utils import compute_in_range_ratio, print_section


class EDAnalyzer:
    def __init__(self, df: pd.DataFrame, param_dict: dict[str, Any]):
        self.df = df
        self.param_dict = param_dict
        self.target_vars = TARGET_VARIABLES
        self.key_params = KEY_PARAMS
        self.report: dict[str, Any] = {}

    def analyze_basic_stats(self) -> dict[str, Any]:
        print("\n1. 基础统计分析...")
        stats: dict[str, dict] = {}
        for col in self.df.columns:
            if col in ["TIME", "source_file"]:
                continue
            data = self.df[col].dropna()
            stats[col] = {
                "count": int(len(data)),
                "mean": float(data.mean()),
                "std": float(data.std()),
                "min": float(data.min()),
                "max": float(data.max()),
                "median": float(data.median()),
                "q25": float(data.quantile(0.25)),
                "q75": float(data.quantile(0.75)),
            }
        self.report["basic_stats"] = stats
        return stats

    def analyze_target_variables(self) -> dict[str, Any]:
        print("\n2. 目标变量分析...")
        target_stats: dict[str, dict] = {}
        for var in self.target_vars:
            if var not in self.df.columns:
                continue
            data = self.df[var].dropna()
            name = self.param_dict.get(var, {}).get("简称", var)
            is_pressure = "CP" in var
            ideal_range = (
                EXPERT_RANGES["pressure_ideal"]
                if is_pressure
                else EXPERT_RANGES["oxygen_ideal"]
            )
            in_ideal = compute_in_range_ratio(data, ideal_range[0], ideal_range[1])
            in_normal = (
                in_ideal
                if not is_pressure
                else compute_in_range_ratio(
                    data,
                    EXPERT_RANGES["pressure_normal"][0],
                    EXPERT_RANGES["pressure_normal"][1],
                )
            )
            target_stats[var] = {
                "name": name,
                "mean": float(data.mean()),
                "std": float(data.std()),
                "in_ideal_range_pct": float(in_ideal),
                "in_normal_range_pct": float(in_normal),
            }
            print(
                f"  {var} ({name}): 均值={data.mean():.2f}, 理想范围占比={in_ideal:.1f}%"
            )
        self.report["target_stats"] = target_stats
        return target_stats

    def analyze_correlations(self) -> dict[str, Any]:
        print("\n3. 相关性分析...")
        numeric_cols = [
            c
            for c in self.df.select_dtypes(include=[np.number]).columns
            if c not in ["TIME", "source_file"]
        ]
        corr_matrix = self.df[numeric_cols].corr()
        corr_results: dict[str, dict] = {}
        for target in self.target_vars[:2]:
            if target not in self.df.columns:
                continue
            corr_with_target = (
                corr_matrix[target].drop(target).abs().sort_values(ascending=False)
            )
            top_10 = corr_with_target.head(10).to_dict()
            corr_results[target] = {
                "name": self.param_dict.get(target, {}).get("简称", target),
                "top_correlations": {k: float(v) for k, v in top_10.items()},
            }
            print(f"\n  与 {target} 相关性TOP10:")
            for param, corr in list(top_10.items())[:5]:
                pname = self.param_dict.get(param, {}).get("简称", param)
                print(f"    {param} ({pname}): {corr:.4f}")
        self.report["correlations"] = corr_results
        return corr_results

    def analyze_distribution(self) -> dict[str, Any]:
        print("\n4. 分布特征分析...")
        distributions: dict[str, dict] = {}
        for col in self.key_params:
            if col not in self.df.columns:
                continue
            data = self.df[col].dropna()
            skewness = float(data.skew())
            kurtosis = float(data.kurtosis())
            distributions[col] = {
                "skewness": skewness,
                "kurtosis": kurtosis,
                "distribution_type": "normal" if abs(skewness) < 1 else "skewed",
            }
        self.report["distributions"] = distributions
        return distributions

    def analyze_time_series(self) -> dict[str, Any]:
        print("\n5. 时序特征分析...")
        ts_stats: dict[str, dict] = {}
        for col in self.key_params[:5]:
            if col not in self.df.columns:
                continue
            data = self.df[col]
            ts_stats[col] = {
                "autocorr_1": float(data.autocorr(1)) if len(data) > 1 else 0,
                "autocorr_10": float(data.autocorr(10)) if len(data) > 10 else 0,
                "volatility": float(data.rolling(60).std().mean()),
            }
        self.report["time_series"] = ts_stats
        return ts_stats

    def get_report(self) -> dict[str, Any]:
        return self.report

    def run_full_analysis(self) -> dict[str, Any]:
        print_section("探索性数据分析 (EDA)")
        print(f"数据维度: {self.df.shape}")
        self.analyze_basic_stats()
        self.analyze_target_variables()
        self.analyze_correlations()
        self.analyze_distribution()
        self.analyze_time_series()
        print_section("EDA分析完成!")
        return self.report


class ParameterClassifier:
    def __init__(self, param_dict: dict[str, Any]):
        self.param_dict = param_dict

    def classify(self) -> dict[str, list[str]]:
        categories: dict[str, list[str]] = {
            "控制变量": [],
            "状态变量": [],
            "其他变量": [],
        }
        control_keywords = [
            "给煤",
            "煤量",
            "控制",
            "指令",
            "设定",
            "转速",
            "开度",
            "阀门",
            "联络阀",
            "液偶",
        ]
        other_keywords = ["时间戳", "源文件"]
        for param_id, info in self.param_dict.items():
            desc = info.get("描述", "") or ""
            name = info.get("简称", "") or ""
            if any(kw in desc or kw in name for kw in control_keywords):
                categories["控制变量"].append(param_id)
            elif any(kw in desc or kw in name for kw in other_keywords):
                categories["其他变量"].append(param_id)
            else:
                categories["状态变量"].append(param_id)
        return categories

    def print_classification(self, categories: dict[str, list[str]]) -> None:
        print("\n【参数分类结果】")
        print("-" * 60)
        for cat, params in categories.items():
            print(f"\n{cat} ({len(params)}个):")
            for p in params[:10]:
                print(f"  - {p}: {self.param_dict[p].get('简称', 'N/A')}")
            if len(params) > 10:
                print(f"  ... 等共 {len(params)} 个")


class CorrelationAnalyzer:
    def __init__(
        self,
        df: pd.DataFrame,
        param_dict: dict[str, Any],
        target_vars: list[str] | None = None,
    ):
        self.df = df
        self.param_dict = param_dict
        self.target_vars = target_vars or TARGET_VARIABLES

    def analyze(self, top_n: int = 15) -> tuple[dict[str, float], dict[str, float]]:
        corr_pressure: dict[str, float] = {}
        corr_oxygen: dict[str, float] = {}
        pressure_target = self.target_vars[0]
        oxygen_target = self.target_vars[4]
        for param in self.df.columns:
            if param not in self.target_vars[:4]:
                try:
                    c = float(self.df[param].corr(self.df[pressure_target]))
                    if not np.isnan(c):
                        corr_pressure[param] = c
                except Exception:
                    pass
            if param not in self.target_vars[4:]:
                try:
                    c = float(self.df[param].corr(self.df[oxygen_target]))
                    if not np.isnan(c):
                        corr_oxygen[param] = c
                except Exception:
                    pass
        return corr_pressure, corr_oxygen

    def print_top_correlations(
        self, corr_dict: dict[str, float], target_name: str, top_n: int = 15
    ) -> None:
        sorted_corr = sorted(corr_dict.items(), key=lambda x: abs(x[1]), reverse=True)[
            :top_n
        ]
        print(f"\n与 {target_name} 相关性最强的前{top_n}个参数:")
        for param, corr in sorted_corr:
            param_name = self.param_dict.get(param, {}).get("简称", "N/A")
            print(f"  {param} ({param_name}): {corr:.4f}")


class StatisticsCalculator:
    def __init__(self, df: pd.DataFrame, param_dict: dict[str, Any]):
        self.df = df
        self.param_dict = param_dict

    def calculate_key_params_stats(
        self, params: list[str] | None = None
    ) -> pd.DataFrame:
        params = params or KEY_PARAMS_FOR_STATS + TARGET_VARIABLES
        stats_df = pd.DataFrame()
        for param in params:
            if param in self.df.columns:
                stats_df[param] = self.df[param].describe()
        stats_df.index = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        return stats_df

    def print_target_stats(self, target_vars: list[str] | None = None) -> None:
        target_vars = target_vars or TARGET_VARIABLES
        print("\n【四个炉膛压力和三个氧含量参数统计特征】")
        print("-" * 60)
        for param in target_vars:
            if param in self.df.columns:
                stats = self.df[param].describe()
                print(f"{param} ({self.param_dict[param].get('简称', 'N/A')}):")
                print(f"  均值: {stats['mean']:.2f}")
                print(f"  标准差: {stats['std']:.2f}")
                print(f"  最小值: {stats['min']:.2f}")
                print(f"  最大值: {stats['max']:.2f}")
                if "CP" in param:
                    in_range = compute_in_range_ratio(self.df[param], -150, -80)
                    print(f"  在最优范围(-80～-150Pa)内的比例: {in_range:.2f}%")
                else:
                    in_range = compute_in_range_ratio(self.df[param], 1.7, 2.3)
                    print(f"  在最优范围(2±0.3%)内的比例: {in_range:.2f}%")
