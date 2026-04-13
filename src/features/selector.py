import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from statsmodels.tsa.stattools import grangercausalitytests

from src.utils.config import (
    TARGET_VARIABLES,
    EXCLUDE_VARIABLES,
    SELF_DERIVED_PATTERNS,
    CAUSAL_FEATURE_PATTERNS,
    CONTROL_PARAMS,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FeatureSelector:
    """特征选择器 - 包含滞后互信息选择、相关性选择方法"""

    def __init__(
        self,
        feature_matrix: pd.DataFrame,
        target_vars: list[str] | None = None,
        exclude_cols: list[str] | None = None,
    ):
        self.feature_matrix = feature_matrix
        self.target_vars = target_vars or TARGET_VARIABLES
        self.exclude_cols = exclude_cols or EXCLUDE_VARIABLES
        self.selected_features: list[str] = []
        self.importance_scores: dict[str, float] = {}
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.scaler_params: dict = {}
        self.target_scaler: StandardScaler | None = None
        self.target_scaler_params: dict = {}

    def _get_numeric_features(self) -> list[str]:
        numeric_cols = self.feature_matrix.select_dtypes(
            include=[np.number]
        ).columns.tolist()
        return [c for c in numeric_cols if c not in self.exclude_cols]

    def _is_self_derived_feature(self, feature: str, target_var: str) -> bool:
        """判断是否为目标变量的自身衍生特征"""
        if not feature.startswith(target_var):
            return False
        return any(p in feature for p in SELF_DERIVED_PATTERNS)

    def _is_causal_feature(self, feature: str) -> bool:
        """判断是否为因果特征（包含控制参数或与目标变量相关的特征）"""
        return any(p in feature for p in CAUSAL_FEATURE_PATTERNS)

    def classify_features(
        self, features: list[str] | None = None, target_var: str | None = None
    ) -> dict[str, list[str] | dict]:
        """将特征分类为自身衍生、因果和其他特征"""
        self_derived = []
        causal = []
        other = []

        for feat in features:
            if self._is_self_derived_feature(feat, target_var):
                self_derived.append(feat)
            elif self._is_causal_feature(feat):
                causal.append(feat)
            else:
                other.append(feat)

        return {
            "self_derived": self_derived,
            "causal": causal,
            "other": other,
            "stats": {
                "total": len(features),
                "self_derived_count": len(self_derived),
                "causal_count": len(causal),
                "other_count": len(other),
                "self_derived_ratio": len(self_derived) / len(features)
                if features
                else 0,
            },
        }

    def _timeseries_sample(self, df: pd.DataFrame, sample_size: int) -> pd.DataFrame:
        """时序采样：分段采样 + 保留变化剧烈点"""
        n_segments = 10
        segment_size = len(df) // n_segments
        base_samples = sample_size // n_segments

        indices = []
        for i in range(n_segments):
            start = i * segment_size
            end = start + segment_size if i < n_segments - 1 else len(df)
            segment_indices = list(range(start, end))

            # 每段均匀采样基础数量
            if len(segment_indices) > base_samples:
                step = len(segment_indices) // base_samples
                indices.extend(segment_indices[::step])
            else:
                indices.extend(segment_indices)

        # 补充变化剧烈的点
        remaining = sample_size - len(indices)
        if remaining > 0:
            target_cols = [c for c in self.target_vars if c in df.columns]
            if target_cols:
                # 所有目标变量的平均变化率
                changes = df[target_cols].diff().abs().mean(axis=1)
            else:
                # 没有目标变量时，用所有特征的方差
                changes = df.std(axis=1)

            high_change_indices = changes.nlargest(remaining * 2).index.tolist()
            for idx in high_change_indices:
                if idx not in indices:
                    indices.append(idx)
                    if len(indices) >= sample_size:
                        break

        return df.iloc[sorted(indices[:sample_size])]

    def get_features_and_targets(
        self, sample_size: int | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        feature_cols = self._get_numeric_features()
        available_targets = [
            t for t in self.target_vars if t in self.feature_matrix.columns
        ]

        if not available_targets:
            logger.error("没有可用的目标变量，请检查配置和数据")
            raise ValueError("没有可用的目标变量")

        df = self.feature_matrix[feature_cols + available_targets].dropna()

        if sample_size and len(df) > sample_size:
            df = self._timeseries_sample(df, sample_size)

        X = df[feature_cols]
        y = df[available_targets]
        return X, y

    def select_by_lagged_mi(
        self,
        k: int = 80,
        horizons: list[int] | None = None,
        sample_size: int = 10000,
    ) -> tuple[list[str], dict[str, float]]:
        """滞后互信息特征选择 - 考虑特征对未来目标的影响

        Returns:
            tuple: (特征列表, 互信息分数字典)
        """

        logger.info(f"使用滞后互信息选择特征，horizons={horizons}, sample_size={sample_size}")
        feature_cols = self._get_numeric_features()
        available_targets = [t for t in self.target_vars if t in self.feature_matrix.columns]

        df = self.feature_matrix[feature_cols + available_targets].dropna()
        if sample_size and len(df) > sample_size:
            df = self._timeseries_sample(df, sample_size)

        X = df[feature_cols]
        y = df[available_targets]

        all_scores: dict[str, float] = {}
        for target in y.columns:
            for horizon in horizons:
                # 目标变量滞后
                y_future = y[target].shift(-horizon)
                X_valid = X.iloc[:-horizon] if horizon > 0 else X.copy()
                y_future_valid = y_future.iloc[:-horizon] if horizon > 0 else y_future.copy()

                valid_mask = ~(y_future_valid.isna() | X_valid.isna().any(axis=1))
                y_clean = y_future_valid[valid_mask]
                X_clean = X_valid[valid_mask]

                mi_scores = mutual_info_regression(
                    X_clean.values,
                    y_clean.values,
                    random_state=42
                )
                # 不同滞后步数加权：近期更重要
                weight = 1.0 / (1 + horizon / 10)
                for feat, score in zip(feature_cols, mi_scores):
                    if feat not in all_scores:
                        all_scores[feat] = 0
                    all_scores[feat] += score * weight

        sorted_features = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected = [f[0] for f in sorted_features]
        mi_scores = dict(sorted_features)

        self.selected_features = selected

        logger.info(f"滞后互信息选择完成，选出 {len(self.selected_features)} 个特征")
        logger.info(f"滞后互信息前20个重要特征:")
        for i, (feat, score) in enumerate(sorted_features[:20]):
            logger.info(f"  {i + 1}. {feat}: {score:.4f}")

        return selected, mi_scores

    def select_by_granger_causality(
        self,
        features: list[str] | None = None,
        max_lag: int = 10,
        sample_size: int = 3000,
        significance: float = 0.05,
        n_jobs: int = -1,
        pre_filter_correlation: float = 0.95,
    ) -> tuple[list[str], dict[str, float]]:
        """Granger因果检验 - 检验特征是否对目标有预测能力

        Args:
            features: 待检验的特征列表
            max_lag: 最大滞后阶数
            sample_size: 样本数量限制
            significance: 显著性阈值
            n_jobs: 并行任务数，-1表示使用所有CPU
            pre_filter_correlation: 预过滤高相关特征的阈值

        Returns:
            tuple: (显著特征列表, 特征显著性分数字典)
        """
       

        logger.info(f"Granger因果检验 (max_lag={max_lag})")

        if features is None:
            features = self._get_numeric_features()

        feature_cols = self._get_numeric_features()
        available_targets = [t for t in self.target_vars if t in self.feature_matrix.columns]

        df = self.feature_matrix[feature_cols + available_targets].dropna()
        if sample_size and len(df) > sample_size:
            df = df.iloc[-sample_size:]

        # 预过滤高相关特征
        if pre_filter_correlation > 0:
            features = self._pre_filter_collinear(
                df, features, threshold=pre_filter_correlation
            )
            logger.info(f"预过滤后剩余 {len(features)} 个特征")

        # 准备检验任务
        test_tasks = []
        for target in available_targets:
            for feat in features:
                test_tasks.append((target, feat, df[[target, feat]].values, max_lag))

        # 并行执行检验
        significant_features: dict[str, float] = {}
        n_workers = n_jobs if n_jobs > 0 else mp.cpu_count()

        logger.info(f"开始并行检验 {len(test_tasks)} 个特征-目标组合 (使用 {n_workers} 个进程)...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(self._granger_test_single, task): task
                for task in test_tasks
            }
            completed = 0
            for future in as_completed(futures):
                completed += 1
                try:
                    result = future.result()
                    if result is not None:
                        feat, pvalue, n_significant = result
                        if pvalue < significance or n_significant >= 3:
                            if feat not in significant_features:
                                significant_features[feat] = 0
                            # 综合评分：显著性次数 × (1-最小p值)
                            significant_features[feat] += n_significant * (1 - pvalue)
                except Exception:
                    pass

                if completed % 50 == 0 or completed == len(test_tasks):
                    logger.info(f"  进度: {completed}/{len(test_tasks)}")

        sorted_features = sorted(
            significant_features.items(), key=lambda x: x[1], reverse=True
        )
        selected = [f[0] for f in sorted_features]
        scores_dict = dict(sorted_features)

        logger.info(f"Granger因果检验完成，选出 {len(selected)} 个显著特征")
        logger.info(f"因果检验前15个显著特征:")
        for i, (feat, score) in enumerate(sorted_features[:15]):
            logger.info(f"  {i + 1}. {feat}: {score:.4f}")

        return selected, scores_dict

    def _pre_filter_collinear(
        self, df: pd.DataFrame, features: list[str], threshold: float = 0.95
    ) -> list[str]:
        """预过滤高相关特征，保留代表性特征"""
        if len(features) < 20:
            return features

        corr_matrix = df[features].corr().abs()

        # 基于相关性聚类，每组保留一个代表
        visited = set()
        representatives = []

        for feat in features:
            if feat in visited:
                continue
            representatives.append(feat)
            visited.add(feat)

            # 标记高相关特征为已访问
            for other in features:
                if other != feat and other not in visited:
                    if corr_matrix.loc[feat, other] > threshold:
                        visited.add(other)

        return representatives

    def _granger_test_single(task: tuple) -> tuple[str, float, int] | None:
        """单个Granger检验任务"""
        _target, feat, test_data, max_lag = task

        try:
            if len(test_data) < max_lag + 20:
                return None
            if np.any(np.isnan(test_data)):
                return None

            # 平稳性检查 - 方差过小可能导致检验失效
            if np.var(test_data[:, 0]) < 1e-10 or np.var(test_data[:, 1]) < 1e-10:
                return None

            result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

            # 收集所有滞后阶数的p值
            pvalues = [result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
            min_pvalue = min(pvalues)

            # 计算显著滞后阶数数量（一致性指标）
            n_significant = sum(1 for p in pvalues if p < 0.05)

            return feat, min_pvalue, n_significant

        except Exception:
            return None

    def select(
        self,
        k: int = 80,
        sample_size: int = 10000,
        horizons: list[int] | None = None,
        use_granger: bool = False,
        must_have_features: list[str] | None = None,
    ) -> list[str]:
        """特征选择

        Args:
            k: 目标特征数量
            sample_size: 采样数量
            horizons: 预测步长列表
            use_granger: 是否使用Granger因果检验
            must_have_features: 必须保留的特征
        """
        if must_have_features is None:
            must_have_features = CONTROL_PARAMS.copy()

        if horizons is None:
            horizons = [1, 3, 5, 10]

        logger.info("时序优化特征选择流程")
        # 滞后互信息选择
        lagged_mi_features, mi_scores = self.select_by_lagged_mi(k * 2, horizons, sample_size)
        # Granger因果检验选择
        if use_granger:
            _, granger_scores = self.select_by_granger_causality(features=lagged_mi_features)
        else:
            granger_scores = {}

        # 综合排序 - 使用重要性分数加权
        all_candidates: dict[str, float] = {}

        # 控制参数强制保留，赋予最高优先级
        for f in must_have_features:
            if f in self._get_numeric_features():
                all_candidates[f] = 999.0

        # 使用互信息分数加权
        for f, score in mi_scores.items():
            all_candidates[f] = all_candidates.get(f, 0.0) + score

        # 使用 Granger 显著性分数加权
        for f, score in granger_scores.items():
            all_candidates[f] = all_candidates.get(f, 0.0) + score

        # 排序时，分数高的排前面，分数相同时按名称升序
        sorted_candidates = sorted(
            all_candidates.items(),
            key=lambda x: (-x[1], x[0])
        )
        selected = [f[0] for f in sorted_candidates[:k]]
        # 移除高相关冗余特征 (放宽阈值到0.95以保留更多特征)
        selected = self.remove_collinear_features(selected, threshold=0.95)
        # 移除低方差特征
        selected = self.remove_low_variance_features(selected, threshold=0.01)
        
        # 如果移除后特征数不足，从候选中补充
        if len(selected) < k:
            logger.info(f"移除后剩余 {len(selected)} 个特征，补充 {k - len(selected)} 个")
            remaining = [f[0] for f in sorted_candidates if f[0] not in selected]
            for f in remaining[:k - len(selected)]:
                selected.append(f)
        # 检查并补充控制参数
        for f in must_have_features:
            if f in self._get_numeric_features() and f not in selected:
                selected.append(f)

        self.selected_features = selected
        # 将合并后的分数保存到 importance_scores（只保留选中特征的分数）
        self.importance_scores = {f: all_candidates.get(f, 0.0) for f in selected}

        logger.info(f"\n最终选择: {len(selected)} 个特征")
        logger.info(f"控制参数保留: {len([f for f in must_have_features if f in selected])} / {len(must_have_features)} ({len([f for f in must_have_features if f in selected]) / len(must_have_features) * 100:.1f}%)")

        return selected

    def remove_collinear_features(
        self,
        features: list[str] | None = None,
        threshold: float = 0.95,
        protected_features: list[str] | None = None,
    ) -> list[str]:
        """移除高相关冗余特征

        Args:
            features: 待筛选的特征列表
            threshold: 相关系数阈值
            protected_features: 受保护的特征（不会被移除）
        """
        if features is None:
            features = self.selected_features

        if protected_features is None:
            protected_features = CONTROL_PARAMS.copy()

        logger.info("移除冗余特征")

        df = self.feature_matrix[features].dropna()
        corr_matrix = df.corr().abs()

        to_remove = set()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                if corr_matrix.iloc[i, j] > threshold:
                    feat_i, feat_j = features[i], features[j]
                    # 保护控制参数不被移除
                    if feat_i in protected_features:
                        if feat_j not in protected_features:
                            to_remove.add(feat_j)
                    elif feat_j in protected_features:
                        to_remove.add(feat_i)
                    elif feat_j not in to_remove:
                        score_i = self.importance_scores.get(feat_i, 0)
                        score_j = self.importance_scores.get(feat_j, 0)
                        if score_i >= score_j:
                            to_remove.add(feat_j)
                        else:
                            to_remove.add(feat_i)

        selected = [f for f in features if f not in to_remove]
        logger.info(f"移除 {len(to_remove)} 个冗余特征")

        return selected

    def remove_low_variance_features(
        self,
        features: list[str] | None = None,
        threshold: float = 0.01,
        protected_features: list[str] | None = None,
    ) -> list[str]:
        """移除低方差特征

        Args:
            features: 待筛选的特征列表
            threshold: 方差阈值
            protected_features: 受保护的特征（不会被移除）
        """
        if features is None:
            features = self.selected_features

        if protected_features is None:
            protected_features = CONTROL_PARAMS.copy()
        
        logger.info("移除低方差特征")

        df = self.feature_matrix[features].dropna()
        variances = df.var()

        selected = []
        for f in features:
            if f in protected_features:
                selected.append(f)  # 受保护的特征强制保留
            elif variances[f] >= threshold:
                selected.append(f)

        removed = len(features) - len(selected)
        logger.info(f"移除 {removed} 个低方差特征")

        return selected

    def fit_scaler(
        self,
        method: Literal["standard", "minmax"] = "standard",
        target: Literal["features", "targets"] = "features",
        features: list[str] | None = None,
    ) -> None:
        """拟合缩放器

        Args:
            method: 缩放方法，"standard" 或 "minmax"
            target: 缩放目标，"features" 缩放特征，"targets" 缩放目标变量
            features: 特征列表，仅当 target="features" 时使用
        """
        if target == "features":
            if features is None:
                features = self.selected_features
            data = self.feature_matrix[features].dropna()
            columns = features
        else:
            data = self.feature_matrix[self.target_vars].dropna()
            columns = self.target_vars

        if method == "standard":
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        scaler.fit(data)

        params = {
            "method": method,
            "mean": scaler.mean_.tolist() if hasattr(scaler, "mean_") else None,
            "std": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            "min": scaler.data_min_.tolist() if hasattr(scaler, "data_min_") else None,
            "max": scaler.data_max_.tolist() if hasattr(scaler, "data_max_") else None,
        }

        if target == "features":
            self.scaler = scaler
            self.scaler_params = params
            logger.info(f"特征缩放器已拟合，方法: {method}，特征数: {len(columns)}")
        else:
            self.target_scaler = scaler
            self.target_scaler_params = params
            logger.info(f"目标变量缩放器已拟合，方法: {method}")

    def build_seq2seq_sequences(
        self,
        seq_length: int = 30,
        output_steps: int = 10,
        features: list[str] | None = None,
        step: int = 1,
        scale_features: bool = True,
        scale_targets: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if features is None:
            features = self.selected_features

        logger.info(f"Seq2Seq序列构建，输出步数={output_steps}")

        df = self.feature_matrix[features + self.target_vars].dropna()

        X_data = df[features].values.astype(np.float32)
        y_data = df[self.target_vars].values.astype(np.float32)

        if scale_features:
            self.fit_scaler(target="features", features=features)
            X_data = self.scaler.transform(X_data).astype(np.float32)

        if scale_targets:
            self.fit_scaler(target="targets")
            y_data = self.target_scaler.transform(y_data).astype(np.float32)

        n_samples = len(df) - seq_length - output_steps + 1
        if n_samples <= 0:
            logger.error(
                f"数据长度不足以构建序列: len={len(df)}, seq_length={seq_length}, output_steps={output_steps}"
            )
            raise ValueError(
                f"数据长度不足以构建序列: len={len(df)}, seq_length={seq_length}, output_steps={output_steps}"
            )

        shape = (n_samples, seq_length, X_data.shape[1])
        strides = (X_data.strides[0], X_data.strides[0], X_data.strides[1])
        X_all = np.lib.stride_tricks.as_strided(X_data, shape=shape, strides=strides)

        y_all = np.zeros(
            (n_samples, output_steps, len(self.target_vars)), dtype=np.float32
        )
        for i in range(output_steps):
            y_all[:, i, :] = y_data[seq_length + i : seq_length + i + n_samples]

        X = X_all[::step].copy()
        y = y_all[::step].copy()

        del X_data, y_data, X_all, y_all

        logger.info(f"序列构建完成:")
        logger.info(f"  - 序列长度: {seq_length}")
        logger.info(f"  - 输出步数: {output_steps}")
        logger.info(f"  - 样本数: {len(X)}")
        logger.info(f"  - 输入形状: {X.shape}")
        logger.info(f"  - 输出形状: {y.shape}")

        return X, y

    def save_results(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results = {
            "selected_features": self.selected_features,
            "importance_scores": self.importance_scores,
            "target_variables": self.target_vars,
            "n_features": len(self.selected_features),
            "scaler_params": self.scaler_params,
            "target_scaler_params": self.target_scaler_params,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        logger.info(f"特征选择结果已保存至: {output_path}")

    def load_results(self, input_path: str | Path) -> dict:
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        self.selected_features = results["selected_features"]
        self.importance_scores = results.get("importance_scores", {})
        self.target_vars = results.get("target_variables", self.target_vars)
        self.scaler_params = results.get("scaler_params", {})
        self.target_scaler_params = results.get("target_scaler_params", {})

        if self.target_scaler_params:
            self.target_scaler = StandardScaler()
            self.target_scaler.mean_ = np.array(self.target_scaler_params["mean"])
            self.target_scaler.scale_ = np.array(self.target_scaler_params["std"])

        logger.info(f"特征选择结果已加载: {input_path}")
        logger.info(f"特征数: {len(self.selected_features)}")

        return results
