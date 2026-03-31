from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from tqdm import tqdm


class BaseFeatureExtractor(ABC):
    """特征提取器基类"""

    def __init__(self, name: str):
        self.name = name
        self.feature_names: list[str] = []

    @abstractmethod
    def extract(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """提取特征，返回特征 DataFrame"""
        pass

    def get_feature_names(self) -> list[str]:
        return self.feature_names

    def _filter_columns(self, df: pd.DataFrame, required: list[str]) -> list[str]:
        """筛选存在的列"""
        return [c for c in required if c in df.columns]

    def _check_columns(self, df: pd.DataFrame, required: list[str]) -> bool:
        """检查所需列是否都存在"""
        return all(c in df.columns for c in required)


class FeatureExtractorPipeline:
    """特征提取管道"""

    def __init__(self, extractors: list[BaseFeatureExtractor] | None = None):
        self.extractors = extractors or []
        self.feature_info: dict[str, Any] = {}

    def add_extractor(self, extractor: BaseFeatureExtractor) -> None:
        self.extractors.append(extractor)

    def extract_all(
        self,
        df: pd.DataFrame,
        include_original: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """提取所有特征并合并"""
        all_features: list[pd.DataFrame] = []
        feature_categories: dict[str, list[str]] = {}
        seen_columns: set[str] = set(df.columns) if include_original else set()

        if include_original:
            all_features.append(df)

        for extractor in tqdm(self.extractors, desc="特征提取"):
            print(f"  提取 {extractor.name} 特征...")
            features = extractor.extract(df, **kwargs)
            features = self._rename_duplicates(features, seen_columns)
            all_features.append(features)
            feature_categories[extractor.name] = list(features.columns)
            seen_columns.update(features.columns)

        feature_matrix = pd.concat(all_features, axis=1)

        self.feature_info = {
            "original_shape": df.shape,
            "feature_matrix_shape": feature_matrix.shape,
            "feature_categories": feature_categories,
        }
        return feature_matrix

    def _rename_duplicates(self, df: pd.DataFrame, seen: set[str]) -> pd.DataFrame:
        """重命名重复列"""
        cols = list(df.columns)
        new_cols = []
        for col in cols:
            if col in seen:
                suffix = 1
                new_col = f"{col}_{suffix}"
                while new_col in seen or new_col in new_cols:
                    suffix += 1
                    new_col = f"{col}_{suffix}"
                new_cols.append(new_col)
            else:
                new_cols.append(col)
        df.columns = new_cols
        return df

    def get_feature_info(self) -> dict[str, Any]:
        return self.feature_info
