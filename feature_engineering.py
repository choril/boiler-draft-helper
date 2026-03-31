#!/usr/bin/env python3
"""
锅炉燃烧系统特征提取与选择主程序

完整流程:
1. 加载原始数据
2. 特征提取 (多维度特征工程)
3. 特征选择 (综合方法 + 因果导向)
4. 结果保存与可视化
"""
import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import PRESSURE_MAIN, OXYGEN_MAIN, CONTROL_PARAMS
from src.features import (
    FeatureExtractorPipeline,
    FeatureSelector,
    FeatureAnalysisVisualizer,
    create_balanced_pipeline
)
from src.utils.utils import print_section


def load_raw_data(data_path: Path) -> pd.DataFrame:
    print_section("加载预处理后的数据")
    print(f"数据路径: {data_path}")

    if data_path.suffix == ".feather":
        df = pd.read_feather(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"不支持的文件格式: {data_path.suffix}")

    print(f"数据维度: {df.shape}")
    return df


def extract_features(
    df: pd.DataFrame,
    output_dir: Path,
    force_recompute: bool = False,
) -> pd.DataFrame:
    """特征提取"""
    print_section("特征提取")

    feature_path = output_dir / "features" / "feature_matrix.feather"

    if feature_path.exists() and not force_recompute:
        print(f"加载已有特征矩阵: {feature_path}")
        feature_matrix = pd.read_feather(feature_path)
        print(f"特征维度: {feature_matrix.shape}")
        return feature_matrix

    print("开始提取特征...")
    pipeline = create_balanced_pipeline()
    feature_matrix = pipeline.extract_all(df)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_feather(feature_path)
    print(f"特征矩阵已保存: {feature_path}")

    return feature_matrix


def select_features(
    feature_matrix: pd.DataFrame,
    output_dir: Path,
    k: int = 80,
    sample_size: int = 10000,
    must_have_features: list[str] | None = None,
) -> tuple[list[str], FeatureSelector]:
    """特征选择"""
    print_section("特征选择")

    target_vars = [PRESSURE_MAIN, OXYGEN_MAIN]
    selector = FeatureSelector(feature_matrix, target_vars=target_vars)

    selected_features = selector.select_optimized(
        k=k,
        sample_size=sample_size,
        remove_collinear=True,
        collinear_threshold=0.95,
        remove_low_var=True,
        var_threshold=0.01,
        must_have_features=must_have_features,
    )

    return selected_features, selector


def save_results(
    selector: FeatureSelector,
    selected_features: list[str],
    output_dir: Path,
):
    """保存结果"""
    print_section("保存结果")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存特征选择结果
    result_path = output_dir / "selected_features.json"
    results = {
        "selected_features": selected_features,
        "importance_scores": selector.importance_scores,
        "target_variables": selector.target_vars,
        "n_features": len(selected_features),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"特征选择结果已保存: {result_path}")

    # 保存标准化器
    import joblib
    joblib.dump(selector.scaler, output_dir / "scaler.pkl")
    joblib.dump(selector.target_scaler, output_dir / "target_scaler.pkl")
    print(f"标准化器已保存: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="特征工程")
    parser.add_argument("--data", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--output-dir", type=str, default="output/features")
    parser.add_argument("--k", type=int, default=80, help="目标特征数量")
    parser.add_argument("--sample-size", type=int, default=10000)
    parser.add_argument("--force", action="store_true", help="强制重新计算特征")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # 加载数据
    df = load_raw_data(Path(args.data))

    # 特征提取
    feature_matrix = extract_features(df, output_dir, force_recompute=args.force)

    # 特征选择（强制保留控制参数）
    selected_features, selector = select_features(
        feature_matrix,
        output_dir,
        k=args.k,
        sample_size=args.sample_size,
        must_have_features=CONTROL_PARAMS.copy(),
    )

    # 保存结果
    save_results(selector, selected_features, output_dir)

    print_section("完成")
    print(f"选择特征数: {len(selected_features)}")


if __name__ == "__main__":
    main()