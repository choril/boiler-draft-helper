#!/usr/bin/env python3
"""
LSTM模型训练脚本

功能：
1. 特征提取与选择
2. 训练多步预测模型
3. 评估并保存模型

用于后续风机控制参数优化模型。
"""

import argparse
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from src.features.extractor import create_extractor_pipeline
from src.features.selector import FeatureSelector
from src.modeling.lstm import LSTM
from src.utils.utils import setup_gpu
from src.utils.logger import get_logger
from src.utils.config import MAIN_TARGETS, CONTROL_PARAMS

logger = get_logger(__name__)


def extract_features(df: pd.DataFrame, feature_path: Path, force_extract: bool = False) -> pd.DataFrame:
    """提取特征"""
    if feature_path.exists() and not force_extract:
        logger.info(f"加载已有特征矩阵: {feature_path}")
        return pd.read_feather(feature_path)

    logger.info("提取特征")
    pipeline = create_extractor_pipeline()
    feature_df = pipeline.extract_all(df)

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_feather(feature_path)
    logger.info(f"特征矩阵已保存: {feature_path}")

    return feature_df


def select_features(
    selection_path: Path,
    feature_df: pd.DataFrame,
    target_vars: list[str],
    force_select: bool = False,
    k: int = 80,
    sample_size: int = 10000,
    use_granger: bool = False,
    must_have_features: list[str] | None = None,
) -> tuple[FeatureSelector, list[str]]:
    """特征选择"""
    selector = FeatureSelector(
        feature_matrix=feature_df,
        target_vars=target_vars
    )

    if selection_path.exists() and not force_select:
        logger.info(f"加载已有特征选择结果: {selection_path}")
        selector.load_results(selection_path)
        return selector, selector.selected_features
        
    logger.info("特征选择")
    selected = selector.select(
        k=k,
        sample_size=sample_size,
        use_granger=use_granger,
        must_have_features=must_have_features,
    )
    # 检查控制参数保留情况
    logger.info("控制参数保留情况:")
    control_in_selected = [c for c in must_have_features if c in selected]
    control_not_selected = [c for c in must_have_features if c not in selected]
    logger.info(f"  已保留 ({len(control_in_selected)}): {control_in_selected}")
    if control_not_selected:
        logger.info(f"  未保留 ({len(control_not_selected)}): {control_not_selected}")

    # 拟合标准化器
    selector.fit_scaler(target="features")
    selector.fit_scaler(target="targets")
    
    logger.info("特征选择完成")

    return selector, selected


def build_sequences(
    selector: FeatureSelector,
    seq_length: int,
    output_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """构建训练序列"""
    logger.info("构建序列")

    X, y = selector.build_seq2seq_sequences(
        seq_length=seq_length,
        output_steps=output_steps,
        scale_features=True,
        scale_targets=True,
    )

    return X, y


def split_data(X, y, train_ratio=0.7, val_ratio=0.15):
    """划分数据集"""
    logger.info("划分数据集")
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    logger.info(f"\n数据划分:")
    logger.info(f"  训练集: {len(X_train)}")
    logger.info(f"  验证集: {len(X_val)}")
    logger.info(f"  测试集: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def evaluate_model(model: LSTM, X_test, y_test, selector):
    """评估模型"""
    logger.info("模型评估")

    results = model.evaluate(X_test, y_test, target_scaler=selector.target_scaler)

    # 添加控制参数信息
    results["control_params"] = [c for c in CONTROL_PARAMS if c in model.feature_names]

    return results


def main():
    parser = argparse.ArgumentParser(description="训练LSTM模型")
    parser.add_argument("--data_dir", type=str, default="output")
    parser.add_argument("--output_dir", type=str, default="output/models/lstm")
    parser.add_argument("--force_extract", action="store_true", help="强制重新提取特征")
    parser.add_argument("--force_select", action="store_true", help="强制重新进行特征选择")
    parser.add_argument("--sample_size", type=int, default=10000, help="特征选择采样数量")
    parser.add_argument("--use_granger", action="store_true", help="使用Granger因果检验进行特征选择")
    parser.add_argument("--seq_length", type=int, default=30)
    parser.add_argument("--output_steps", type=int, default=10)
    parser.add_argument("--n_features", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hidden_units", type=int, default=128)
    parser.add_argument("--patience", type=int, default=15, help="早停耐心值")

    args = parser.parse_args()

    # 初始化GPU
    logger.info("初始化GPU")
    gpu_strategy = setup_gpu(gpus="all", memory_growth=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    logger.info("加载数据")
    data_path = Path(args.data_dir) / "all_data_cleaned.feather"
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    df = pd.read_feather(data_path)
    logger.info(f"数据维度: {df.shape}")

    # 提取特征
    feature_path = Path(args.data_dir) / "features" / "feature_matrix.feather"
    feature_df = extract_features(df, feature_path, args.force_extract)

    # 添加目标变量
    for col in MAIN_TARGETS:
        if col in df.columns and col not in feature_df.columns:
            feature_df[col] = df[col]

    # 特征选择
    target_vars = MAIN_TARGETS
    selection_path = output_dir / "selected_features.json"
    selector, selected = select_features(
        selection_path,
        feature_df,
        target_vars,
        force_select=args.force_select,
        k=args.n_features,
        sample_size=args.sample_size,
        use_granger=args.use_granger,
        must_have_features=CONTROL_PARAMS.copy(),
    )
    
    # 保存特征选择结果
    selector.save_results(selection_path)
    
    # 保存标准化器
    joblib.dump(selector.scaler, output_dir / "scaler.pkl")
    joblib.dump(selector.target_scaler, output_dir / "target_scaler.pkl")
    
    # 构建序列
    X, y = build_sequences(selector, args.seq_length, args.output_steps)

    # 划分数据
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    # 构建模型
    logger.info("构建模型")
    model = LSTM(
        seq_length=args.seq_length,
        n_features=len(selected),
        output_steps=args.output_steps,
        hidden_units=args.hidden_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        l2_reg=1e-4,
        strategy=gpu_strategy,
        feature_names=selected,
        smoothness_weight=0.001,
    )

    model.summary()

    # 训练
    model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        verbose=1,
    )

    # 评估
    results = evaluate_model(model, X_test, y_test, selector)

    # 保存模型
    logger.info("保存模型")
    model.save(output_dir)

    # 保存评估结果
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n模型保存路径: {output_dir}")
    logger.info(f"特征数量: {len(selected)}")
    logger.info(f"控制参数数量: {len(results.get('control_params', []))}")
    logger.info("训练完成")


if __name__ == "__main__":
    main()