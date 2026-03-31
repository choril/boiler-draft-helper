#!/usr/bin/env python3
"""
Transformer 多目标多步预测训练脚本
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from src.utils.config import PRESSURE_MAIN, OXYGEN_MAIN, CONTROL_PARAMS
from src.features.extractor import create_balanced_pipeline
from src.features.selector import FeatureSelector
from src.modeling.transformer import MultiStepTransformer, setup_gpu
from src.utils.utils import print_section


def main():
    parser = argparse.ArgumentParser(description="Transformer 多目标多步预测")
    parser.add_argument("--data", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--feature-path", type=str, default="output/features/feature_matrix.feather")
    parser.add_argument("--output-dir", type=str, default="output/models/transformer")
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--output-steps", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dff", type=int, default=512)
    parser.add_argument("--num-encoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k", type=int, default=80, help="特征数量")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化GPU
    print_section("初始化GPU")
    strategy = setup_gpu(gpus="all", memory_growth=True)

    # 加载数据
    print_section("加载数据")
    raw_data = pd.read_feather(args.data)
    print(f"数据维度: {raw_data.shape}")

    # 加载或创建特征矩阵
    feature_path = Path(args.feature_path)
    if feature_path.exists():
        feature_matrix = pd.read_feather(feature_path)
    else:
        pipeline = create_balanced_pipeline()
        feature_matrix = pipeline.extract_all(raw_data)
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        feature_matrix.to_feather(feature_path)

    print(f"特征维度: {feature_matrix.shape}")

    # 特征选择
    print_section("特征选择")
    target_vars = [PRESSURE_MAIN, OXYGEN_MAIN]
    selector = FeatureSelector(feature_matrix, target_vars=target_vars)

    selected_features = selector.select_optimized(
        k=args.k,
        sample_size=10000,
        must_have_features=CONTROL_PARAMS.copy(),
    )

    selector.fit_scaler(method="standard")
    selector.fit_scaler(target="targets")

    # 构建序列
    print_section("构建训练序列")
    X, y = selector.build_seq2seq_sequences(
        seq_length=args.seq_length,
        output_steps=args.output_steps,
    )

    # 划分数据集
    n = len(X)
    n_test = int(n * 0.15)
    n_val = int((n - n_test) * 0.15)

    X_test, y_test = X[-n_test:], y[-n_test:]
    X_val, y_val = X[-(n_test + n_val): -n_test], y[-(n_test + n_val): -n_test]
    X_train, y_train = X[: -(n_test + n_val)], y[: -(n_test + n_val)]

    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

    # 创建模型
    print_section("创建Transformer模型")
    model = MultiStepTransformer(
        seq_length=args.seq_length,
        n_features=len(selected_features),
        output_steps=args.output_steps,
        d_model=args.d_model,
        num_heads=args.num_heads,
        dff=args.dff,
        num_encoder_layers=args.num_encoder_layers,
        dropout_rate=args.dropout,
        strategy=strategy,
        feature_names=selected_features,
    )

    # 训练
    print_section("训练模型")
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=15,
        verbose=1,
    )

    # 评估
    print_section("模型评估")
    results = model.evaluate(X_test, y_test)

    # 保存
    print_section("保存模型")
    model.save(output_dir)

    # 保存特征和标准化器
    with open(output_dir / "selected_features.json", "w") as f:
        json.dump({
            "selected_features": selected_features,
            "target_variables": target_vars,
        }, f, indent=2)

    import joblib
    joblib.dump(selector.scaler, output_dir / "scaler.pkl")
    joblib.dump(selector.target_scaler, output_dir / "target_scaler.pkl")

    print(f"模型已保存: {output_dir}")
    print_section("完成")


if __name__ == "__main__":
    main()