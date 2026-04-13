#!/usr/bin/env python3
"""
优化模型效果验证脚本

从测试集截取数据片段，运行优化，人工比对效果。
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from src.modeling.lstm import LSTM
from src.modeling.optimization import (
    OptimizationConfig,
    create_optimizer,
    ControlRecommender,
)
from src.utils.config import PRESSURE_MAIN, OXYGEN_MAIN, CONTROL_PARAMS
from src.utils.utils import setup_gpu
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_model_and_data(model_path: Path, data_path: Path, feature_path: Path):
    """加载模型和数据"""
    # 初始化GPU
    gpu_strategy = setup_gpu(gpus="all", memory_growth=True)

    # 使用 LSTM.load 方法加载完整模型
    model = LSTM.load(model_path, strategy=gpu_strategy)

    # 加载特征选择结果
    with open(model_path / "selected_features.json", "r") as f:
        selection_data = json.load(f)
    selected_features = selection_data.get("selected_features", [])

    # 加载标准化器
    scaler = joblib.load(model_path / "scaler.pkl")
    target_scaler = joblib.load(model_path / "target_scaler.pkl")

    # 加载特征矩阵
    feature_matrix = pd.read_feather(feature_path)

    # 加载原始数据（用于获取真实目标值）
    raw_data = pd.read_feather(data_path)

    # 添加目标变量到特征矩阵（如果缺失）
    for col in [PRESSURE_MAIN, OXYGEN_MAIN]:
        if col in raw_data.columns and col not in feature_matrix.columns:
            feature_matrix[col] = raw_data[col]

    return model, scaler, target_scaler, selected_features, feature_matrix, raw_data


def get_test_indices(
    feature_matrix: pd.DataFrame,
    seq_length: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple:
    """获取测试集对应的原始数据索引范围"""
    n_samples = len(feature_matrix) - seq_length
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))

    # 测试集对应的原始索引范围
    test_start_idx = val_end
    test_end_idx = n_samples

    return test_start_idx, test_end_idx


def extract_test_segment(
    feature_matrix: pd.DataFrame,
    selected_features: List[str],
    test_start_idx: int,
    test_end_idx: int,
    seq_length: int,
    segment_idx: int,
) -> tuple:
    """从测试集截取一个数据片段"""
    # 确保片段在测试集范围内
    max_segments = test_end_idx - test_start_idx - seq_length + 1
    if segment_idx >= max_segments:
        raise ValueError(f"片段索引 {segment_idx} 超出测试集范围 (最大 {max_segments})")

    # 计算原始数据索引
    start_idx = test_start_idx + segment_idx
    end_idx = start_idx + seq_length

    # 提取特征数据
    feature_data = feature_matrix[selected_features].iloc[start_idx:end_idx].copy()
    features_orig = feature_data.values

    # 提取目标变量真实值（用于对比）
    pressure_true = feature_matrix[PRESSURE_MAIN].iloc[start_idx:end_idx].values
    oxygen_true = feature_matrix[OXYGEN_MAIN].iloc[start_idx:end_idx].values

    # 提取未来10步的真实目标值（如果存在）
    future_end = min(end_idx + seq_length // 3, len(feature_matrix))  # 约10步
    pressure_future = feature_matrix[PRESSURE_MAIN].iloc[end_idx:future_end].values
    oxygen_future = feature_matrix[OXYGEN_MAIN].iloc[end_idx:future_end].values

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "features_orig": features_orig,
        "pressure_true": pressure_true,
        "oxygen_true": oxygen_true,
        "pressure_future": pressure_future,
        "oxygen_future": oxygen_future,
        "feature_data": feature_data,
    }


def run_optimization(
    model: LSTM,
    scaler,
    target_scaler,
    selected_features: List[str],
    features_orig: np.ndarray,
    config: OptimizationConfig,
    method: str = "bayesian",
) -> dict:
    """运行优化"""
    optimizer = create_optimizer(
        method=method,
        model=model,
        scaler=scaler,
        target_scaler=target_scaler,
        feature_names=selected_features,
        config=config,
    )

    # 获取当前控制参数值
    current_values = optimizer.get_current_control_values(features_orig)

    # 预测原始效果
    features_scaled = scaler.transform(features_orig)
    features_batch = features_scaled[np.newaxis, :, :]
    pred_scaled = model.model(features_batch, training=False).numpy()
    pred_before = target_scaler.inverse_transform(pred_scaled[0])

    # 运行优化
    result = optimizer.optimize(features_orig, method=method)

    # 生成推荐
    recommender = ControlRecommender(optimizer, config)
    recommendation = recommender.generate_recommendation(result, safety_margin=0.8)

    return {
        "optimization_result": result,
        "recommendation": recommendation,
        "current_values": current_values,
        "pred_before": pred_before,
        "pred_after": np.column_stack([result.predicted_pressure, result.predicted_oxygen]),
    }


def display_comparison(segment: dict, opt_result: dict, selected_features: List[str]):
    """展示对比结果"""
    print("\n" + "=" * 80)
    print("优化效果验证 - 人工比对")
    print("=" * 80)

    # 1. 基本信息
    print(f"\n【数据片段信息】")
    print(f"  原始索引范围: [{segment['start_idx']}, {segment['end_idx']})")
    print(f"  片段长度: {len(segment['features_orig'])} 步")

    # 2. 原始目标变量
    print(f"\n【原始目标变量 (最近30步真实值)】")
    print(f"  负压 ({PRESSURE_MAIN}):")
    print(f"    均值: {segment['pressure_true'].mean():.2f} Pa")
    print(f"    标准差: {segment['pressure_true'].std():.2f} Pa")
    print(f"    范围: [{segment['pressure_true'].min():.2f}, {segment['pressure_true'].max():.2f}] Pa")
    print(f"  含氧量 ({OXYGEN_MAIN}):")
    print(f"    均值: {segment['oxygen_true'].mean():.2f} %")
    print(f"    标准差: {segment['oxygen_true'].std():.2f} %")
    print(f"    范围: [{segment['oxygen_true'].min():.2f}, {segment['oxygen_true'].max():.2f}] %")

    # 3. 未来真实值（如果有）
    if len(segment['pressure_future']) > 0:
        print(f"\n【未来真实值 (用于验证预测)】")
        print(f"  未来步数: {len(segment['pressure_future'])} 步")
        print(f"  负压真实值: {segment['pressure_future']}")
        print(f"  含氧量真实值: {segment['oxygen_future']}")

    # 4. 优化前的控制参数
    print(f"\n【优化前控制参数 (最后时刻值)】")
    current_values = opt_result["current_values"]
    control_param_names = [c for c in CONTROL_PARAMS if c in selected_features]

    # 获取控制参数的原始值
    feature_data = segment['feature_data']
    for i, ctrl in enumerate(control_param_names):
        if ctrl in feature_data.columns:
            # 显示整个片段的变化趋势
            values = feature_data[ctrl].values
            print(f"  {ctrl}:")
            print(f"    当前值: {values[-1]:.2f}")
            print(f"    片段均值: {values.mean():.2f}")
            print(f"    片段范围: [{values.min():.2f}, {values.max():.2f}]")

    # 5. 优化后的控制参数
    print(f"\n【优化后控制参数】")
    result = opt_result["optimization_result"]
    for param, value in result.optimal_values.items():
        adj = result.adjustments.get(param, 0)
        direction = "↑" if adj > 0 else "↓" if adj < 0 else "-"
        print(f"  {param}:")
        print(f"    优化值: {value:.2f}")
        print(f"    调整量: {adj:+.2f} ({direction})")

    # 6. 预测对比
    print(f"\n【预测效果对比】")
    pred_before = opt_result["pred_before"]
    pred_after = opt_result["pred_after"]

    print(f"\n  负压预测:")
    print(f"    优化前预测: {pred_before[:, 0]}")
    print(f"    优化后预测: {pred_after[:, 0]}")
    if len(segment['pressure_future']) > 0:
        print(f"    真实值:     {segment['pressure_future'][:len(pred_before[:, 0])]}")

    print(f"\n  含氧量预测:")
    print(f"    优化前预测: {pred_before[:, 1]}")
    print(f"    优化后预测: {pred_after[:, 1]}")
    if len(segment['oxygen_future']) > 0:
        print(f"    真实值:     {segment['oxygen_future'][:len(pred_before[:, 1])]}")

    # 7. 优化效果统计
    print(f"\n【优化效果统计】")
    print(f"  损失变化: {result.loss_before:.4f} → {result.loss_after:.4f}")
    print(f"  改善比例: {result.improvement_ratio:.1%}")
    print(f"  评估次数: {result.n_evaluations}")
    print(f"  耗时: {result.elapsed_time:.3f}s")

    # 8. 详细数据表格
    print(f"\n【详细数据表格 - 控制参数】")
    print("-" * 80)

    # 表头
    header = f"{'参数名':<15} {'片段起始值':<12} {'片段终止值':<12} {'优化值':<12} {'调整量':<12} {'变化幅度%':<12}"
    print(header)
    print("-" * 80)

    for ctrl in control_param_names:
        if ctrl in feature_data.columns:
            values = feature_data[ctrl].values
            start_val = values[0]
            end_val = values[-1]
            opt_val = result.optimal_values.get(ctrl, end_val)
            adj = result.adjustments.get(ctrl, 0)
            change_pct = abs(adj / end_val * 100) if end_val != 0 else 0

            print(f"{ctrl:<15} {start_val:<12.2f} {end_val:<12.2f} {opt_val:<12.2f} {adj:<+12.2f} {change_pct:<12.1f}")

    print("-" * 80)

    # 9. 预测与真实值对比表格
    if len(segment['pressure_future']) > 0:
        print(f"\n【预测与真实值对比表格】")
        print("-" * 80)
        header = f"{'步数':<8} {'负压预测前':<15} {'负压预测后':<15} {'负压真实':<15} {'含氧预测前':<15} {'含氧预测后':<15} {'含氧真实':<15}"
        print(header)
        print("-" * 80)

        n_steps = min(len(pred_before), len(segment['pressure_future']))
        for i in range(n_steps):
            p_pred_b = pred_before[i, 0]
            p_pred_a = pred_after[i, 0]
            p_true = segment['pressure_future'][i]
            o_pred_b = pred_before[i, 1]
            o_pred_a = pred_after[i, 1]
            o_true = segment['oxygen_future'][i]

            print(f"{i+1:<8} {p_pred_b:<15.2f} {p_pred_a:<15.2f} {p_true:<15.2f} {o_pred_b:<15.2f} {o_pred_a:<15.2f} {o_true:<15.2f}")

        print("-" * 80)

        # 计算预测误差
        print(f"\n【预测误差分析】")
        p_mae_before = np.mean(np.abs(pred_before[:n_steps, 0] - segment['pressure_future'][:n_steps]))
        p_mae_after = np.mean(np.abs(pred_after[:n_steps, 0] - segment['pressure_future'][:n_steps]))
        o_mae_before = np.mean(np.abs(pred_before[:n_steps, 1] - segment['oxygen_future'][:n_steps]))
        o_mae_after = np.mean(np.abs(pred_after[:n_steps, 1] - segment['oxygen_future'][:n_steps]))

        print(f"  负压 MAE (优化前): {p_mae_before:.2f} Pa")
        print(f"  负压 MAE (优化后): {p_mae_after:.2f} Pa")
        print(f"  负压 MAE 改善: {(p_mae_before - p_mae_after) / p_mae_before * 100:.1f}%")
        print(f"  含氧量 MAE (优化前): {o_mae_before:.2f} %")
        print(f"  含氧量 MAE (优化后): {o_mae_after:.2f} %")
        print(f"  含氧量 MAE 改善: {(o_mae_before - o_mae_after) / o_mae_before * 100:.1f}%")

    print("\n" + "=" * 80)


def save_results(segment: dict, opt_result: dict, output_path: Path):
    """保存结果到文件"""
    result_data = {
        "segment_info": {
            "start_idx": segment["start_idx"],
            "end_idx": segment["end_idx"],
            "pressure_true_mean": float(segment["pressure_true"].mean()),
            "pressure_true_std": float(segment["pressure_true"].std()),
            "oxygen_true_mean": float(segment["oxygen_true"].mean()),
            "oxygen_true_std": float(segment["oxygen_true"].std()),
        },
        "optimization": {
            "loss_before": opt_result["optimization_result"].loss_before,
            "loss_after": opt_result["optimization_result"].loss_after,
            "improvement_ratio": opt_result["optimization_result"].improvement_ratio,
            "optimal_values": opt_result["optimization_result"].optimal_values,
            "adjustments": opt_result["optimization_result"].adjustments,
        },
        "predictions": {
            "pressure_before": opt_result["pred_before"][:, 0].tolist(),
            "pressure_after": opt_result["pred_after"][:, 0].tolist(),
            "oxygen_before": opt_result["pred_before"][:, 1].tolist(),
            "oxygen_after": opt_result["pred_after"][:, 1].tolist(),
            "pressure_future_true": segment["pressure_future"].tolist() if len(segment["pressure_future"]) > 0 else [],
            "oxygen_future_true": segment["oxygen_future"].tolist() if len(segment["oxygen_future"]) > 0 else [],
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_path}")


def main():
    # 配置路径 - 使用有完整文件的模型
    model_path = Path("output/models/lstm_120_no_granger")
    data_path = Path("output/all_data_cleaned.feather")
    feature_path = Path("output/features/feature_matrix.feather")
    output_dir = Path("output/optimization_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型和数据
    print("加载模型和数据...")
    model, scaler, target_scaler, selected_features, feature_matrix, raw_data = \
        load_model_and_data(model_path, data_path, feature_path)

    seq_length = model.seq_length

    # 获取测试集索引范围
    test_start, test_end = get_test_indices(feature_matrix, seq_length)
    print(f"测试集索引范围: [{test_start}, {test_end})")
    print(f"可用片段数: {test_end - test_start - seq_length + 1}")

    # 优化配置
    config = OptimizationConfig(
        pressure_target=-115.0,
        pressure_min=-150.0,
        pressure_max=-80.0,
        oxygen_target=2.0,
        oxygen_min=1.5,
        oxygen_max=2.5,
        n_bayesian_trials=50,
    )

    # 选择测试片段
    # 可以指定多个片段索引进行测试
    segment_indices = [0, 100, 500]  # 测试集起始、中间位置的片段

    for seg_idx in segment_indices:
        print(f"\n{'='*60}")
        print(f"测试片段 {seg_idx}")
        print(f"{'='*60}")

        try:
            # 截取数据片段
            segment = extract_test_segment(
                feature_matrix=feature_matrix,
                selected_features=selected_features,
                test_start_idx=test_start,
                test_end_idx=test_end,
                seq_length=seq_length,
                segment_idx=seg_idx,
            )

            # 运行优化
            opt_result = run_optimization(
                model=model,
                scaler=scaler,
                target_scaler=target_scaler,
                selected_features=selected_features,
                features_orig=segment["features_orig"],
                config=config,
                method="bayesian",
            )

            # 展示对比
            display_comparison(segment, opt_result, selected_features)

            # 保存结果
            save_path = output_dir / f"test_segment_{seg_idx}.json"
            save_results(segment, opt_result, save_path)

        except Exception as e:
            print(f"片段 {seg_idx} 处理失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()