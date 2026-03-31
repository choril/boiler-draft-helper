#!/usr/bin/env python3
"""
风机控制参数优化脚本

基于已训练的 LSTM 多步预测模型，使用高级优化算法找到最优控制参数。

支持优化方法:
1. bayesian (TPE) - 贝叶斯优化，适合有限评估预算
2. NSGA-II - 多目标优化，生成 Pareto 前沿
3. gradient - 梯度下降，基于 TensorFlow 自动微分
4. hybrid - 混合优化，自动选择最优算法
5. mpc - 滚动时域优化，适合实时控制
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

from src.utils.config import (
    PRESSURE_MAIN,
    OXYGEN_MAIN,
    CONTROL_PARAMS,
    EXPERT_RANGES,
)
from src.modeling.lstm import LSTM, setup_gpu
from src.modeling.optimization import (
    OptimizationConfig,
    OptimizationResult,
    SceneDetector,
    create_optimizer,
    ControlRecommender,
)
from src.features.selector import FeatureSelector
from src.utils.utils import print_section


# =============================================================================
# 数据和模型准备
# =============================================================================

def load_trained_model(
    model_path: Path,
    scaler_path: Path,
    target_scaler_path: Path,
    selected_features_path: Path,
    gpu_strategy: tf.distribute.Strategy,
) -> Tuple[LSTM, Any, Any, List[str]]:
    """加载已训练的模型和相关组件"""
    print_section("加载模型")

    # 加载特征选择结果
    with open(selected_features_path, "r") as f:
        selection_data = json.load(f)

    selected_features = selection_data.get("selected_features", [])
    if not selected_features:
        raise ValueError(f"未找到特征选择结果: {selected_features_path}")

    print(f"特征数量: {len(selected_features)}")

    # 加载标准化器
    scaler = joblib.load(scaler_path)
    target_scaler = joblib.load(target_scaler_path)
    print(f"标准化器加载成功")

    # 加载模型配置
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            model_config = json.load(f)
        print(f"模型配置: {model_config}")
    else:
        # 默认配置
        model_config = {
            "seq_length": 30,
            "n_features": len(selected_features),
            "output_steps": 10,
            "hidden_units": 128,
            "dropout_rate": 0.15,
            "learning_rate": 0.001,
            "l2_reg": 1e-4,
            "smoothness_weight": 0.001,
        }

    # 加载模型
    model = LSTM(
        seq_length=model_config.get("seq_length", 30),
        n_features=len(selected_features),
        output_steps=model_config.get("output_steps", 10),
        hidden_units=model_config.get("hidden_units", 128),
        dropout_rate=model_config.get("dropout_rate", 0.15),
        learning_rate=model_config.get("learning_rate", 0.001),
        l2_reg=model_config.get("l2_reg", 1e-4),
        strategy=gpu_strategy,
        feature_names=selected_features,
        smoothness_weight=model_config.get("smoothness_weight", 0.001),
    )

    # 加载权重
    keras_path = model_path / "model.keras"
    if keras_path.exists():
        model.model.load_weights(str(keras_path))
        print(f"模型权重加载成功: {keras_path}")
    else:
        raise FileNotFoundError(f"模型文件不存在: {keras_path}")

    print(f"模型输入形状: {model.model.input_shape}")
    print(f"模型输出形状: {model.model.output_shape}")

    return model, scaler, target_scaler, selected_features


def prepare_feature_data(
    feature_matrix: pd.DataFrame,
    selected_features: List[str],
    scaler: Any,
) -> FeatureSelector:
    """准备特征数据"""
    print_section("准备特征数据")

    target_vars = [PRESSURE_MAIN, OXYGEN_MAIN]

    selector = FeatureSelector(
        feature_matrix=feature_matrix,
        target_vars=target_vars,
    )
    selector.selected_features = selected_features
    selector.scaler = scaler

    # 添加目标变量
    for col in target_vars:
        if col in feature_matrix.columns and col not in selector.feature_matrix.columns:
            selector.feature_matrix[col] = feature_matrix[col]

    # 拟合目标标准化器（如果需要）
    if selector.target_scaler is None:
        selector.fit_scaler(target="targets")

    print(f"特征数据准备完成")
    print(f"  特征数: {len(selected_features)}")
    print(f"  目标变量: {target_vars}")

    return selector


# =============================================================================
# 优化测试
# =============================================================================

def run_single_optimization(
    optimizer: Any,
    recommender: ControlRecommender,
    feature_matrix: pd.DataFrame,
    selected_features: List[str],
    scene_info: Dict,
    seq_length: int,
    method: str = "hybrid",
) -> Optional[Dict]:
    """在单个场景上运行优化"""
    start_idx = scene_info["start_idx"]

    print(f"\n场景 [{start_idx}:{start_idx + seq_length}]")
    print(f"  负压均值: {scene_info['pressure_mean']:.2f} Pa, 标准差: {scene_info['pressure_std']:.2f}")
    print(f"  含氧量均值: {scene_info['oxygen_mean']:.2f}%, 标准差: {scene_info['oxygen_std']:.2f}")

    # 获取特征数据
    feature_data = feature_matrix[selected_features].iloc[
        start_idx:start_idx + seq_length
    ].copy()
    feature_data = feature_data.fillna(feature_data.median())
    features_orig = feature_data.values

    start_time = time.time()
    try:
        result = optimizer.optimize(features_orig, method=method)
        elapsed = time.time() - start_time

        print(f"  优化耗时: {elapsed:.3f}s")
        print(f"  评估次数: {result.n_evaluations}")
        print(f"  优化前损失: {result.loss_before:.4f}")
        print(f"  优化后损失: {result.loss_after:.4f}")
        print(f"  改善比例: {result.improvement_ratio:.1%}")
        print(f"  收敛状态: {'成功' if result.converged else '未收敛'}")

        print(f"  预测负压 (优化前): {result.predictions_before[:, 0].mean():.2f} Pa")
        print(f"  预测负压 (优化后): {result.predicted_pressure.mean():.2f} Pa")
        print(f"  预测含氧量 (优化前): {result.predictions_before[:, 1].mean():.2f}%")
        print(f"  预测含氧量 (优化后): {result.predicted_oxygen.mean():.2f}%")

        # 打印最优参数
        print("\n  【最优控制参数】")
        for param, value in result.optimal_values.items():
            adj = result.adjustments.get(param, 0)
            direction = "↑" if adj > 0 else "↓" if adj < 0 else "-"
            print(f"    {param}: {value:.2f} ({direction} {abs(adj):.2f})")

        recommendation = recommender.generate_recommendation(result, safety_margin=0.8)

        return {
            "scene_info": scene_info,
            "optimization_result": {
                "loss_before": result.loss_before,
                "loss_after": result.loss_after,
                "improvement_ratio": result.improvement_ratio,
                "converged": result.converged,
                "method": result.method,
                "elapsed_time": elapsed,
                "n_evaluations": result.n_evaluations,
            },
            "optimal_values": result.optimal_values,
            "adjustments": result.adjustments,
            "predictions": {
                "pressure_before": result.predictions_before[:, 0].tolist(),
                "pressure_after": result.predicted_pressure.tolist(),
                "oxygen_before": result.predictions_before[:, 1].tolist(),
                "oxygen_after": result.predicted_oxygen.tolist(),
            },
            "recommendation": recommendation,
            "pareto_front": result.pareto_front,
        }

    except Exception as e:
        print(f"  优化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_optimization_test(
    model: LSTM,
    scaler: Any,
    target_scaler: Any,
    selected_features: List[str],
    feature_matrix: pd.DataFrame,
    raw_data: pd.DataFrame,
    config: OptimizationConfig,
    output_dir: Path,
    scene_type: str = "volatile",
    top_k: int = 5,
    method: str = "hybrid",
) -> Dict:
    """运行优化测试"""
    print_section(f"场景测试: {scene_type}")

    # 场景检测
    detector = SceneDetector(
        pressure_col=PRESSURE_MAIN,
        oxygen_col=OXYGEN_MAIN,
        window_size=60,
    )
    scenes = detector.get_top_scenes(raw_data, scene_type, top_k)
    print(f"检测到 {len(scenes)} 个 {scene_type} 场景")

    if not scenes:
        print(f"未找到 {scene_type} 场景")
        return {"scene_type": scene_type, "results": []}

    # 创建优化器和推荐器
    optimizer = create_optimizer(
        method=method,
        model=model,
        scaler=scaler,
        target_scaler=target_scaler,
        feature_names=selected_features,
        config=config,
    )
    recommender = ControlRecommender(optimizer, config)

    # 测试每个场景
    all_results = []
    seq_length = model.seq_length

    for scene in scenes:
        result = run_single_optimization(
            optimizer=optimizer,
            recommender=recommender,
            feature_matrix=feature_matrix,
            selected_features=selected_features,
            scene_info=scene,
            seq_length=seq_length,
            method=method,
        )
        if result:
            all_results.append(result)

    return {
        "scene_type": scene_type,
        "n_scenes": len(scenes),
        "n_tested": len(all_results),
        "optimization_method": method,
        "results": all_results,
    }


def run_multi_objective_test(
    model: LSTM,
    scaler: Any,
    target_scaler: Any,
    selected_features: List[str],
    feature_matrix: pd.DataFrame,
    raw_data: pd.DataFrame,
    config: OptimizationConfig,
    output_dir: Path,
    top_k: int = 3,
) -> Dict:
    """运行多目标优化测试"""
    print_section("多目标优化测试 (NSGA-II)")

    # 场景检测
    detector = SceneDetector(
        pressure_col=PRESSURE_MAIN,
        oxygen_col=OXYGEN_MAIN,
        window_size=60,
    )
    scenes = detector.get_top_scenes(raw_data, "volatile", top_k)

    if not scenes:
        print("未找到波动场景")
        return {"results": []}

    # 创建多目标优化器
    optimizer = create_optimizer(
        method="NSGA-II",
        model=model,
        scaler=scaler,
        target_scaler=target_scaler,
        feature_names=selected_features,
        config=config,
    )

    all_results = []
    seq_length = model.seq_length

    for scene in scenes:
        start_idx = scene["start_idx"]
        print(f"\n场景 [{start_idx}:{start_idx + seq_length}]")

        feature_data = feature_matrix[selected_features].iloc[
            start_idx:start_idx + seq_length
        ].copy()
        feature_data = feature_data.fillna(feature_data.median())
        features_orig = feature_data.values

        result = optimizer.optimize(features_orig, return_pareto=True)

        print(f"  Pareto 前沿解数量: {len(result.pareto_front) if result.pareto_front else 0}")
        print(f"  最优解加权损失: {result.loss_after:.4f}")

        if result.pareto_front:
            print("\n  【Pareto 前沿分析】")
            for i, p in enumerate(result.pareto_front[:5]):
                print(f"    解{i+1}: 负压MSE={p['obj1']:.2f}, 含氧MSE={p['obj2']:.2f}")

        all_results.append({
            "scene_info": scene,
            "optimal_values": result.optimal_values,
            "predicted_pressure": result.predicted_pressure.tolist(),
            "predicted_oxygen": result.predicted_oxygen.tolist(),
            "pareto_front": result.pareto_front,
        })

    return {
        "scene_type": "volatile",
        "method": "NSGA-II",
        "results": all_results,
    }


# =============================================================================
# 可视化和报告
# =============================================================================

def visualize_results(test_results: Dict, output_dir: Path, scene_type: str):
    """可视化优化结果"""
    results = test_results["results"]
    if not results:
        return

    n_scenes = min(3, len(results))
    fig, axes = plt.subplots(n_scenes, 2, figsize=(14, 4 * n_scenes))
    if n_scenes == 1:
        axes = axes[np.newaxis, :]

    for i, result in enumerate(results[:n_scenes]):
        predictions = result["predictions"]
        opt = result["optimization_result"]

        # 负压预测
        ax1 = axes[i, 0]
        steps = range(1, len(predictions["pressure_before"]) + 1)

        ax1.plot(steps, predictions["pressure_before"], 'b-o', label='优化前', markersize=4)
        ax1.plot(steps, predictions["pressure_after"], 'r-s', label='优化后', markersize=4)
        ax1.axhline(y=-115, color='g', linestyle='--', label='目标值 (-115 Pa)')
        ax1.axhline(y=-150, color='gray', linestyle=':', alpha=0.5)
        ax1.axhline(y=-80, color='gray', linestyle=':', alpha=0.5)
        ax1.fill_between(steps, -150, -80, alpha=0.1, color='green')

        ax1.set_xlabel('预测步数')
        ax1.set_ylabel('负压 (Pa)')
        ax1.set_title(f'场景{i+1}: 负压预测 (改善: {opt["improvement_ratio"]:.1%})')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)

        # 含氧量预测
        ax2 = axes[i, 1]
        ax2.plot(steps, predictions["oxygen_before"], 'b-o', label='优化前', markersize=4)
        ax2.plot(steps, predictions["oxygen_after"], 'r-s', label='优化后', markersize=4)
        ax2.axhline(y=2.0, color='g', linestyle='--', label='目标值 (2.0%)')
        ax2.axhline(y=1.7, color='gray', linestyle=':', alpha=0.5)
        ax2.axhline(y=2.3, color='gray', linestyle=':', alpha=0.5)
        ax2.fill_between(steps, 1.7, 2.3, alpha=0.1, color='green')

        ax2.set_xlabel('预测步数')
        ax2.set_ylabel('含氧量 (%)')
        ax2.set_title(f'场景{i+1}: 含氧量预测')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = output_dir / f"optimization_results_{scene_type}.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"可视化结果已保存: {save_path}")


def visualize_pareto_front(pareto_results: Dict, output_dir: Path):
    """可视化 Pareto 前沿"""
    results = pareto_results["results"]
    if not results:
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['blue', 'green', 'red', 'orange', 'purple']

    for i, result in enumerate(results):
        pareto_front = result.get("pareto_front", [])
        if not pareto_front:
            continue

        obj1 = [p["obj1"] for p in pareto_front]
        obj2 = [p["obj2"] for p in pareto_front]

        ax.scatter(obj1, obj2, c=colors[i % len(colors)], s=50, alpha=0.7,
                   label=f'场景{i+1} Pareto前沿')

        # 标记最优解
        opt = result["optimal_values"]

    ax.set_xlabel('负压 MSE (目标1)')
    ax.set_ylabel('含氧量 MSE (目标2)')
    ax.set_title('多目标优化 Pareto 前沿')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = output_dir / "pareto_front.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Pareto 前沿可视化已保存: {save_path}")


def generate_summary_report(all_results: List[Dict], output_dir: Path) -> str:
    """生成汇总报告"""
    report = []
    report.append("# 风机控制参数优化测试报告\n\n")
    report.append(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 测试汇总表
    report.append("## 测试汇总\n\n")
    report.append("| 场景类型 | 优化方法 | 测试数 | 平均改善率 | 平均耗时 |\n")
    report.append("|----------|----------|--------|------------|----------|\n")

    for result in all_results:
        scene_type = result.get("scene_type", "unknown")
        method = result.get("optimization_method", result.get("method", "unknown"))
        n_tested = result.get("n_tested", len(result.get("results", [])))
        results = result.get("results", [])

        if results:
            avg_improvement = np.mean([
                r.get("optimization_result", {}).get("improvement_ratio", 0)
                for r in results
            ])
            avg_time = np.mean([
                r.get("optimization_result", {}).get("elapsed_time", 0)
                for r in results
            ])
            report.append(f"| {scene_type} | {method} | {n_tested} | {avg_improvement:.1%} | {avg_time:.3f}s |\n")
        else:
            report.append(f"| {scene_type} | {method} | {n_tested} | N/A | N/A |\n")

    # 最优控制参数汇总
    report.append("\n## 最优控制参数汇总\n\n")

    for result in all_results:
        if not result.get("results"):
            continue

        report.append(f"### {result.get('scene_type', 'unknown')} 场景 ({result.get('optimization_method', result.get('method', 'unknown'))})\n\n")

        for i, r in enumerate(result["results"][:3]):
            scene_info = r.get("scene_info", {})
            opt = r.get("optimization_result", {})
            opt_values = r.get("optimal_values", {})
            adjustments = r.get("adjustments", {})

            report.append(f"#### 场景 {i+1}\n\n")
            if scene_info:
                report.append(f"- **负压状态**: 均值 {scene_info.get('pressure_mean', 'N/A'):.2f} Pa\n")
                report.append(f"- **含氧量状态**: 均值 {scene_info.get('oxygen_mean', 'N/A'):.2f}%\n")
            if opt:
                report.append(f"- **优化效果**: 损失降低 {opt.get('improvement_ratio', 0):.1%}\n")
                report.append(f"- **评估次数**: {opt.get('n_evaluations', 'N/A')}\n\n")

            report.append("**最优控制参数**:\n\n")
            report.append("| 参数 | 最优值 | 调整量 |\n")
            report.append("|------|--------|--------|\n")

            for param, value in opt_values.items():
                adj = adjustments.get(param, 0)
                report.append(f"| {param} | {value:.2f} | {adj:+.2f} |\n")
            report.append("\n")

    report_text = "".join(report)

    report_path = output_dir / "optimization_test_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n报告已保存: {report_path}")

    return report_text


def print_optimal_parameters(all_results: List[Dict]):
    """打印最优参数组合"""
    print("\n" + "=" * 70)
    print("最优风机控制参数组合汇总")
    print("=" * 70)

    for result in all_results:
        if not result.get("results"):
            continue

        method = result.get("optimization_method", result.get("method", "unknown"))
        print(f"\n【{result.get('scene_type', 'unknown')} 场景 - {method}】")

        for i, r in enumerate(result["results"][:3]):
            opt_values = r.get("optimal_values", {})
            adjustments = r.get("adjustments", {})
            opt = r.get("optimization_result", {})

            improvement = opt.get("improvement_ratio", 0) if opt else 0

            print(f"\n  场景 {i+1} (改善率: {improvement:.1%}):")
            print("  " + "-" * 50)

            for param, value in opt_values.items():
                adj = adjustments.get(param, 0)
                direction = "↑" if adj > 0 else "↓" if adj < 0 else "-"
                print(f"    {param}: {value:.2f} ({direction} {abs(adj):.2f})")


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="风机控制参数优化测试")

    # 数据和模型路径
    parser.add_argument("--data", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--features", type=str, default="output/features/feature_matrix.feather")
    parser.add_argument("--model", type=str, default="output/models/lstm")
    parser.add_argument("--output-dir", type=str, default="output/optimization")

    # 优化参数
    parser.add_argument("--method", type=str, default="hybrid",
                        choices=["bayesian", "TPE", "NSGA-II", "gradient", "hybrid", "mpc"],
                        help="优化方法")
    parser.add_argument("--top-k", type=int, default=5, help="每个场景类型的测试数量")
    parser.add_argument("--scene-types", type=str, default="volatile,stable",
                        help="场景类型列表")

    # 目标范围
    parser.add_argument("--pressure-target", type=float, default=-115.0)
    parser.add_argument("--pressure-min", type=float, default=-150.0)
    parser.add_argument("--pressure-max", type=float, default=-80.0)
    parser.add_argument("--oxygen-target", type=float, default=2.0)
    parser.add_argument("--oxygen-min", type=float, default=1.7)
    parser.add_argument("--oxygen-max", type=float, default=2.3)

    # 优化配置
    parser.add_argument("--n-trials", type=int, default=50, help="贝叶斯优化试验次数")
    parser.add_argument("--max-iter", type=int, default=200, help="最大迭代次数")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 GPU
    print_section("初始化 GPU")
    gpu_strategy = setup_gpu(gpus="all", memory_growth=True)

    # 加载模型
    model_path = Path(args.model)
    model, scaler, target_scaler, selected_features = load_trained_model(
        model_path=model_path,
        scaler_path=model_path / "scaler.pkl",
        target_scaler_path=model_path / "target_scaler.pkl",
        selected_features_path=model_path / "selected_features.json",
        gpu_strategy=gpu_strategy,
    )

    # 加载特征数据
    print_section("加载特征数据")
    if Path(args.features).exists():
        feature_matrix = pd.read_feather(args.features)
    else:
        raise FileNotFoundError(f"特征矩阵不存在: {args.features}")

    print(f"特征矩阵维度: {feature_matrix.shape}")

    # 加载原始数据（用于场景检测）
    print_section("加载原始数据")
    raw_data = pd.read_feather(args.data)
    print(f"原始数据维度: {raw_data.shape}")

    # 优化配置
    config = OptimizationConfig(
        pressure_target=args.pressure_target,
        pressure_min=args.pressure_min,
        pressure_max=args.pressure_max,
        oxygen_target=args.oxygen_target,
        oxygen_min=args.oxygen_min,
        oxygen_max=args.oxygen_max,
        n_bayesian_trials=args.n_trials,
        max_iterations=args.max_iter,
    )

    # 运行优化测试
    scene_types = [s.strip() for s in args.scene_types.split(",")]
    all_results = []

    for scene_type in scene_types:
        result = run_optimization_test(
            model=model,
            scaler=scaler,
            target_scaler=target_scaler,
            selected_features=selected_features,
            feature_matrix=feature_matrix,
            raw_data=raw_data,
            config=config,
            output_dir=output_dir,
            scene_type=scene_type,
            top_k=args.top_k,
            method=args.method,
        )
        all_results.append(result)
        visualize_results(result, output_dir, scene_type)

    # 多目标优化测试（可选）
    if args.method in ["NSGA-II", "hybrid", "all"]:
        pareto_results = run_multi_objective_test(
            model=model,
            scaler=scaler,
            target_scaler=target_scaler,
            selected_features=selected_features,
            feature_matrix=feature_matrix,
            raw_data=raw_data,
            config=config,
            output_dir=output_dir,
            top_k=min(3, args.top_k),
        )
        all_results.append(pareto_results)
        visualize_pareto_front(pareto_results, output_dir)

    # 保存完整结果
    results_path = output_dir / "optimization_test_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n完整结果已保存: {results_path}")

    # 打印最优参数
    print_optimal_parameters(all_results)

    # 生成汇总报告
    generate_summary_report(all_results, output_dir)

    print_section("测试完成")


if __name__ == "__main__":
    main()