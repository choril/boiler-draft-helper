"""
损失权重调优指南

一、step_weights（各步预测权重）

问题：多步预测中，近期预测更重要还是远期？
权重设计：w_h = 1.0 - decay * h，近期权重高

调优方法：
1. 分析预测难度随步数增加的趋势
2. 计算各步预测的baseline误差（如均值预测）
3. 根据实际需求调整衰减率

示例分析脚本：
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
from typing import List, Dict
import json

from src.predictor.config import Config
from src.predictor.utils import get_logger

logger = get_logger(__name__)


def analyze_step_prediction_difficulty(
    data_path: str,
    config: Config,
    output_dir: str = "outputs/weight_analysis",
) -> Dict:
    """分析各步预测难度，为step_weights提供依据

    方法：
    1. 使用简单baseline（历史均值）预测各步
    2. 计算各步的MSE
    3. 误差越大的步，权重应该越高（因为更难预测，更需要关注）

    Args:
        data_path: 数据路径
        config: 配置对象
        output_dir: 输出目录

    Returns:
        analysis: 分析结果
    """
    import pandas as pd
    from src.predictor.dataset import BoilerDataset

    # 加载数据
    dataset = BoilerDataset(config, data_path=data_path)
    encoder_input, decoder_input, target, window_stats = dataset.build_samples()

    # 从encoder_input提取历史Y（前n_y维）
    y_hist = encoder_input[:, :, :config.n_y]  # (N, L, n_y)
    y_future = target  # (N, H, n_y)

    # Baseline预测：使用历史窗口最后时刻值
    baseline_pred = np.repeat(y_hist[:, -1:, :], config.H, axis=1)  # (N, H, n_y)

    # 计算各步MSE
    mse_per_step = []
    for h in range(config.H):
        mse_h = np.mean((baseline_pred[:, h, :] - y_future[:, h, :]) ** 2)
        mse_per_step.append(mse_h)

    # 计算各步变化幅度（预测难度）
    change_per_step = []
    for h in range(config.H):
        if h == 0:
            change = np.mean(np.abs(y_future[:, h, :] - y_hist[:, -1, :]))
        else:
            change = np.mean(np.abs(y_future[:, h, :] - y_future[:, h-1, :]))
        change_per_step.append(change)

    # 推荐权重方案
    # 方案1：基于MSE（预测难度）- MSE越大，权重越高
    mse_array = np.array(mse_per_step)
    mse_based_weights = mse_array / mse_array.sum() * config.H  # 归一化到总和=H

    # 方案2：线性衰减（默认方案，近期权重高）
    linear_weights = [1.0 - 0.1 * h for h in range(config.H)]

    # 方案3：反向权重（远期更难预测，权重高）
    reverse_weights = [0.1 * h + 0.5 for h in range(config.H)]

    # 方案4：基于变化幅度（变化大的步骤权重高）
    difficulty = np.array(change_per_step)
    change_based_weights = difficulty / difficulty.sum() * config.H  # 归一化

    # 转换为Python原生类型（JSON序列化）
    analysis = {
        'mse_per_step': [float(x) for x in mse_per_step],
        'change_per_step': [float(x) for x in change_per_step],
        'linear_weights': linear_weights,
        'reverse_weights': reverse_weights,
        'mse_based_weights': [float(x) for x in mse_based_weights],
        'change_based_weights': [float(x) for x in change_based_weights],
    }

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "step_weight_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info("各步预测难度分析:")
    logger.info(f"  MSE per step: {mse_per_step}")
    logger.info(f"  变化幅度 per step: {change_per_step}")
    logger.info(f"  线性衰减权重（默认）: {linear_weights}")
    logger.info(f"  MSE权重（远期高）: {[round(x, 2) for x in mse_based_weights]}")
    logger.info(f"  变化幅度权重: {[round(x, 2) for x in change_based_weights]}")

    return analysis


def analyze_differential_importance(
    data_path: str,
    config: Config,
    output_dir: str = "outputs/weight_analysis",
) -> Dict:
    """分析差分损失的重要性，为diff_weight提供依据

    方法：
    1. 分析数据的变化特性（跳变频率、幅度）
    2. 计算差分信号的统计特性
    3. 非平稳程度越高，diff_weight应该越大

    Args:
        data_path: 数据路径
        config: 配置对象

    Returns:
        analysis: 分析结果
    """
    import pandas as pd
    from src.predictor.dataset import BoilerDataset

    # 加载数据
    dataset = BoilerDataset(config, data_path=data_path)
    encoder_input, decoder_input, target, window_stats = dataset.build_samples()

    y_future = target  # (N, H, n_y)

    # 计算差分
    diff = y_future[:, 1:, :] - y_future[:, :-1, :]  # (N, H-1, n_y)

    # 差分统计
    diff_mean = np.mean(diff)
    diff_std = np.std(diff)
    diff_abs_mean = np.mean(np.abs(diff))

    # 跳变频率（差分超过阈值的比例）
    thresholds = {
        'pressure': 10.0,  # 负压跳变阈值 Pa
        'oxygen': 0.3,     # 含氧跳变阈值 %
    }

    # 负压跳变（前4列）
    pressure_diff = diff[:, :, :4]
    pressure_jump_ratio = np.mean(np.abs(pressure_diff) > thresholds['pressure'])

    # 含氧跳变（后3列）
    oxygen_diff = diff[:, :, 4:7]
    oxygen_jump_ratio = np.mean(np.abs(oxygen_diff) > thresholds['oxygen'])

    total_jump_ratio = (pressure_jump_ratio + oxygen_jump_ratio) / 2

    # 推荐diff_weight
    # 跳变比例越高，差分损失权重越大
    # 范围：0.05 ~ 0.3
    if total_jump_ratio < 0.1:
        recommended_diff_weight = 0.05  # 平稳数据
    elif total_jump_ratio < 0.2:
        recommended_diff_weight = 0.1   # 中等非平稳
    elif total_jump_ratio < 0.3:
        recommended_diff_weight = 0.15  # 较高非平稳
    else:
        recommended_diff_weight = 0.2   # 高度非平稳

    # 转换为Python原生类型
    analysis = {
        'diff_mean': float(diff_mean),
        'diff_std': float(diff_std),
        'diff_abs_mean': float(diff_abs_mean),
        'pressure_jump_ratio': float(pressure_jump_ratio),
        'oxygen_jump_ratio': float(oxygen_jump_ratio),
        'total_jump_ratio': float(total_jump_ratio),
        'recommended_diff_weight': float(recommended_diff_weight),
    }

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "diff_weight_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)

    logger.info("差分重要性分析:")
    logger.info(f"  差分均值: {diff_mean:.4f}")
    logger.info(f"  差分标准差: {diff_std:.4f}")
    logger.info(f"  负压跳变比例: {pressure_jump_ratio:.4f}")
    logger.info(f"  含氧跳变比例: {oxygen_jump_ratio:.4f}")
    logger.info(f"  推荐diff_weight: {recommended_diff_weight}")

    return analysis


def grid_search_weights(
    data_path: str,
    config: Config,
    weight_candidates: Dict[str, List],
    output_dir: str = "outputs/weight_search",
    max_epochs: int = 30,  # 快速搜索用较少epochs
) -> Dict:
    """网格搜索最佳权重组合

    Args:
        data_path: 数据路径
        config: 配置对象
        weight_candidates: 权重候选值
            {
                'step_weight_decays': [0.05, 0.1, 0.15],
                'diff_weights': [0.05, 0.1, 0.15, 0.2],
            }
        output_dir: 输出目录
        max_epochs: 快速训练epochs

    Returns:
        results: 搜索结果
    """
    import pandas as pd
    from src.predictor.dataset import BoilerDataset
    from src.predictor.model import create_model
    from src.predictor.loss import PredictionLoss
    from src.predictor.trainer import Trainer

    # 获取数据
    dataset = BoilerDataset(config, data_path=data_path)
    train_loader, val_loader, _ = dataset.get_loaders(
        val_ratio=0.2, test_ratio=0.1, batch_size=config.train.batch_size
    )

    results = []

    for decay in weight_candidates['step_weight_decays']:
        step_weights = [1.0 - decay * h for h in range(config.H)]

        for diff_weight in weight_candidates['diff_weights']:
            logger.info(f"测试权重: decay={decay}, diff_weight={diff_weight}")

            # 创建模型和损失
            model = create_model(config)
            loss_fn = PredictionLoss(
                step_weights=step_weights,
                diff_weight=diff_weight,
                horizon=config.H,
            )

            # 快速训练
            trainer = Trainer(model, config, loss_fn)
            history = trainer.fit(train_loader, val_loader, epochs=max_epochs)

            results.append({
                'step_weight_decay': decay,
                'step_weights': step_weights,
                'diff_weight': diff_weight,
                'best_val_loss': trainer.best_val_loss,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
            })

    # 找最佳组合
    best = min(results, key=lambda x: x['best_val_loss'])

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "weight_search_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 50)
    logger.info("权重搜索完成")
    logger.info(f"最佳组合:")
    logger.info(f"  step_weight_decay: {best['step_weight_decay']}")
    logger.info(f"  step_weights: {best['step_weights']}")
    logger.info(f"  diff_weight: {best['diff_weight']}")
    logger.info(f"  best_val_loss: {best['best_val_loss']:.4f}")

    return {
        'all_results': results,
        'best': best,
    }


if __name__ == "__main__":
    # 示例用法
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--mode", type=str, default="analyze",
                        choices=["analyze", "search"])
    args = parser.parse_args()

    config = Config()

    if args.mode == "analyze":
        # 分析模式：给出推荐权重
        analyze_step_prediction_difficulty(args.data_path, config)
        analyze_differential_importance(args.data_path, config)

    elif args.mode == "search":
        # 搜索模式：网格搜索最佳权重
        weight_candidates = {
            'step_weight_decays': [0.05, 0.1, 0.15],
            'diff_weights': [0.05, 0.1, 0.15, 0.2],
        }
        grid_search_weights(args.data_path, config, weight_candidates)