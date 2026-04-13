#!/usr/bin/env python3
"""
MPC控制器运行脚本

运行模式：
1. 仿真测试：使用历史数据进行闭环仿真
2. 实时运行：连接实际控制系统（需要接口）

仿真流程：
1. 加载代理模型
2. 加载测试数据
3. 滚动执行MPC
4. 记录控制序列和预测
5. 评估整体性能
"""

import argparse
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from src.modeling.pinn_proxy import PINNProxyMLP
from src.mpc.controller import MPCController, create_mpc_controller
from src.mpc.safety_monitor import SafetyLevel
from src.config.hyperparams import MPC_CONFIG, L, H
from src.config.constraints import PRESSURE_TARGET, OXYGEN_TARGET
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_proxy_model(model_path: str, config_path: str, device: torch.device) -> PINNProxyMLP:
    """加载代理模型"""
    with open(config_path, 'r') as f:
        config = json.load(f)

    model = PINNProxyMLP(
        input_dim=config['input_dim'],
        output_dim=config['output_dim'],
        hidden_layers=config['hidden_layers'],
        dropout_rate=config['dropout'],
    )

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    logger.info(f"代理模型已加载: {model_path}")
    logger.info(f"  输入维度: {config['input_dim']}")
    logger.info(f"  输出维度: {config['output_dim']}")

    return model


def run_simulation(
    controller: MPCController,
    test_data: np.ndarray,
    n_y: int,
    n_u: int,
    n_steps: int = 100,
) -> dict:
    """运行仿真测试

    Args:
        controller: MPC控制器
        test_data: 测试数据 (N, n_features)
        n_y: 目标变量维度
        n_u: 控制变量维度
        n_steps: 仿真步数

    Returns:
        results: 仿真结果字典
    """
    logger.info(f"开始仿真测试，步数: {n_steps}")

    # 提取数据
    L = controller.L
    H = controller.H

    # 结果记录
    executed_controls = []
    predictions = []
    actual_values = []
    costs = []
    safety_levels = []
    optimization_times = []

    # 初始化历史窗口
    first_row = test_data[0]
    n_features = test_data.shape[1]

    # 历史窗口初始化
    history_window = np.tile(first_row, (L, 1))

    # 当前控制初始化
    current_control = first_row[:n_u]

    # 仿真循环
    for step in range(min(n_steps, len(test_data) - L - H)):
        # 获取当前真实值
        current_row = test_data[step + L]
        current_y = current_row[:n_y]
        current_u = current_row[n_y:n_y + n_u]
        current_x = current_row[n_y + n_u:]

        # 更新历史窗口（使用真实数据）
        history_window = test_data[step:step + L]

        # 执行MPC
        result = controller.run_step(
            current_y=current_y,
            current_u=current_control,  # 使用上一步的控制
            current_x=current_x,
            history_y=history_window[:, :n_y],
            history_u=history_window[:, n_y:n_y + n_u],
            history_x=history_window[:, n_y + n_u:],
        )

        # 记录结果
        executed_controls.append(result.executed_control)
        predictions.append(result.prediction)
        actual_values.append(test_data[step + L:step + L + H, :n_y])
        costs.append(result.cost)
        safety_levels.append(result.safety_result.level.name)
        optimization_times.append(result.optimization_time)

        # 更新控制（用于下一步）
        current_control = result.executed_control

        # 打印进度
        if (step + 1) % 10 == 0:
            logger.info(f"步骤 {step + 1}/{n_steps}")
            logger.info(f"  控制: {result.executed_control[:3]}... (前3个)")
            logger.info(f"  目标值: {result.cost:.4f}")
            logger.info(f"  安全等级: {result.safety_result.level.name}")

    # 计算整体性能
    actual_values = np.array(actual_values)  # (n_steps, H, n_y)
    predictions = np.array(predictions)

    # 负压偏差（相对于目标）
    pressure_actual = actual_values[:, :, :4].mean(axis=2)
    pressure_pred = predictions[:, :, :4].mean(axis=2)
    pressure_error = np.mean((pressure_actual - PRESSURE_TARGET) ** 2)

    # 含氧偏差
    oxygen_actual = actual_values[:, :, 4:7].mean(axis=2)
    oxygen_pred = predictions[:, :, 4:7].mean(axis=2)
    oxygen_error = np.mean((oxygen_actual - OXYGEN_TARGET) ** 2)

    results = {
        'n_steps': n_steps,
        'executed_controls': np.array(executed_controls),
        'predictions': predictions,
        'actual_values': actual_values,
        'pressure_error': pressure_error,
        'oxygen_error': oxygen_error,
        'avg_cost': np.mean(costs),
        'avg_optimization_time': np.mean(optimization_times),
        'alarm_count': sum(1 for s in safety_levels if s == 'ALARM'),
        'safety_levels': safety_levels,
    }

    logger.info("仿真完成:")
    logger.info(f"  负压偏差: {pressure_error:.2f}")
    logger.info(f"  含氧偏差: {oxygen_error:.2f}")
    logger.info(f"  平均目标值: {np.mean(costs):.4f}")
    logger.info(f"  平均优化时间: {np.mean(optimization_times):.2f}s")
    logger.info(f"  告警次数: {results['alarm_count']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="运行MPC控制器")
    parser.add_argument("--proxy_model_path", type=str, default="output/models/proxy/proxy_model.pt")
    parser.add_argument("--proxy_config_path", type=str, default="output/models/proxy/proxy_config.json")
    parser.add_argument("--data_path", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--output_dir", type=str, default="output/mpc_results")
    parser.add_argument("--n_steps", type=int, default=100, help="仿真步数")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--n_evaluations", type=int, default=30, help="优化评估次数")

    args = parser.parse_args()

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"计算设备: {device}")

    # ========== 1. 加载代理模型 ==========
    model = load_proxy_model(
        args.proxy_model_path,
        args.proxy_config_path,
        device
    )

    # ========== 2. 创建MPC控制器 ==========
    mpc_config = {
        'n_evaluations': args.n_evaluations,
        'horizon': H,
        'pressure_weight': 1.0,
        'oxygen_weight': 1.0,
        'control_change_weight': 0.1,
    }

    controller = create_mpc_controller(model, device=device, config=mpc_config)

    # ========== 3. 加载测试数据 ==========
    logger.info(f"加载测试数据: {args.data_path}")
    if args.data_path.endswith('.feather'):
        df = pd.read_feather(args.data_path)
    else:
        df = pd.read_csv(args.data_path)

    # 处理数据
    df = df.drop(columns=['TIME', 'source_file'], errors='ignore')
    test_data = df.values.astype(np.float32)
    test_data = np.nan_to_num(test_data, nan=0.0)

    # 使用后半部分作为测试数据
    test_start = int(len(test_data) * 0.85)
    test_data = test_data[test_start:]

    logger.info(f"测试数据维度: {test_data.shape}")

    # ========== 4. 运行仿真 ==========
    n_y = 7
    n_u = 7

    results = run_simulation(
        controller=controller,
        test_data=test_data,
        n_y=n_y,
        n_u=n_u,
        n_steps=args.n_steps,
    )

    # ========== 5. 保存结果 ==========
    logger.info("保存结果...")

    # 保存控制序列
    pd.DataFrame(results['executed_controls']).to_csv(
        output_dir / "control_sequence.csv", index=False
    )

    # 保存摘要
    summary = {
        'n_steps': results['n_steps'],
        'pressure_error': results['pressure_error'],
        'oxygen_error': results['oxygen_error'],
        'avg_cost': results['avg_cost'],
        'avg_optimization_time': results['avg_optimization_time'],
        'alarm_count': results['alarm_count'],
        'alarm_ratio': results['alarm_count'] / results['n_steps'] if results['n_steps'] > 0 else 0,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"结果已保存至: {output_dir}")
    logger.info("MPC仿真完成！")


if __name__ == "__main__":
    main()