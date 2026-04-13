#!/usr/bin/env python3
"""
代理模型训练脚本

训练流程：
1. 加载清洗后的数据
2. 构建扁平化样本（用于MLP）
3. 可选：使用教师模型预测作为目标（蒸馏）
4. 训练轻量MLP代理模型
5. 测试推理速度
6. 评估和保存
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from src.data.window_sampler import MPCWindowSampler
from src.data.mpc_dataset import ProxyDataset, create_data_loaders, split_narx_data
from src.modeling.pinn_proxy import create_proxy_model, ProxyTrainer
from src.config.hyperparams import PROXY_MLP_CONFIG, L, H
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="训练代理模型")
    parser.add_argument("--data_path", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--output_dir", type=str, default="output/models/proxy")
    parser.add_argument("--narx_model_path", type=str, default=None, help="NARX模型路径（用于蒸馏）")
    parser.add_argument("--L", type=int, default=L, help="历史窗口长度")
    parser.add_argument("--H", type=int, default=H, help="预测步长")
    parser.add_argument("--epochs", type=int, default=30, help="训练epoch数")
    parser.add_argument("--batch_size", type=int, default=256, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--physics_weight", type=float, default=0.1, help="物理约束权重")
    parser.add_argument("--hidden_layers", type=str, default="256,128,64", help="隐藏层配置")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout率")
    parser.add_argument("--patience", type=int, default=5, help="早停patience")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="计算设备")
    parser.add_argument("--force_sample", action="store_true", help="强制重新构建样本")
    parser.add_argument("--speed_test", action="store_true", help="测试推理速度")

    args = parser.parse_args()

    # 解析隐藏层
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"计算设备: {device}")

    # ========== 1. 加载数据和构建样本 ==========
    sample_cache_path = output_dir / "proxy_samples_cache.npz"

    if sample_cache_path.exists() and not args.force_sample:
        logger.info(f"加载缓存样本: {sample_cache_path}")
        cache = np.load(sample_cache_path)
        encoder_input = cache['encoder_input']
        decoder_input = cache['decoder_input']
        target = cache['target']
        info = json.loads(str(cache['info']))
    else:
        logger.info("构建代理模型样本...")
        sampler = MPCWindowSampler(
            data_path=args.data_path,
            history_length=args.L,
            prediction_horizon=args.H,
        )
        encoder_input, decoder_input, target, info = sampler.build_samples_with_future_control()

        np.savez(
            sample_cache_path,
            encoder_input=encoder_input,
            decoder_input=decoder_input,
            target=target,
            info=json.dumps(info),
        )
        logger.info(f"样本已缓存: {sample_cache_path}")

    # ========== 2. 创建扁平化Dataset ==========
    logger.info("创建代理模型Dataset...")
    full_dataset = ProxyDataset(encoder_input, decoder_input, target)

    # 划分数据集
    n_samples = len(full_dataset)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

    train_dataset = torch.utils.data.Subset(full_dataset, range(train_end))
    val_dataset = torch.utils.data.Subset(full_dataset, range(train_end, val_end))
    test_dataset = torch.utils.data.Subset(full_dataset, range(val_end, n_samples))

    # 打乱训练集索引
    train_indices = np.random.permutation(train_end)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)

    # DataLoader
    train_loader = create_data_loaders(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = create_data_loaders(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = create_data_loaders(test_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"数据集大小:")
    logger.info(f"  训练集: {len(train_dataset)}")
    logger.info(f"  验证集: {len(val_dataset)}")
    logger.info(f"  测试集: {len(test_dataset)}")
    logger.info(f"  输入维度: {full_dataset.n_features_flat}")
    logger.info(f"  输出维度: {full_dataset.n_targets_flat}")

    # ========== 3. 创建模型 ==========
    config = {
        'hidden_layers': hidden_layers,
        'dropout_rate': args.dropout,
        'activation': 'relu',
    }

    n_y = 7
    n_u = 7
    n_x = encoder_input.shape[2] - n_y - n_u

    model = create_proxy_model(
        L=args.L,
        H=args.H,
        n_y=n_y,
        n_u=n_u,
        n_x=n_x,
        config=config,
    )

    # ========== 4. 训练 ==========
    trainer = ProxyTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        physics_weight=args.physics_weight,
        H=args.H,
        n_y=n_y,
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        verbose=True,
    )

    # ========== 5. 推理速度测试 ==========
    if args.speed_test:
        speed_metrics = trainer.measure_inference_speed(n_samples=1000)
        logger.info(f"推理速度:")
        logger.info(f"  批量1000样本: {speed_metrics['batch_time_ms']:.2f} ms")
        logger.info(f"  单样本: {speed_metrics['per_sample_us']:.2f} μs")

        # MPC要求：< 10ms单次推理
        if speed_metrics['per_sample_us'] < 10000:
            logger.info("  ✓ 满足MPC实时要求 (<10ms)")
        else:
            logger.warning("  ! 可能不满足MPC实时要求，考虑简化模型")

    # ========== 6. 评估 ==========
    logger.info("评估模型...")
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for X, Y in test_loader:
            X = X.to(device)
            pred = model(X).cpu().numpy()
            predictions.append(pred)
            targets.append(Y.numpy())

    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    # 计算指标
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)

    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
    }

    logger.info(f"测试集评估指标:")
    logger.info(f"  MSE: {mse:.4f}")
    logger.info(f"  MAE: {mae:.4f}")
    logger.info(f"  RMSE: {rmse:.4f}")

    # ========== 7. 保存 ==========
    logger.info("保存模型...")
    model_path = output_dir / "proxy_model.pt"
    torch.save(model.state_dict(), model_path)

    # 保存配置
    config_save = {
        'L': args.L,
        'H': args.H,
        'n_y': n_y,
        'n_u': n_u,
        'n_x': n_x,
        'input_dim': model.input_dim,
        'output_dim': model.output_dim,
        'hidden_layers': hidden_layers,
        'dropout': args.dropout,
    }
    with open(output_dir / "proxy_config.json", 'w') as f:
        json.dump(config_save, f, indent=2)

    # 保存训练历史
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # 保存评估结果
    with open(output_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"模型已保存至: {output_dir}")
    logger.info("训练完成！")


if __name__ == "__main__":
    main()