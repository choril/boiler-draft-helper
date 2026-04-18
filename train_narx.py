#!/usr/bin/env python3
"""
NARX-LSTM模型训练脚本

训练流程：
1. 加载清洗后的数据
2. 检测原始数据中的负压跳变时间段（新增）
3. 构建MPC滑动窗口样本（跳过跳变区域）
4. 划分数据集
5. 创建NARX-LSTM模型
6. 训练模型（teacher forcing策略）
7. 评估和保存

改进：
- 跳变检测在原始数据级别完成，构建窗口时直接跳过这些区域
- 支持RevIN实例归一化（针对非平稳数据）
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path

from src.data.window_sampler import MPCWindowSampler
from src.data.mpc_dataset import (
    NARXDataset,
    create_data_loaders,
    split_narx_data,
    compute_sample_weights,
    create_weighted_sampler,
)
from src.data.revin import inverse_normalize_predictions  # 新增
from src.modeling.narx_lstm import create_narx_lstm_model, NARXLSTMTrainer
from src.config.hyperparams import NARX_LSTM_CONFIG, L, H
from src.utils.logger import get_logger

logger = get_logger(__name__)


def convert_to_json_serializable(obj):
    """将包含numpy数组的对象转换为JSON可序列化的格式

    Args:
        obj: 可能包含numpy数组的对象（dict, list, ndarray等）

    Returns:
        JSON可序列化的对象
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(description="训练NARX-LSTM模型")
    parser.add_argument("--data_path", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--output_dir", type=str, default="output/models/narx_lstm")
    parser.add_argument("--L", type=int, default=L, help="历史窗口长度")
    parser.add_argument("--H", type=int, default=H, help="预测步长")
    parser.add_argument("--epochs", type=int, default=100, help="训练epoch数")
    parser.add_argument("--batch_size", type=int, default=128, help="批大小")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
    parser.add_argument("--encoder_hidden", type=int, default=256, help="编码器隐藏层大小")
    parser.add_argument("--decoder_hidden", type=int, default=256, help="解码器隐藏层大小")
    parser.add_argument("--encoder_layers", type=int, default=2, help="编码器层数")
    parser.add_argument("--decoder_layers", type=int, default=2, help="解码器层数")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout率")
    parser.add_argument("--patience", type=int, default=15, help="早停patience")
    parser.add_argument("--teacher_forcing", type=float, default=0.5, help="Teacher forcing比例")
    parser.add_argument("--force_sample", action="store_true", help="强制重新构建样本")
    parser.add_argument("--physics_weight", type=float, default=0.2, help="物理约束损失权重")
    parser.add_argument("--sample_weight", action="store_true", help="对突变样本加权")
    parser.add_argument("--detect_jump", action="store_true", help="检测并跳过原始数据中的负压跳变时间段")
    parser.add_argument("--jump_threshold", type=float, default=200.0, help="跳变阈值(Pa)，超过此值的区域被跳过（默认200Pa，过滤故障级别跳变）")
    parser.add_argument("--device", type=str, default="cuda", help="训练设备 (cuda/cpu)")
    parser.add_argument("--use_revin", action="store_true", help="使用RevIN实例归一化（针对非平稳数据）")  # 新增

    args = parser.parse_args()

    # 设置设备
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")

    # 输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ========== 1. 加载数据和构建样本 ==========
    # 缓存文件名包含RevIN和跳变检测参数，确保参数变化时重新构建
    cache_suffix = ""
    if args.detect_jump:
        cache_suffix += f"_jump{args.jump_threshold}"
    if args.use_revin:
        cache_suffix += "_revin"
    sample_cache_path = output_dir / f"samples_cache{cache_suffix}.npz"

    if sample_cache_path.exists() and not args.force_sample:
        logger.info(f"加载缓存样本: {sample_cache_path}")
        cache = np.load(sample_cache_path)
        encoder_input = cache['encoder_input']
        decoder_input = cache['decoder_input']
        target = cache['target']
        info = json.loads(str(cache['info']))

        # 加载窗口统计量（如果有）
        window_stats = None
        if args.use_revin and 'y_mean' in cache:
            window_stats = {
                'y_mean': cache['y_mean'],
                'y_std': cache['y_std'],
            }

        # 加载标准化器参数（用于样本权重计算）
        scaler_params_path = output_dir / "scaler_params.json"
        if scaler_params_path.exists():
            with open(scaler_params_path, 'r') as f:
                scaler_params = json.load(f)
            y_scale = np.array(scaler_params['y']['scale'])
            y_mean = np.array(scaler_params['y']['mean'])
        else:
            logger.warning("未找到scaler_params.json，将使用默认值")
            y_scale = np.ones(7)
            y_mean = np.zeros(7)
    else:
        logger.info("构建MPC样本...")
        if sample_cache_path.exists():
            sample_cache_path.unlink()

        # 创建采样器，传入use_revin参数
        sampler = MPCWindowSampler(
            data_path=args.data_path,
            history_length=args.L,
            prediction_horizon=args.H,
            jump_threshold=args.jump_threshold,
            use_revin=args.use_revin,  # 新增
        )

        # 构建样本，传入use_revin参数
        encoder_input, decoder_input, target, info = sampler.build_samples_with_future_control(
            detect_jump=args.detect_jump,
            jump_threshold=args.jump_threshold,
            use_revin=args.use_revin,  # 新增
        )

        # 获取窗口统计量（RevIN模式）
        window_stats = info.get('window_stats', None)

        # 准备JSON序列化的info（转换numpy数组为列表）
        info_for_json = convert_to_json_serializable(info)

        # 保存缓存（包含窗口统计量）
        save_dict = {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'target': target,
            'info': json.dumps(info_for_json),
        }
        if args.use_revin and window_stats is not None:
            save_dict['y_mean'] = window_stats['y_mean']
            save_dict['y_std'] = window_stats['y_std']

        np.savez(sample_cache_path, **save_dict)
        logger.info(f"样本已缓存: {sample_cache_path}")

        # 保存标准化器参数
        scaler_params = sampler.get_scaler_params()
        with open(output_dir / "scaler_params.json", 'w') as f:
            json.dump(scaler_params, f, indent=2)

        # 提取Y标准化参数（用于样本权重计算）
        y_scale = np.array(scaler_params['y']['scale'])
        y_mean = np.array(scaler_params['y']['mean'])

    # ========== 2. 划分数据集 ==========
    logger.info("划分数据集...")

    train_data, val_data, test_data = split_narx_data(
        encoder_input, decoder_input, target,
        shuffle_train=True,
    )

    # 划分窗口统计量（RevIN模式）
    if args.use_revin and window_stats is not None:
        n_train = len(train_data['encoder_input'])
        n_val = len(val_data['encoder_input'])
        train_y_mean = window_stats['y_mean'][:n_train]
        train_y_std = window_stats['y_std'][:n_train]
        val_y_mean = window_stats['y_mean'][n_train:n_train+n_val]
        val_y_std = window_stats['y_std'][n_train:n_train+n_val]
        test_y_mean = window_stats['y_mean'][n_train+n_val:]
        test_y_std = window_stats['y_std'][n_train+n_val:]
    else:
        train_y_mean, train_y_std = None, None
        val_y_mean, val_y_std = None, None
        test_y_mean, test_y_std = None, None

    # 创建Dataset（传入窗口统计量）
    train_dataset = NARXDataset(
        train_data['encoder_input'],
        train_data['decoder_input'],
        train_data['target'],
        teacher_forcing_ratio=args.teacher_forcing,
        y_mean=train_y_mean,  # 新增
        y_std=train_y_std,    # 新增
    )
    val_dataset = NARXDataset(
        val_data['encoder_input'],
        val_data['decoder_input'],
        val_data['target'],
        teacher_forcing_ratio=0.0,
        y_mean=val_y_mean,
        y_std=val_y_std,
    )
    test_dataset = NARXDataset(
        test_data['encoder_input'],
        test_data['decoder_input'],
        test_data['target'],
        teacher_forcing_ratio=0.0,
        y_mean=test_y_mean,
        y_std=test_y_std,
    )

    # 样本权重重采样
    train_sampler = None
    if args.sample_weight and not args.use_revin:
        # 注意：RevIN模式下样本权重计算需要特殊处理（暂不支持）
        sample_weights = compute_sample_weights(
            encoder_input=train_data['encoder_input'],
            target=train_data['target'],
            y_scale=y_scale,
            y_mean=y_mean,
            pressure_change_threshold=40.0,  # Pa，原始单位
            high_change_weight=3.0,
        )
        train_sampler = create_weighted_sampler(train_dataset, sample_weights)
        logger.info("使用样本权重重采样，突变样本权重加倍")
    elif args.sample_weight and args.use_revin:
        logger.warning("RevIN模式下暂不支持样本权重重采样")

    # 创建DataLoader
    train_loader = create_data_loaders(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = create_data_loaders(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_loader = create_data_loaders(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    logger.info(f"数据集大小:")
    logger.info(f"  训练集: {len(train_dataset)}")
    logger.info(f"  验证集: {len(val_dataset)}")
    logger.info(f"  测试集: {len(test_dataset)}")

    # ========== 3. 创建模型 ==========
    n_y = info.get('n_y', 7)
    n_u = info.get('n_u', 7)
    n_x = encoder_input.shape[2] - n_y - n_u

    config = {
        'encoder_hidden_units': args.encoder_hidden,
        'decoder_hidden_units': args.decoder_hidden,
        'encoder_num_layers': args.encoder_layers,
        'decoder_num_layers': args.decoder_layers,
        'dropout_rate': args.dropout,
        'encoder_bidirectional': True,
        'output_steps': args.H,
        'step_weights': NARX_LSTM_CONFIG.get('step_weights', [1.0, 0.9, 0.8, 0.7, 0.6]),
    }

    model = create_narx_lstm_model(n_y, n_u, n_x, config)

    # ========== 4. 训练 ==========
    # RevIN模式下禁用物理约束（暂时）
    physics_weight = 0.0 if args.use_revin else args.physics_weight
    if args.use_revin and args.physics_weight > 0:
        logger.warning("RevIN模式下暂时禁用物理约束，后续版本将重新集成")

    trainer = NARXLSTMTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
        step_weights=config['step_weights'],
        physics_weight=physics_weight,
    )

    if physics_weight > 0:
        logger.info(f"使用物理约束损失，权重: {physics_weight}")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=args.patience,
        teacher_forcing_ratio=args.teacher_forcing,
        verbose=True,
    )

    # ========== 5. 评估 ==========
    logger.info("评估模型...")
    metrics = trainer.evaluate(test_loader)

    logger.info("测试集评估指标:")
    for step, step_metrics in metrics['per_step'].items():
        logger.info(f"  步骤{step}: RMSE={step_metrics['rmse']:.4f}, MAE={step_metrics['mae']:.4f}")
    logger.info(f"  总体: RMSE={metrics['overall']['rmse']:.4f}, MAE={metrics['overall']['mae']:.4f}")

    # RevIN模式下的反标准化评估（额外信息）
    if args.use_revin:
        logger.info("RevIN模式下评估指标为标准化空间的数值")
        logger.info("实际预测需要使用窗口统计量反标准化")

    # ========== 6. 保存 ==========
    logger.info("保存模型...")
    model_path = output_dir / "model.pt"
    torch.save(trainer.get_model_state_dict(), model_path)

    # 保存配置
    config_save = {
        'n_y': n_y,
        'n_u': n_u,
        'n_x': n_x,
        'L': args.L,
        'H': args.H,
        'encoder_hidden': args.encoder_hidden,
        'decoder_hidden': args.decoder_hidden,
        'encoder_layers': args.encoder_layers,
        'decoder_layers': args.decoder_layers,
        'dropout': args.dropout,
        'bidirectional': True,
        'physics_weight': physics_weight,
        'sample_weight': args.sample_weight,
        'detect_jump': args.detect_jump,
        'jump_threshold': args.jump_threshold if args.detect_jump else None,
        'jump_periods_count': info.get('jump_periods_count', 0) if args.detect_jump else 0,
        'use_revin': args.use_revin,  # 新增
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_save, f, indent=2)

    # 保存训练历史
    with open(output_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)

    # 保存评估结果
    with open(output_dir / "evaluation_metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    # 保存窗口统计量（RevIN模式）
    if args.use_revin and window_stats is not None:
        np.savez(
            output_dir / "window_stats.npz",
            y_mean=window_stats['y_mean'],
            y_std=window_stats['y_std'],
        )
        logger.info("窗口统计量已保存")

    logger.info(f"模型已保存至: {output_dir}")
    logger.info("训练完成！")


if __name__ == "__main__":
    main()