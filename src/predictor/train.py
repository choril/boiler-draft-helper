"""
训练脚本

使用方式：
    # 基础训练
    python src/predictor/train.py --data_path data/processed/train_data.csv

    # 启用特征工程
    python src/predictor/train.py --data_path data/processed/train_data.csv --use_features

    # 自定义配置
    python src/predictor/train.py --data_path data/processed/train_data.csv \
        --epochs 100 --learning_rate 0.001 --batch_size 32 \
        --hidden_size 128 --num_layers 2
"""

import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
if project_root not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
import torch
import numpy as np
from typing import Optional

from src.predictor import (
    Config,
    BoilerDataset,
    BoilerPredictor,
    create_model,
    get_logger,
    save_json,
)
from src.predictor.loss import create_loss_fn
from src.predictor.trainer import Trainer

logger = get_logger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="锅炉预测模型训练")

    # 数据参数
    parser.add_argument("--data_path", type=str, required=True,
                        help="训练数据路径")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")

    # 特征工程参数
    parser.add_argument("--use_features", type=bool, default=False,
                        help="是否启用特征工程")
    parser.add_argument("--select_k", type=int, default=120,
                        help="特征选择保留数量")

    # 模型参数
    parser.add_argument("--L", type=int, default=15,
                        help="历史窗口长度")
    parser.add_argument("--H", type=int, default=5,
                        help="预测步数")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="LSTM隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="LSTM层数")
    parser.add_argument("--bidirectional", type=bool, default=True,
                        help="是否使用双向LSTM")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout率")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="最大训练轮数")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch大小")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="权重衰减")
    parser.add_argument("--early_stop_patience", type=int, default=10,
                        help="早停耐心值")
    parser.add_argument("--diff_weight", type=float, default=0.1,
                        help="差分损失权重")

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="outputs/predictor",
                        help="输出目录")
    parser.add_argument("--model_name", type=str, default="boiler_predictor",
                        help="模型名称")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")

    return parser.parse_args()


def create_config_from_args(args) -> Config:
    """从命令行参数创建配置"""
    config = Config()

    # 窗口配置
    config.window.history_length = args.L
    config.window.prediction_horizon = args.H

    # 目标/控制变量数量（固定）
    config.n_y = 7  # TARGET_VARS 数量
    config.n_u = 7  # CONTROL_VARS 数量
    config.n_x = 38  # 状态变量数量（默认）

    # 模型配置
    config.model.hidden_size = args.hidden_size
    config.model.num_layers = args.num_layers
    config.model.bidirectional = args.bidirectional
    config.model.dropout = args.dropout

    # 训练配置
    config.train.epochs = args.epochs
    config.train.batch_size = args.batch_size
    config.train.learning_rate = args.learning_rate
    config.train.weight_decay = args.weight_decay
    config.train.early_stop_patience = args.early_stop_patience

    # 损失配置
    config.loss.diff_weight = args.diff_weight

    return config


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """主训练流程"""
    args = parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # 创建配置
    config = create_config_from_args(args)

    logger.info("=" * 50)
    logger.info("锅炉预测模型训练")
    logger.info("=" * 50)
    logger.info(f"数据路径: {args.data_path}")
    logger.info(f"历史窗口 L={config.L}, 预测步数 H={config.H}")
    logger.info(f"模型: hidden={config.model.hidden_size}, layers={config.model.num_layers}")
    logger.info(f"训练: epochs={config.train.epochs}, batch={config.train.batch_size}, lr={config.train.learning_rate}")
    logger.info(f"特征工程: {args.use_features}")

    # 创建数据集
    logger.info("创建数据集...")
    dataset = BoilerDataset(
        data_path=args.data_path,
        config=config,
        use_feature_extraction=args.use_features,
        use_feature_selection=args.use_features,
        feature_selection_k=args.select_k,
    )

    # 获取数据加载器
    train_loader, val_loader, test_loader = dataset.get_loaders(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        batch_size=config.train.batch_size,
    )

    # 动态更新 n_x（在get_loaders之后，因为特征选择在其中执行）
    config.n_x = dataset.n_x
    logger.info(f"状态变量维度: n_x={config.n_x}")

    logger.info(f"数据集大小: 训练={len(train_loader.dataset)}, 验证={len(val_loader.dataset)}, 测试={len(test_loader.dataset)}")

    # 创建模型
    logger.info("创建模型...")
    model = create_model(config)

    # 创建损失函数
    loss_fn = create_loss_fn(config)

    # 创建训练器
    trainer = Trainer(model, config, loss_fn)

    # 训练
    logger.info("开始训练...")
    history = trainer.fit(train_loader, val_loader)

    # 评估测试集
    logger.info("评估测试集...")
    test_metrics = trainer.evaluate(test_loader, inverse_transform=True)

    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trainer.save(output_dir, args.model_name)

    # 保存测试指标
    save_json(test_metrics, output_dir / "test_metrics.json")

    # 打印最终结果
    logger.info("=" * 50)
    logger.info("训练完成")
    logger.info("=" * 50)
    logger.info(f"最佳验证损失: {trainer.best_val_loss:.4f}")
    logger.info(f"测试MSE: {test_metrics['mse']:.4f}")
    logger.info(f"测试差分准确率: {test_metrics['diff_accuracy']:.4f}")
    if 'rmse_original' in test_metrics:
        logger.info(f"测试RMSE(原始尺度): {test_metrics['rmse_original']:.4f}")
        logger.info(f"测试MAE(原始尺度): {test_metrics['mae_original']:.4f}")
    logger.info(f"模型保存至: {output_dir}")


if __name__ == "__main__":
    main()