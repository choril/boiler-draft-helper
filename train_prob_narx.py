#!/usr/bin/env python3
"""
概率NARX模型训练脚本

训练流程：
1. 用MPCWindowSampler加载数据、构建滑动窗口样本
2. 划分训练/验证/测试集
3. 创建ProbNARX模型（双头概率架构）
4. 训练：负压用Gaussian NLL，含氧用Huber Loss
5. 评估：CRPS/覆盖率（负压）+ MAE/RMSE（含氧）
6. 保存模型和标准化器

用法：
    python train_prob_narx.py
    python train_prob_narx.py --hidden_dim 256 --epochs 80 --batch_size 128
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.window_sampler import MPCWindowSampler
from src.data.mpc_dataset import split_narx_data, NARXDataset, create_data_loaders
from src.modeling.prob_narx import ProbNARX, ProbNARXLoss
from src.config.hyperparams import L, H, DATA_SPLIT_CONFIG
from src.config.variables import (
    TARGET_VARIABLES,
    CONTROL_VARIABLES,
    PRESSURE_VARIABLES,
    OXYGEN_VARIABLES,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="训练概率NARX模型")
    # 数据
    p.add_argument("--data_path", default="output/all_data_cleaned.feather")
    p.add_argument("--history_length", type=int, default=30)
    p.add_argument("--horizon", type=int, default=H)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--jump_threshold", type=float, default=200.0)
    # 模型
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--future_hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    # 损失
    p.add_argument("--oxygen_weight", type=float, default=1.0)
    p.add_argument("--sigma_reg_weight", type=float, default=0.001)
    p.add_argument("--safety_weight", type=float, default=0.01)
    # 训练
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    # 输出
    p.add_argument("--output_dir", default="output/models/prob_narx")
    return p.parse_args()


def build_data(args):
    """构建数据集"""
    logger.info("=" * 60)
    logger.info("构建数据...")

    sampler = MPCWindowSampler(
        data_path=args.data_path,
        history_length=args.history_length,
        prediction_horizon=args.horizon,
        stride=args.stride,
        jump_threshold=args.jump_threshold,
        use_revin=False,  # 概率模型不需要RevIN，σ头自动适应尺度
    )

    encoder_input, decoder_input, target, info = (
        sampler.build_samples_with_future_control(
            scale=True,
            detect_jump=True,
            jump_threshold=args.jump_threshold,
            jump_mode="stratified",
            jump_weight=0.3,
        )
    )

    logger.info(f"encoder_input: {encoder_input.shape}")
    logger.info(f"decoder_input: {decoder_input.shape}")
    logger.info(f"target: {target.shape}")

    # 划分数据
    train_data, val_data, test_data = split_narx_data(
        encoder_input,
        decoder_input,
        target,
        train_ratio=DATA_SPLIT_CONFIG["train_ratio"],
        val_ratio=DATA_SPLIT_CONFIG["val_ratio"],
        shuffle_train=True,
    )

    # 划分sample_weights
    n_total = len(encoder_input)
    train_end = int(n_total * DATA_SPLIT_CONFIG["train_ratio"])
    train_weights = info["sample_weights"][:train_end]

    return train_data, val_data, test_data, info, sampler, train_weights


def create_loaders(train_data, val_data, test_data, train_weights, args):
    """创建DataLoader"""
    # 训练集：用NARXDataset兼容已有接口
    train_ds = NARXDataset(
        train_data["encoder_input"],
        train_data["decoder_input"],
        train_data["target"],
        teacher_forcing_ratio=0.0,  # 概率模型不用teacher forcing
    )
    val_ds = NARXDataset(
        val_data["encoder_input"],
        val_data["decoder_input"],
        val_data["target"],
        teacher_forcing_ratio=0.0,
    )
    test_ds = NARXDataset(
        test_data["encoder_input"],
        test_data["decoder_input"],
        test_data["target"],
        teacher_forcing_ratio=0.0,
    )

    # 加权采样器
    from torch.utils.data import WeightedRandomSampler

    # split_narx_data可能做了shuffle，但weights对应原始顺序
    # 这里直接用uniform weights（stratified已在sampler层面处理过）
    # 或者我们构建自己的weighted sampler
    sample_weights_tensor = torch.from_numpy(train_weights).float()
    # split_narx_data做了shuffle，所以weights和数据的对应关系已打乱
    # 最安全的做法：用均匀权重（stratified模式下sampler已降低了跳变样本的出现频率）
    # 但如果split_narx_data已经shuffle了，我们不能再对应weights
    # 所以这里不用weighted sampler，直接shuffle
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


def setup_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("使用CPU")
    return device


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    loss_accum = {
        "pressure_nll": 0.0,
        "oxygen_huber": 0.0,
        "sigma_mean": 0.0,
    }
    n_batches = 0

    for batch in loader:
        enc_in = batch["encoder_input"].to(device)
        dec_in = batch["decoder_input"].to(device)
        target = batch["target"].to(device)

        pred = model(enc_in, dec_in)
        losses = criterion(pred, target)

        optimizer.zero_grad()
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += losses["total"].item()
        for k in loss_accum:
            if k in losses:
                loss_accum[k] += losses[k].item()
        n_batches += 1

    avg = {k: v / n_batches for k, v in loss_accum.items()}
    avg["total"] = total_loss / n_batches
    return avg


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loss_accum = {
        "pressure_nll": 0.0,
        "oxygen_huber": 0.0,
        "sigma_mean": 0.0,
    }
    n_batches = 0

    all_mu, all_sigma, all_ohat, all_target = [], [], [], []

    for batch in loader:
        enc_in = batch["encoder_input"].to(device)
        dec_in = batch["decoder_input"].to(device)
        target = batch["target"].to(device)

        pred = model(enc_in, dec_in)
        losses = criterion(pred, target)

        total_loss += losses["total"].item()
        for k in loss_accum:
            if k in losses:
                loss_accum[k] += losses[k].item()
        n_batches += 1

        all_mu.append(pred["pressure_mu"].cpu())
        all_sigma.append(pred["pressure_sigma"].cpu())
        all_ohat.append(pred["oxygen"].cpu())
        all_target.append(target.cpu())

    avg = {k: v / n_batches for k, v in loss_accum.items()}
    avg["total"] = total_loss / n_batches

    # 拼接所有预测
    mu = torch.cat(all_mu, dim=0)
    sigma = torch.cat(all_sigma, dim=0)
    o_hat = torch.cat(all_ohat, dim=0)
    targets = torch.cat(all_target, dim=0)

    # 计算额外指标
    n_p = mu.shape[-1]
    y_p = targets[:, :, :n_p]
    y_o = targets[:, :, n_p:]

    # 负压覆盖率（预测区间[μ-2σ, μ+2σ]包含真实值的比例）
    in_interval = ((y_p >= mu - 2 * sigma) & (y_p <= mu + 2 * sigma)).float()
    avg["pressure_coverage_95"] = in_interval.mean().item()

    # 负压MAE（μ vs 真实值）
    avg["pressure_mu_mae"] = (mu - y_p).abs().mean().item()

    # 含氧量MAE
    avg["oxygen_mae"] = (o_hat - y_o).abs().mean().item()

    return avg


def train(args):
    device = setup_device()

    # 构建数据
    train_data, val_data, test_data, info, sampler, train_weights = build_data(args)
    train_loader, val_loader, test_loader = create_loaders(
        train_data, val_data, test_data, train_weights, args
    )

    # 模型维度
    input_dim = info["encoder_input_shape"][2]  # n_y + n_u + n_x
    control_dim = info["decoder_input_shape"][2]  # n_u
    n_pressure = len(PRESSURE_VARIABLES)
    n_oxygen = len(OXYGEN_VARIABLES)

    logger.info(f"模型输入维度: input_dim={input_dim}, control_dim={control_dim}")
    logger.info(f"预测维度: 负压={n_pressure}, 含氧={n_oxygen}")

    # 创建模型
    model = ProbNARX(
        input_dim=input_dim,
        control_dim=control_dim,
        n_pressure=n_pressure,
        n_oxygen=n_oxygen,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        future_hidden=args.future_hidden,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,}")

    # 损失函数
    # 注意：安全约束范围需要转换到标准化空间
    # 标准化后 boundary不适用原始Pa值，先禁用或设很宽范围
    criterion = ProbNARXLoss(
        n_pressure=n_pressure,
        n_oxygen=n_oxygen,
        horizon=args.horizon,
        oxygen_weight=args.oxygen_weight,
        sigma_reg_weight=args.sigma_reg_weight,
        safety_weight=0.0,  # 标准化空间下禁用原始边界约束
    ).to(device)

    # 优化器 + 调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
    )

    # 训练循环
    best_val_loss = float("inf")
    patience_counter = 0
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info(f"epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    history = {"train": [], "val": []}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # --- 训练 ---
        model.train()
        train_loss = 0.0
        train_metrics = {"pressure_nll": 0.0, "oxygen_huber": 0.0, "sigma_mean": 0.0}
        n_batches = 0

        for batch in train_loader:
            enc_in = batch["encoder_input"].to(device)
            dec_in = batch["decoder_input"].to(device)
            target = batch["target"].to(device)

            pred = model(enc_in, dec_in)
            losses = criterion(pred, target)

            optimizer.zero_grad()
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()

            train_loss += losses["total"].item()
            for k in train_metrics:
                if k in losses:
                    train_metrics[k] += losses[k].item()
            n_batches += 1

        train_loss /= n_batches
        for k in train_metrics:
            train_metrics[k] /= n_batches

        # --- 验证 ---
        val_metrics = evaluate(model, val_loader, criterion, device)
        val_loss = val_metrics["total"]

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"p_nll={val_metrics['pressure_nll']:.4f} | "
            f"o_huber={val_metrics['oxygen_huber']:.4f} | "
            f"σ={val_metrics['sigma_mean']:.3f} | "
            f"p_cov95={val_metrics['pressure_coverage_95']:.3f} | "
            f"p_mae={val_metrics['pressure_mu_mae']:.4f} | "
            f"o_mae={val_metrics['oxygen_mae']:.4f} | "
            f"lr={lr_now:.6f} | "
            f"{elapsed:.1f}s"
        )

        history["train"].append({"epoch": epoch, "loss": train_loss, **train_metrics})
        history["val"].append({"epoch": epoch, **val_metrics})

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "info": {
                        k: v
                        for k, v in info.items()
                        if not isinstance(v, np.ndarray)
                    },
                },
                output_dir / "best_model.pt",
            )
            logger.info(f"  ★ 保存最佳模型 (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"早停于epoch {epoch} (patience={args.patience})")
                break

    # --- 测试集评估 ---
    logger.info("=" * 60)
    logger.info("测试集评估...")

    ckpt = torch.load(output_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)

    logger.info("测试集结果:")
    logger.info(f"  total_loss:     {test_metrics['total']:.4f}")
    logger.info(f"  pressure_nll:   {test_metrics['pressure_nll']:.4f}")
    logger.info(f"  oxygen_huber:   {test_metrics['oxygen_huber']:.4f}")
    logger.info(f"  sigma_mean:     {test_metrics['sigma_mean']:.3f}")
    logger.info(f"  p_coverage_95:  {test_metrics['pressure_coverage_95']:.3f}")
    logger.info(f"  p_mu_mae:       {test_metrics['pressure_mu_mae']:.4f}")
    logger.info(f"  o_mae:          {test_metrics['oxygen_mae']:.4f}")

    # 保存标准化器和配置
    scaler_params = sampler.get_scaler_params()
    with open(output_dir / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)

    results = {
        "best_epoch": ckpt["epoch"],
        "best_val_loss": float(best_val_loss),
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "args": vars(args),
        "n_params": n_params,
        "data_info": {
            k: v for k, v in info.items() if not isinstance(v, np.ndarray)
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # 保存训练历史
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"所有结果已保存到 {output_dir}")
    return model, test_metrics


if __name__ == "__main__":
    args = parse_args()
    train(args)
