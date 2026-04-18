"""
工具函数模块

包含：
- 日志配置
- 文件读写
- 模型保存/加载
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """获取配置好的Logger

    Args:
        name: 日志名称（默认使用模块名）
        level: 日志级别

    Returns:
        logger: 配置好的Logger对象
    """
    if name is None:
        name = "predictor"

    logger = logging.getLogger(name)

    # 避免重复配置
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # 格式化
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    return logger


def save_json(data: dict, path: Path) -> None:
    """保存JSON文件"""
    import json
    import numpy as np

    def convert_to_serializable(obj):
        """转换为JSON可序列化类型"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj) if isinstance(obj, (np.float32, np.float64)) else int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(convert_to_serializable(data), f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict:
    """加载JSON文件"""
    import json

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_model(model, config, metrics, output_dir: Path, model_name: str = "model") -> None:
    """保存模型及相关文件

    Args:
        model: PyTorch模型
        config: 配置对象
        metrics: 评估指标
        output_dir: 输出目录
        model_name: 模型名称
    """
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存模型权重
    torch.save(model.state_dict(), output_dir / f"{model_name}.pt")

    # 保存配置
    save_json(config.to_dict(), output_dir / "config.json")

    # 保存指标
    save_json(metrics, output_dir / "metrics.json")

    get_logger().info(f"模型已保存至: {output_dir}")


def load_model(model_class, config_path: Path, device: str = "cuda"):
    """加载模型

    Args:
        model_class: 模型类
        config_path: 配置文件路径
        device: 加载设备

    Returns:
        model: 加载的模型
        config: 配置对象
    """
    import torch

    config_dict = load_json(config_path)
    # 从config重建Config对象（简化处理）
    model_path = config_path.parent / "model.pt"

    # 需要根据实际模型类接口调整
    model = model_class()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    return model, config_dict


__all__ = [
    "get_logger",
    "save_json",
    "load_json",
    "save_model",
    "load_model",
]