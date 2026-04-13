"""
控制参数生效时延分析工具

从历史数据中挖掘每个控制参数对目标变量的时延:
1. 互相关分析 (CCF): 快速扫描全局时延
2. 阶跃响应分析: 物理可解释的事件驱动分析
3. Granger 因果滞后: 统计显著性验证

输出: 每个控制参数 → 每个目标变量的时延表

用法:
    python -m src.analysis.delay_analyzer
    python -m src.analysis.delay_analyzer --max_lag 30 --top_k 10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.stattools import grangercausalitytests

from src.utils.config import (
    PRESSURE_VARIABLES,
    OXYGEN_VARIABLES,
    CONTROL_PARAMS,
    FAN_PARAMS,
)
from src.utils.logger import get_logger

logger = get_logger("delay_analyzer")


class DelayAnalyzer:
    """
    控制参数→目标变量 时延分析器

    三种互补方法:
    - 互相关函数 (CCF): 计算控制参数移位 τ 后与目标变量的相关性，峰值处为估计时延
    - 阶跃响应: 检测控制参数突变事件，对齐后平均目标响应，首次显著偏离处为时延
    - Granger 因果: 检验各滞后阶数的显著性，最显著阶数为时延
    """

    def __init__(
        self,
        data_path: str | Path = "output/all_data_cleaned.feather",
        target_cols: Optional[list[str]] = None,
        control_cols: Optional[list[str]] = None,
    ):
        self.target_cols = target_cols or [PRESSURE_VARIABLES[0], OXYGEN_VARIABLES[0]]
        self.control_cols = control_cols or CONTROL_PARAMS

        logger.info(f"加载数据: {data_path}")
        df = pd.read_feather(data_path)

        # 检查列可用性
        self.available_targets = [c for c in self.target_cols if c in df.columns]
        self.available_controls = [c for c in self.control_cols if c in df.columns]
        logger.info(f"目标变量: {self.available_targets}")
        logger.info(f"控制参数: {len(self.available_controls)} 个")

        # 提取数值 (去掉 NaN)
        self.target_data = df[self.available_targets].fillna(method="ffill").fillna(0).values
        self.control_data = df[self.available_controls].fillna(method="ffill").fillna(0).values

        # 控制参数名称映射 (便于阅读)
        self._build_name_map()

    def _build_name_map(self):
        """构建控制参数的可读名称映射"""
        self.name_map = {}
        reverse_map = {}
        for fan_type, params in FAN_PARAMS.items():
            for key, col in params.items():
                reverse_map[col] = f"{fan_type}_{key}"

        for col in self.available_controls:
            self.name_map[col] = reverse_map.get(col, col)

    # ----------------------------------------------------------
    # 方法1: 互相关函数
    # ----------------------------------------------------------

    def cross_correlation_analysis(
        self,
        max_lag: int = 30,
        sample_size: int = 50000,
    ) -> dict[str, dict[str, dict]]:
        """
        互相关分析: 控制参数移位 τ 后与目标变量的相关性。

        Args:
            max_lag: 最大滞后步数 (分钟)
            sample_size: 采样数量 (加速计算)

        Returns:
            {control_col: {target_col: {"best_lag": int, "best_corr": float, "corr_curve": list}}}
        """
        logger.info(f"互相关分析 (max_lag={max_lag}, sample_size={sample_size})")
        n = min(sample_size, len(self.target_data))
        start = len(self.target_data) - n

        results = {}
        for ci, ctrl_col in enumerate(self.available_controls):
            ctrl = self.control_data[start:, ci]
            results[ctrl_col] = {}

            for ti, tgt_col in enumerate(self.available_targets):
                tgt = self.target_data[start:, ti]

                # 去均值
                ctrl_c = ctrl - ctrl.mean()
                tgt_c = tgt - tgt.mean()

                # 互相关: 正 lag = ctrl 在前, tgt 在后
                corr = signal.correlate(tgt_c, ctrl_c, mode="full", method="auto")
                corr = corr / (len(ctrl_c) * ctrl_c.std() * tgt_c.std() + 1e-8)

                # 取正 lag 部分 (ctrl 领先于 tgt)
                center = len(ctrl_c) - 1
                pos_corr = corr[center: center + max_lag + 1]

                # 找峰值
                best_lag = int(np.argmax(np.abs(pos_corr)))
                best_corr = float(pos_corr[best_lag])

                results[ctrl_col][tgt_col] = {
                    "best_lag": best_lag,
                    "best_corr": best_corr,
                    "corr_curve": pos_corr.tolist(),
                }

        logger.info("互相关分析完成")
        return results

    # ----------------------------------------------------------
    # 方法2: 阶跃响应分析
    # ----------------------------------------------------------

    def step_response_analysis(
        self,
        change_threshold_sigma: float = 2.0,
        max_response_lag: int = 30,
        min_events: int = 20,
        sample_size: int = 100000,
    ) -> dict[str, dict[str, dict]]:
        """
        阶跃响应分析: 检测控制参数突变，对齐后平均目标响应。

        Args:
            change_threshold_sigma: 突变检测阈值 (标准差倍数)
            max_response_lag: 最大响应观察窗口 (分钟)
            min_events: 最少事件数 (少于此数不报告)
            sample_size: 采样数量

        Returns:
            {control_col: {target_col: {"delay": int, "n_events": int, "avg_response": list}}}
        """
        logger.info(f"阶跃响应分析 (threshold={change_threshold_sigma}σ, max_lag={max_response_lag})")
        n = min(sample_size, len(self.target_data))
        start = len(self.target_data) - n

        results = {}
        for ci, ctrl_col in enumerate(self.available_controls):
            ctrl = self.control_data[start:, ci]
            results[ctrl_col] = {}

            # 检测突变事件
            diff = np.diff(ctrl)
            threshold = diff.std() * change_threshold_sigma
            event_indices = np.where(np.abs(diff) > threshold)[0]

            # 过滤: 事件之间至少间隔 max_response_lag
            filtered_events = []
            last_event = -max_response_lag
            for idx in event_indices:
                if idx - last_event >= max_response_lag and idx + max_response_lag < n:
                    filtered_events.append(idx)
                    last_event = idx

            for ti, tgt_col in enumerate(self.available_targets):
                tgt = self.target_data[start:, ti]

                if len(filtered_events) < min_events:
                    results[ctrl_col][tgt_col] = {
                        "delay": None,
                        "n_events": len(filtered_events),
                        "avg_response": None,
                        "reason": f"事件数不足 ({len(filtered_events)} < {min_events})",
                    }
                    continue

                # 对齐事件，提取响应窗口
                responses = []
                for event_idx in filtered_events:
                    window = tgt[event_idx: event_idx + max_response_lag + 1]
                    if len(window) == max_response_lag + 1:
                        # 归一化: 相对于事件发生时的值
                        responses.append(window - window[0])

                if not responses:
                    continue

                avg_response = np.mean(responses, axis=0)  # (max_response_lag+1,)
                response_std = np.std(responses, axis=0)

                # 估计时延: 首次显著偏离 0 的位置
                baseline_std = response_std[0] if response_std[0] > 0 else tgt.std() * 0.01
                significant = np.abs(avg_response) > baseline_std * 0.5
                # 连续2步显著才算
                delay = max_response_lag  # 默认
                for t in range(1, len(significant) - 1):
                    if significant[t] and significant[t + 1]:
                        delay = t
                        break

                results[ctrl_col][tgt_col] = {
                    "delay": int(delay),
                    "n_events": len(responses),
                    "avg_response": avg_response.tolist(),
                    "response_std": response_std.tolist(),
                }

        logger.info("阶跃响应分析完成")
        return results

    # ----------------------------------------------------------
    # 方法3: Granger 因果滞后
    # ----------------------------------------------------------

    def granger_causality_analysis(
        self,
        max_lag: int = 15,
        sample_size: int = 3000,
        significance: float = 0.05,
    ) -> dict[str, dict[str, dict]]:
        """
        Granger 因果检验: 各滞后阶数的显著性。

        Args:
            max_lag: 最大滞后阶数
            sample_size: 样本量
            significance: 显著性水平

        Returns:
            {control_col: {target_col: {"best_lag": int, "n_significant": int, "pvalues": list}}}
        """
        logger.info(f"Granger 因果分析 (max_lag={max_lag}, sample_size={sample_size})")
        n = min(sample_size, len(self.target_data))
        start = len(self.target_data) - n

        results = {}
        for ci, ctrl_col in enumerate(self.available_controls):
            ctrl = self.control_data[start:, ci]
            results[ctrl_col] = {}

            for ti, tgt_col in enumerate(self.available_targets):
                tgt = self.target_data[start:, ti]

                # 准备数据: [target, control]
                test_data = np.column_stack([tgt, ctrl])

                # 平稳性检查
                if np.var(tgt) < 1e-10 or np.var(ctrl) < 1e-10:
                    results[ctrl_col][tgt_col] = {"best_lag": None, "reason": "方差过小"}
                    continue

                try:
                    gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)

                    pvalues = []
                    for lag in range(1, max_lag + 1):
                        p = gc_result[lag][0]["ssr_ftest"][1]
                        pvalues.append(float(p))

                    n_significant = sum(1 for p in pvalues if p < significance)
                    best_lag = int(np.argmin(pvalues)) + 1  # 1-indexed

                    results[ctrl_col][tgt_col] = {
                        "best_lag": best_lag,
                        "n_significant": n_significant,
                        "pvalues": pvalues,
                    }
                except Exception as e:
                    results[ctrl_col][tgt_col] = {"best_lag": None, "reason": str(e)[:100]}

        logger.info("Granger 因果分析完成")
        return results

    # ----------------------------------------------------------
    # 综合分析
    # ----------------------------------------------------------

    def analyze_all(
        self,
        max_lag: int = 30,
        output_path: str | Path = "output/delay_analysis.json",
    ) -> dict:
        """
        运行全部三种分析，综合出最终时延表。

        综合策略:
        - 如果 CCF 和阶跃响应一致 → 采信
        - 如果不一致 → 取较小值 (更保守，避免引入过长的延迟)
        - Granger 验证: 只有 Granger 显著的才纳入

        Returns:
            综合时延表 {control_col: {target_col: {"delay": int, "confidence": str, "details": dict}}}
        """
        logger.info("=" * 60)
        logger.info("开始综合时延分析")
        logger.info("=" * 60)

        # 三种方法
        ccf_results = self.cross_correlation_analysis(max_lag=max_lag)
        step_results = self.step_response_analysis(max_response_lag=max_lag)
        granger_results = self.granger_causality_analysis(max_lag=min(max_lag, 15))

        # 综合
        final_results = {}
        for ctrl_col in self.available_controls:
            ctrl_name = self.name_map.get(ctrl_col, ctrl_col)
            final_results[ctrl_col] = {"name": ctrl_name}

            for tgt_col in self.available_targets:
                ccf = ccf_results.get(ctrl_col, {}).get(tgt_col, {})
                step = step_results.get(ctrl_col, {}).get(tgt_col, {})
                granger = granger_results.get(ctrl_col, {}).get(tgt_col, {})

                delays = []
                confidences = []

                # CCF
                ccf_lag = ccf.get("best_lag")
                if ccf_lag is not None and ccf_lag > 0:
                    delays.append(ccf_lag)
                    confidences.append("ccf")

                # 阶跃响应
                step_lag = step.get("delay")
                if step_lag is not None:
                    delays.append(step_lag)
                    confidences.append("step")

                # Granger
                granger_lag = granger.get("best_lag")
                if granger_lag is not None and granger.get("n_significant", 0) >= 2:
                    delays.append(granger_lag)
                    confidences.append("granger")

                if delays:
                    # 综合时延: 取中位数
                    final_delay = int(np.median(delays))
                    confidence = "+".join(confidences)
                else:
                    final_delay = None
                    confidence = "insufficient_data"

                final_results[ctrl_col][tgt_col] = {
                    "delay": final_delay,
                    "confidence": confidence,
                    "ccf_lag": ccf_lag,
                    "step_lag": step_lag,
                    "granger_lag": granger_lag,
                    "granger_significant": granger.get("n_significant", 0),
                }

        # 保存
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        logger.info(f"时延分析结果已保存至 {output_path}")

        # 打印摘要
        self._print_summary(final_results)

        return final_results

    def _print_summary(self, results: dict):
        """打印可读的时延摘要"""
        target_names = {
            PRESSURE_VARIABLES[0]: "负压",
            OXYGEN_VARIABLES[0]: "氧量",
        }

        logger.info("\n" + "=" * 70)
        logger.info("控制参数生效时延摘要 (单位: 分钟)")
        logger.info("=" * 70)

        for ctrl_col, data in results.items():
            name = data.get("name", ctrl_col)
            for tgt_col in self.available_targets:
                tgt_name = target_names.get(tgt_col, tgt_col)
                info = data.get(tgt_col, {})
                delay = info.get("delay")
                conf = info.get("confidence", "")

                if delay is not None:
                    logger.info(f"  {name:40s} → {tgt_name}: {delay:3d} min  ({conf})")
                else:
                    reason = info.get("reason", "数据不足")
                    logger.info(f"  {name:40s} → {tgt_name}: N/A     ({reason})")

        logger.info("=" * 70)


# ============================================================
# 入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="控制参数时延分析")
    parser.add_argument("--data_path", type=str, default="output/all_data_cleaned.feather")
    parser.add_argument("--max_lag", type=int, default=30, help="最大滞后步数 (分钟)")
    parser.add_argument("--output", type=str, default="output/delay_analysis.json")
    args = parser.parse_args()

    analyzer = DelayAnalyzer(data_path=args.data_path)
    analyzer.analyze_all(max_lag=args.max_lag, output_path=args.output)


if __name__ == "__main__":
    main()
