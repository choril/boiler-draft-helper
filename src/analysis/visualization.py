from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.config import EXPERT_RANGES, FAN_EXPERT_RANGES, KEY_PARAMS, TARGET_VARIABLES

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class PlotGenerator:
    def __init__(
        self, df: pd.DataFrame, param_dict: dict[str, Any], output_dir: str = "output"
    ):
        self.df = df
        self.param_dict = param_dict
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_eda_plots(self) -> str:
        print("\n6. 生成可视化图表...")
        fig = plt.figure(figsize=(20, 12))
        self._plot_pressure_distribution(fig, 1)
        self._plot_oxygen_distribution(fig, 2)
        self._plot_coal_distribution(fig, 3)
        self._plot_correlation_heatmap(fig, 4)
        self._plot_coal_vs_pressure(fig, 5)
        self._plot_pressure_timeseries(fig, 6)
        plt.tight_layout()
        path = self.output_dir / "eda_plots.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  图表已保存: {path}")
        return str(path)

    def generate_analysis_plots(self) -> str:
        fig = plt.figure(figsize=(20, 16))
        self._plot_pressure_with_expert_range(fig, 1)
        self._plot_oxygen_with_target(fig, 2)
        self._plot_coal_with_expert_range(fig, 3)
        self._plot_key_params_correlation(fig, 4)
        self._plot_primary_air_vs_pressure(fig, 5)
        self._plot_coal_vs_pressure_colored(fig, 6)
        self._plot_pressure_timeseries_comparison(fig, 7)
        self._plot_control_variables_timeseries(fig, 8)
        self._plot_fan_current_comparison(fig, 9)
        plt.tight_layout()
        path = self.output_dir / "feature_analysis_plots.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"图表已保存至: {path}")
        return str(path)

    def _plot_pressure_distribution(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(2, 3, pos)
        if TARGET_VARIABLES[0] in self.df.columns:
            data = self.df[TARGET_VARIABLES[0]].dropna()
            ax.hist(data, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(
                x=-80, color="green", linestyle="--", linewidth=2, label="Ideal Range"
            )
            ax.axvline(x=-150, color="green", linestyle="--", linewidth=2)
            ax.axvspan(-150, -80, alpha=0.2, color="green")
            ax.set_xlabel("Furnace Pressure (Pa)")
            ax.set_ylabel("Frequency")
            ax.set_title("Furnace Pressure Distribution")
            ax.legend()

    def _plot_oxygen_distribution(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(2, 3, pos)
        if TARGET_VARIABLES[4] in self.df.columns:
            data = self.df[TARGET_VARIABLES[4]].dropna()
            ax.hist(data, bins=100, color="coral", alpha=0.7, edgecolor="white")
            ax.axvline(
                x=2.0, color="red", linestyle="--", linewidth=2, label="Target: 2.0%"
            )
            ax.axvline(x=1.7, color="orange", linestyle=":", linewidth=2)
            ax.axvline(x=2.3, color="orange", linestyle=":", linewidth=2)
            ax.set_xlabel("Oxygen Content (%)")
            ax.set_ylabel("Frequency")
            ax.set_title("Oxygen Content Distribution")
            ax.legend()

    def _plot_coal_distribution(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(2, 3, pos)
        if "D62AX002" in self.df.columns:
            data = self.df["D62AX002"].dropna()
            ax.hist(data, bins=100, color="forestgreen", alpha=0.7, edgecolor="white")
            ax.set_xlabel("Coal Feed Rate (t/h)")
            ax.set_ylabel("Frequency")
            ax.set_title("Coal Feed Rate Distribution")

    def _plot_correlation_heatmap(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(2, 3, pos)
        plot_cols = [c for c in KEY_PARAMS if c in self.df.columns][:8]
        if plot_cols:
            corr = self.df[plot_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(
                corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", ax=ax, center=0
            )
            ax.set_title("Key Parameters Correlation")

    def _plot_coal_vs_pressure(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(2, 3, pos)
        if "D62AX002" in self.df.columns and TARGET_VARIABLES[0] in self.df.columns:
            sample = self.df.iloc[::100]
            scatter = ax.scatter(
                sample["D62AX002"],
                sample[TARGET_VARIABLES[0]],
                c=sample.get("D61AX023", 0),
                cmap="Blues",
                alpha=0.5,
                s=10,
            )
            ax.set_xlabel("Coal Feed (t/h)")
            ax.set_ylabel("Pressure (Pa)")
            ax.set_title("Coal vs Pressure")
            plt.colorbar(scatter, ax=ax, label="Primary Air", shrink=0.8)

    def _plot_pressure_timeseries(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(2, 3, pos)
        if TARGET_VARIABLES[0] in self.df.columns:
            ts_sample = self.df.iloc[::500]
            ax.plot(
                range(len(ts_sample)),
                ts_sample[TARGET_VARIABLES[0]],
                linewidth=1,
                alpha=0.8,
            )
            ax.axhline(y=-80, color="green", linestyle="--", alpha=0.5)
            ax.axhline(y=-150, color="green", linestyle="--", alpha=0.5)
            ax.fill_between(range(len(ts_sample)), -150, -80, alpha=0.1, color="green")
            ax.set_xlabel("Time (sampled)")
            ax.set_ylabel("Pressure (Pa)")
            ax.set_title("Furnace Pressure Time Series")

    def _plot_pressure_with_expert_range(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if TARGET_VARIABLES[0] in self.df.columns:
            data = self.df[TARGET_VARIABLES[0]].dropna()
            ax.hist(data, bins=100, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(
                x=-80, color="green", linestyle="--", linewidth=2, label="Ideal Range"
            )
            ax.axvline(x=-150, color="green", linestyle="--", linewidth=2)
            ax.axvline(
                x=-20, color="orange", linestyle=":", linewidth=2, label="Normal Range"
            )
            ax.axvline(x=-230, color="orange", linestyle=":", linewidth=2)
            ax.axvspan(-150, -80, alpha=0.2, color="green", label="Optimal Zone")
            ax.set_xlabel("Furnace Pressure (Pa)")
            ax.set_ylabel("Frequency")
            ax.set_title("Furnace Pressure Distribution\n(with Expert Range)")
            ax.legend(fontsize=8)

    def _plot_oxygen_with_target(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if TARGET_VARIABLES[4] in self.df.columns:
            data = self.df[TARGET_VARIABLES[4]].dropna()
            ax.hist(data, bins=100, color="coral", alpha=0.7, edgecolor="white")
            ax.axvline(
                x=float(data.mean()),
                color="red",
                linestyle="--",
                label=f"Mean: {float(data.mean()):.2f}%",
            )
            ax.set_xlabel("Oxygen Content (%)")
            ax.set_ylabel("Frequency")
            ax.set_title("Oxygen Content Distribution")
            ax.legend(fontsize=8)

    def _plot_coal_with_expert_range(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if "D62AX002" in self.df.columns:
            data = self.df["D62AX002"].dropna()
            ax.hist(data, bins=100, color="forestgreen", alpha=0.7, edgecolor="white")
            ax.axvline(
                x=68, color="green", linestyle="--", linewidth=2, label="Ideal: 68 t/h"
            )
            ax.axvline(
                x=40, color="orange", linestyle=":", linewidth=2, label="Normal: 40 t/h"
            )
            ax.set_xlabel("Coal Feed Rate (t/h)")
            ax.set_ylabel("Frequency")
            ax.set_title("Coal Feed Rate Distribution\n(with Expert Range)")
            ax.legend(fontsize=8)

    def _plot_key_params_correlation(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        params = [
            "D62AX002",
            "D66P53A10",
            "D61AX023",
            "D61AX024",
            "2LA10CT11",
            "2LB30CS901",
            "2LB10CS001",
            "2NC10CS901",
        ]
        params = [p for p in params if p in self.df.columns] + [
            p
            for p in [TARGET_VARIABLES[0], TARGET_VARIABLES[4]]
            if p in self.df.columns
        ]
        if params:
            corr_matrix = self.df[params].corr()
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="RdBu_r",
                ax=ax,
                center=0,
                square=True,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_title("Key Parameters Correlation")

    def _plot_primary_air_vs_pressure(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if "D61AX023" in self.df.columns and TARGET_VARIABLES[0] in self.df.columns:
            sample_df = self.df.iloc[::50]
            scatter = ax.scatter(
                sample_df["D61AX023"],
                sample_df[TARGET_VARIABLES[0]],
                c=sample_df["D62AX002"],
                cmap="YlOrRd",
                alpha=0.6,
                s=15,
            )
            ax.set_xlabel("Primary Air Flow (m³/h)")
            ax.set_ylabel("Furnace Pressure (Pa)")
            ax.set_title("Primary Air vs Pressure\n(colored by Coal Feed)")
            plt.colorbar(scatter, ax=ax, label="Coal (t/h)", shrink=0.8)

    def _plot_coal_vs_pressure_colored(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if "D62AX002" in self.df.columns and TARGET_VARIABLES[0] in self.df.columns:
            sample_df = self.df.iloc[::50]
            scatter = ax.scatter(
                sample_df["D62AX002"],
                sample_df[TARGET_VARIABLES[0]],
                c=sample_df["D61AX023"],
                cmap="Blues",
                alpha=0.6,
                s=15,
            )
            ax.set_xlabel("Coal Feed Rate (t/h)")
            ax.set_ylabel("Furnace Pressure (Pa)")
            ax.set_title("Coal Feed vs Pressure\n(colored by Primary Air)")
            plt.colorbar(scatter, ax=ax, label="Primary Air (m³/h)", shrink=0.8)

    def _plot_pressure_timeseries_comparison(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if TARGET_VARIABLES[0] in self.df.columns:
            time_sample = self.df.iloc[::500]
            time_idx = range(len(time_sample))
            ax.plot(
                time_idx,
                time_sample[TARGET_VARIABLES[0]],
                label="Left Pressure",
                alpha=0.8,
                linewidth=1,
            )
            if TARGET_VARIABLES[1] in self.df.columns:
                ax.plot(
                    time_idx,
                    time_sample[TARGET_VARIABLES[1]],
                    label="Right Pressure",
                    alpha=0.8,
                    linewidth=1,
                )
            ax.axhline(y=-80, color="green", linestyle="--", alpha=0.5)
            ax.axhline(y=-150, color="green", linestyle="--", alpha=0.5)
            ax.fill_between(time_idx, -150, -80, alpha=0.1, color="green")
            ax.set_xlabel("Time Index (sampled)")
            ax.set_ylabel("Pressure (Pa)")
            ax.set_title("Furnace Pressure Time Series")
            ax.legend(fontsize=8)

    def _plot_control_variables_timeseries(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        if "D62AX002" in self.df.columns:
            time_sample = self.df.iloc[::500]
            time_idx = range(len(time_sample))
            ax_twin = ax.twinx()
            ax.plot(
                time_idx, time_sample["D62AX002"], "b-", label="Coal Feed", alpha=0.8
            )
            if "D61AX023" in self.df.columns:
                ax_twin.plot(
                    time_idx,
                    time_sample["D61AX023"],
                    "r-",
                    label="Primary Air",
                    alpha=0.8,
                )
            ax.set_xlabel("Time Index (sampled)")
            ax.set_ylabel("Coal Feed (t/h)", color="b")
            ax_twin.set_ylabel("Primary Air (m³/h)", color="r")
            ax.set_title("Key Control Variables")
            ax.legend(loc="upper left", fontsize=8)
            ax_twin.legend(loc="upper right", fontsize=8)

    def _plot_fan_current_comparison(self, fig: plt.Figure, pos: int) -> None:
        ax = fig.add_subplot(3, 3, pos)
        fan_currents = [
            "2BBA14Q11",
            "2BBB12Q11",
            "2BBA13Q11",
            "2BBB11Q11",
            "2BBA15Q11",
            "2BBB13Q11",
        ]
        fan_labels = [
            "PA Fan A",
            "PA Fan B",
            "SA Fan A",
            "SA Fan B",
            "ID Fan A",
            "ID Fan B",
        ]
        avg_currents = [
            float(self.df[col].mean()) if col in self.df.columns else 0.0
            for col in fan_currents
        ]
        colors = ["skyblue", "skyblue", "lightgreen", "lightgreen", "salmon", "salmon"]
        bars = ax.bar(fan_labels, avg_currents, color=colors, edgecolor="black")
        ax.set_ylabel("Average Current (A)")
        ax.set_title("Fan Current Comparison")
        ax.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars, avg_currents):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
