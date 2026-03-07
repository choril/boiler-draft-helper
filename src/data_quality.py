"""
数据质量评估 Pipeline
"""

import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from logger import get_logger


LOGGER = get_logger()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_DIR = PROJECT_ROOT / "data/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class Config:
    """配置管理类"""

    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.output_dir = self.project_root / "data" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.param_dict_path = self.output_dir / "param_dict.json"
        self.data_path = self.output_dir / "all_data_cleaned.feather"

        LOGGER.info(f"输出目录：{self.output_dir}")
        LOGGER.info(f"数据文件：{self.data_path}")
        LOGGER.info(f"参数文件：{self.param_dict_path}")

    def load_param_dict(self) -> Dict:
        """加载参数配置字典"""
        if not self.param_dict_path.exists():
            LOGGER.warning(f"参数文件不存在：{self.param_dict_path}")
            return {}

        try:
            with open(self.param_dict_path, "r", encoding="utf-8") as f:
                param_dict = json.load(f)
            LOGGER.info(f"成功加载参数字典，共 {len(param_dict)} 个参数")
            return param_dict
        except Exception as e:
            LOGGER.error(f"加载参数字典失败: {e}")
            return {}


class DataQualityAnalyzer:
    """数据质量分析类 - 负责执行各项质量检查"""

    def __init__(self, param_dict: Dict):
        self.param_dict = param_dict

    def analyze(self, df: pd.DataFrame) -> Dict:
        """执行完整的数据质量分析流程

        分析内容：
        1. 缺失值分析
        2. 异常值检测
        3. 时间连续性检查
        """
        LOGGER.info("开始数据质量分析")

        results = {
            "missing_values": self.analyze_missing_values(df),
            "outliers": self.detect_outliers(df),
            "time_continuity": self.check_time_continuity(df),
        }

        LOGGER.info("数据质量分析完成！")

        return results

    def analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """缺失值分析"""
        LOGGER.info("[1] 开始缺失值分析...")

        missing_stats = {}
        total_rows = len(df)

        for col in df.columns:
            if col not in ["TIME", "source_file"]:
                missing_count = df[col].isna().sum()
                missing_percent = (missing_count / total_rows) * 100
                missing_stats[col] = {
                    "missing_count": int(missing_count),
                    "missing_percent": round(missing_percent, 2),
                    "total_rows": total_rows,
                }

        total_missing = sum(v["missing_count"] for v in missing_stats.values())
        LOGGER.info(f"[1] 缺失值分析完成：共 {total_missing} 个缺失值")

        return missing_stats

    def detect_outliers(self, df: pd.DataFrame, iqr_multiplier: float = 1.5) -> Dict:
        """异常值检测

        检测策略：
        - 有参数配置的变量：使用量程范围（量程L ~ 量程H）检测异常
        - 无参数配置的变量：使用 IQR 方法（四分位距）检测异常
        """
        LOGGER.info("[2] 开始异常值检测...")

        outlier_stats = {}
        total_outliers = 0

        for col in df.columns:
            if col not in ["TIME", "source_file"] and pd.api.types.is_numeric_dtype(
                df[col]
            ):
                stats = self._detect_column_outliers(df, col, iqr_multiplier)
                if stats:
                    outlier_stats[col] = stats
                    total_outliers += stats["outlier_count"]

        LOGGER.info(f"[2] 异常值检测完成：共 {total_outliers} 个异常值")

        return outlier_stats

    def _detect_column_outliers(
        self, df: pd.DataFrame, col: str, iqr_multiplier: float
    ) -> Dict:
        """检测单列的异常值"""
        series = df[col].dropna()
        if len(series) == 0:
            return {}

        lower_bound, upper_bound, method = self._get_bounds(series, col, iqr_multiplier)

        outliers = series[(series < lower_bound) | (series > upper_bound)]

        return {
            "outlier_count": len(outliers),
            "outlier_percent": round((len(outliers) / len(series)) * 100, 2),
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2),
            "total_values": len(series),
            "method": method,
        }

    def _get_bounds(self, series: pd.Series, col: str, iqr_multiplier: float) -> tuple:
        """获取异常值检测的边界"""
        if col in self.param_dict:
            param_info = self.param_dict[col]
            lower = param_info.get("量程L")
            upper = param_info.get("量程H")

            if lower is not None and upper is not None:
                LOGGER.debug(f"使用参数字典中的量程: {col} -> [{lower}, {upper}]")
                return lower, upper, "param_dict"

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - iqr_multiplier * IQR
        upper = Q3 + iqr_multiplier * IQR

        LOGGER.debug(f"使用IQR方法计算量程: {col} -> [{lower:.2f}, {upper:.2f}]")
        return lower, upper, "IQR"

    def check_time_continuity(self, df: pd.DataFrame) -> Dict:
        """时间连续性检查"""
        LOGGER.info("[3] 开始时间连续性检查...")

        if "TIME" not in df or df["TIME"].dropna().empty:
            LOGGER.warning("数据中没有 TIME 列或 TIME 列为空")
            return {"total_records": 0, "continuous_gaps": []}

        try:
            ts = (
                pd.to_datetime(df["TIME"]).dropna().sort_values().reset_index(drop=True)
            )
            diffs = ts.diff()
            expected = pd.Timedelta(seconds=60)
            mask = diffs > expected

            gaps = self._extract_gaps(ts, mask)

            result = {
                "total_records": len(ts),
                "continuous_gaps": gaps,
                "total_gap_duration_seconds": sum(
                    g.get("duration_seconds", 0) for g in gaps
                ),
                "max_gap_duration": str(diffs.max()) if not diffs.empty else "0s",
            }

            LOGGER.info(f"[3] 时间连续性检查完成：发现 {len(gaps)} 个时间缺口")
            return result

        except Exception as e:
            LOGGER.error(f"[3] 时间连续性检查失败: {e}")
            return {"total_records": 0, "continuous_gaps": []}

    def _extract_gaps(self, ts: pd.Series, mask: pd.Series) -> List[Dict]:
        """提取时间缺口信息"""
        gap_indices = mask[mask].index
        gaps = []

        for idx in gap_indices:
            start_point = ts.iloc[idx - 1]
            end_point = ts.iloc[idx]
            duration = end_point - start_point

            gaps.append(
                {
                    "start_time": start_point.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_point.strftime("%Y-%m-%d %H:%M:%S"),
                    "duration": str(duration),
                    "duration_seconds": duration.total_seconds(),
                }
            )

        return gaps


class QualityReportGenerator:
    """质量报告生成类 - 负责生成 JSON 和 Markdown 格式的报告"""

    @staticmethod
    def generate_json_report(quality_results: Dict, output_path: Path) -> Dict:
        """生成 JSON 格式的质量报告"""
        LOGGER.info(f"[4] 正在生成 JSON 报告：{output_path}")

        report = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "quality_metrics": quality_results,
            "summary": QualityReportGenerator._calculate_summary(quality_results),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        LOGGER.info(f"[4] JSON 报告已保存：{output_path}")
        return report

    @staticmethod
    def _calculate_summary(quality_results: Dict) -> Dict:
        """计算质量摘要统计"""
        missing_values = quality_results.get("missing_values", {})
        outliers = quality_results.get("outliers", {})
        time_continuity = quality_results.get("time_continuity", {})

        total_missing = sum(v["missing_count"] for v in missing_values.values())
        total_values = (
            sum(v["total_rows"] for v in missing_values.values())
            if missing_values
            else 0
        )
        overall_missing_rate = (
            (total_missing / total_values * 100) if total_values > 0 else 0
        )

        total_outliers = sum(v["outlier_count"] for v in outliers.values())
        total_valid_values = (
            sum(v["total_values"] for v in outliers.values()) if outliers else 0
        )
        overall_outlier_rate = (
            (total_outliers / total_valid_values * 100) if total_valid_values > 0 else 0
        )

        return {
            "overall_missing_rate": round(overall_missing_rate, 2),
            "overall_outlier_rate": round(overall_outlier_rate, 2),
            "time_gap_count": len(time_continuity.get("continuous_gaps", [])),
            "total_records": time_continuity.get("total_records", 0),
        }

    @staticmethod
    def generate_markdown_report(report: Dict, output_path: Path):
        """生成 Markdown 格式的质量报告"""
        LOGGER.info(f"[5] 正在生成 Markdown 报告：{output_path}")

        md_content = QualityReportGenerator._build_markdown_content(report)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        LOGGER.info(f"[5] Markdown 报告已保存：{output_path}")

    @staticmethod
    def _build_markdown_content(report: Dict) -> str:
        """构建 Markdown 报告内容"""
        summary = report["summary"]
        metrics = report["quality_metrics"]

        content = f"""# 数据质量评估报告

**生成时间**: {report['timestamp']}

## 摘要

| 指标 | 值 |
|------|----|
| 整体缺失率 | {summary['overall_missing_rate']}% |
| 整体异常率 | {summary['overall_outlier_rate']}% |
| 时间间隔缺口数 | {summary['time_gap_count']} |
| 总记录数 | {summary['total_records']} |

## 详细分析

### 1. 缺失值分析

| 变量名 | 缺失数 | 缺失率 | 总记录数 |
|--------|--------|--------|----------|
"""

        content += QualityReportGenerator._build_missing_values_table(
            metrics.get("missing_values", {})
        )
        content += "\n### 2. 异常值分析\n\n"
        content += "| 变量名 | 异常数 | 异常率 | 下界 | 上界 | 总有效值 | 检测方法 |\n"
        content += "|--------|--------|--------|------|------|----------|----------|\n"
        content += QualityReportGenerator._build_outliers_table(
            metrics.get("outliers", {})
        )
        content += "\n### 3. 时间连续性分析\n\n"
        content += QualityReportGenerator._build_time_continuity_section(
            metrics.get("time_continuity", {})
        )

        return content

    @staticmethod
    def _build_missing_values_table(missing_values: Dict) -> str:
        """构建缺失值表格"""
        table = ""
        for col, stats in missing_values.items():
            table += f"| {col} | {stats['missing_count']} | {stats['missing_percent']}% | {stats['total_rows']} |\n"
        return table

    @staticmethod
    def _build_outliers_table(outliers: Dict) -> str:
        """构建异常值表格"""
        table = ""
        for col, stats in outliers.items():
            method = stats.get("method", "IQR")
            table += f"| {col} | {stats['outlier_count']} | {stats['outlier_percent']}% | {stats['lower_bound']} | {stats['upper_bound']} | {stats['total_values']} | {method} |\n"
        return table

    @staticmethod
    def _build_time_continuity_section(time_continuity: Dict) -> str:
        """构建时间连续性部分"""
        section = f"- 总记录数: {time_continuity.get('total_records', 0)}\n"
        section += f"- 最大缺口时长: {time_continuity.get('max_gap_duration', '0s')}\n"

        gaps = time_continuity.get("continuous_gaps", [])
        if gaps:
            section += "\n**时间缺口详情**:\n\n"
            section += "| 开始时间 | 结束时间 | 时长 |\n"
            section += "|----------|----------|------|\n"
            for gap in gaps:
                section += (
                    f"| {gap['start_time']} | {gap['end_time']} | {gap['duration']} |\n"
                )

        return section


class DataQualityPipeline:
    """数据质量评估主流程类"""

    def __init__(self):
        self.config = Config()
        self.analyzer = None
        self.report_generator = QualityReportGenerator()

    def run(self):
        """执行完整的数据质量评估流程

        流程：
        1. 加载参数配置
        2. 读取清洗后的数据
        3. 执行数据质量分析
        4. 生成质量报告（JSON 和 Markdown）
        """
        LOGGER.info("开始数据质量评估 Pipeline")

        if not self.config.data_path.exists():
            LOGGER.error(f"数据文件不存在：{self.config.data_path}")
            return

        param_dict = self.config.load_param_dict()

        LOGGER.info(f"正在加载数据文件：{self.config.data_path}")
        df = pd.read_feather(self.config.data_path)
        LOGGER.info(f"数据加载完成，共 {len(df):,} 行，{len(df.columns)} 列")

        self.analyzer = DataQualityAnalyzer(param_dict)
        quality_results = self.analyzer.analyze(df)

        json_report_path = self.config.output_dir / "data_quality_report.json"
        report = self.report_generator.generate_json_report(
            quality_results, json_report_path
        )

        md_report_path = self.config.output_dir / "data_quality_report.md"
        self.report_generator.generate_markdown_report(report, md_report_path)

        LOGGER.info("数据质量评估 Pipeline 完成！")
        LOGGER.info(f"JSON 报告：{json_report_path}")
        LOGGER.info(f"Markdown 报告：{md_report_path}")


def main():
    pipeline = DataQualityPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
