"""
数据预处理 Pipeline
"""

import os
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List
from src.utils.logger import get_logger
from src.utils.config import UNIMPORTANT_PARAMS
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from src.event_period_cleaner import EventPeriodCleaner, load_intervention_records


LOGGER = get_logger()


class Config:
    """配置管理类"""

    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.data_dir = (
            self.project_root / "data" / "川宁项目4锅炉2号机组-60秒-20260308"
        )
        self.output_dir = self.project_root / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.param_file = self._find_param_file()
        self.unimportant_params = UNIMPORTANT_PARAMS

        LOGGER.info(f"数据目录：{self.data_dir}")
        LOGGER.info(f"参数文件：{self.param_file}")
        LOGGER.info(f"输出目录：{self.output_dir}")
        LOGGER.info(f"非重要参数共 {len(self.unimportant_params)} 个")

    def _find_param_file(self) -> Path:
        """查找参数文件"""
        files = list((self.project_root / "data").glob("*参数配置*.xlsx"))
        if files:
            return files[0]
        raise FileNotFoundError("未找到参数文件")

    def load_param_dict(self) -> tuple[Dict, List[str]]:
        """加载参数配置字典 - 从参数配置文件读取"""
        try:
            config = pd.read_excel(
                self.param_file, sheet_name="config", header=1, engine="openpyxl"
            )
            config = config.dropna(subset=["点名"])
        
            field_count = len(config)
            unique_count = config["点名"].nunique()
            LOGGER.info(
                f"参数配置：字段数 = {field_count}, 唯一点名数 = {unique_count}"
            )

            duplicate_points = []
            if field_count != unique_count:
                duplicates = config[config.duplicated(subset=["点名"], keep=False)]
                duplicate_points = duplicates["点名"].unique().tolist()
                LOGGER.warning(f"发现 {len(duplicate_points)} 个重复的点名：")
                for point_name in duplicate_points:
                    LOGGER.warning(f"  * {point_name}")

            param_dict = {}
            for _, row in config.iterrows():
                point_name = str(row["点名"])
                param_dict[point_name] = {
                    k: (None if pd.isna(v) else v)
                    for k, v in row.items()
                    if k != "点名" and "Unnamed" not in k
                }
            # 添加时间戳和源文件参数
            new_param = {
                "TIME":{
                    "简称":"时间戳",
                    "描述":"时间戳",
                    "单位": None,
                    "量程H": None,
                    "量程L": None,
                },
                "source_file":{
                    "简称":"源文件",
                    "描述":"数据来源文件",
                    "单位": None,
                    "量程H": None,
                    "量程L": None,
                }
            }
            param_dict.update(new_param)
            return param_dict, duplicate_points
        except Exception as e:
            LOGGER.error(f"读取参数配置文件失败：{e}")
            return {}, []

    def save_param_dict(self, param_dict: Dict):
        """保存参数配置字典"""
        with open(self.output_dir / "param_dict.json", "w", encoding="utf-8") as f:
            json.dump(param_dict, f, ensure_ascii=False, indent=2)


class DataLoader:
    """数据加载类 - 负责从 Excel 文件中读取和合并数据"""

    def __init__(self, data_dir: Path, duplicate_points: List[str] = None):
        self.data_dir = data_dir
        self.duplicate_points = duplicate_points or []

    def read_single_day(self, file: Path, sheet: str) -> pd.DataFrame:
        """读取单个工作表"""
        try:
            df = pd.read_excel(file, sheet_name=sheet, header=0, engine="openpyxl")
            if df.empty:
                return pd.DataFrame()

            df.columns = df.columns.astype(str)
            df = df.rename(columns={df.columns[0]: "TIME"})
            df.drop(0, inplace=True, errors="ignore")

            df = self._remove_duplicate_columns(df)

            return df
        except Exception as e:
            LOGGER.error(f"读取文件 {file.name} 页签 {sheet} 失败：{e}")
            return pd.DataFrame()

    def _remove_duplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除重复的列（pandas自动重命名为 .1, .2 等）"""
        if not self.duplicate_points:
            return df

        cols_to_drop = []
        for col in df.columns:
            if col in ["TIME", "source_file"]:
                continue

            for dup_point in self.duplicate_points:
                if col.startswith(f"{dup_point}."):
                    cols_to_drop.append(col)
                    break

        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

        return df

    def process_file(self, file: Path) -> pd.DataFrame:
        """处理单个文件"""
        try:
            xl = pd.ExcelFile(file, engine="openpyxl")
            valid_sheets = [s for s in xl.sheet_names if s != "config"]

            if not valid_sheets:
                LOGGER.warning(f"文件 {file.name} 中没有找到有效的数字页签")
                return pd.DataFrame()

            valid_sheets.sort(key=int)

            all_dfs = []
            for sheet in valid_sheets:
                df = self.read_single_day(file, sheet)
                if not df.empty:
                    df["source_file"] = file.name
                    all_dfs.append(self._clean_dataframe(df))

            if not all_dfs:
                LOGGER.warning(f"文件 {file.name} 中没有找到有效的数据页签")
                return pd.DataFrame()

            return pd.concat(all_dfs, ignore_index=True)
        except Exception as e:
            LOGGER.error(f"处理文件 {file.name} 时出错：{e}")
            return pd.DataFrame()

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据框，转换数值类型"""
        THRESHOLD = 1e-10
        exclude_cols = ["TIME", "source_file"]
        for col in df.columns:
            if col not in exclude_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                mask = df[col].abs() < THRESHOLD
                df.loc[mask, col] = 0
        return df.infer_objects(copy=False)

    def load_all_months(self) -> pd.DataFrame:
        """并行加载所有月份数据"""
        files = sorted(list(self.data_dir.glob("锅炉2号机组-60秒*.xlsx")))

        if not files:
            LOGGER.warning(f"在 {self.data_dir} 下未找到文件")
            return pd.DataFrame()

        all_dfs = []
        max_workers = min(os.cpu_count() or 4, len(files))

        LOGGER.info(f"开始并行读取 {len(files)} 个文件 (使用 {max_workers} 个进程)...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_file, f): f for f in files}

            for future in tqdm(
                as_completed(futures), total=len(files), desc="读取进度", unit="file"
            ):
                try:
                    result_df = future.result()
                    if not result_df.empty:
                        all_dfs.append(result_df)
                except Exception as e:
                    file_name = futures[future].name
                    LOGGER.error(f"文件 {file_name} 处理异常：{e}")

        if not all_dfs:
            return pd.DataFrame()

        return self._merge_and_clean(all_dfs)

    def _merge_and_clean(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """合并数据并执行基础清洗"""
        LOGGER.info("正在合并数据并按时间排序...")
        final_df = pd.concat(dfs, ignore_index=True)
        final_df["TIME"] = pd.to_datetime(final_df["TIME"])
        final_df = final_df.sort_values(by="TIME").reset_index(drop=True)
        final_df["source_file"] = final_df.pop("source_file")
        final_df = self._remove_empty_rows(final_df)

        return final_df

    def _remove_empty_rows(self, df: pd.DataFrame):
        """删除空值行和零值行"""

        # 删除零值和空值加起来超过阈值的行
        df = self._remove_zero_and_nan_rows(df)

        rows_before = len(df)
        # 关键列存在空值则删除整行
        critical_cols = ["TIME", "D62AX002", "MSFLOW"]
        df.dropna(subset=critical_cols, inplace=True)
        rows_after = len(df)
        LOGGER.info(
            f"已删除关键列存在空值的行：{rows_before} -> {rows_after} (删除了 {rows_before - rows_after} 行)"
        )

        return df

    def _remove_zero_and_nan_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除数值列大多为零或NaN的行"""
        numeric_cols = df.select_dtypes(include=["number"]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in ["TIME", "source_file"]
        ]

        if not numeric_cols:
            return df

        threshold = len(numeric_cols) // 4

        is_zero_matrix = df[numeric_cols] == 0
        is_nan_matrix = df[numeric_cols].isna()
        zero_or_nan_counts_per_row = is_zero_matrix.sum(axis=1) + is_nan_matrix.sum(
            axis=1
        )

        rows_to_drop_mask = zero_or_nan_counts_per_row >= threshold
        drop_count = rows_to_drop_mask.sum()

        if drop_count > 0:
            df = df[~rows_to_drop_mask].reset_index(drop=True)
            LOGGER.info(f"已删除零值和空值列数 >= {threshold} 的行：{drop_count} 行")

        return df


class DataCleaner:
    """数据清洗类 - 负责缺失值填充、异常值处理和数据标准化"""

    def __init__(self, param_dict: Dict):
        self.param_dict = param_dict

    def clean(
        self, df: pd.DataFrame, method: str, window: int, threshold: float
    ) -> pd.DataFrame:
        """执行完整清洗流程"""
        self.fill_missing_values(df)
        self.handle_outliers(df, method, window, threshold)
        self.standardize(df)

        return df

    def fill_missing_values(self, df: pd.DataFrame):
        """填充缺失值"""
        LOGGER.info("开始填充缺失值...")

        numeric_cols = df.select_dtypes(include=[float, int]).columns
        missing_before = df[numeric_cols].isna().sum().sum()

        if missing_before == 0:
            LOGGER.warning("数据中没有缺失值，跳过缺失填充")
            return

        for col in numeric_cols:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                fill_value = df[col].median()
                df[col] = df[col].fillna(fill_value)
                LOGGER.debug(
                    f"列 {col} 填充 {missing_count} 个缺失值，填充值：{fill_value:.4f}"
                )

        missing_after = df[numeric_cols].isna().sum().sum()
        LOGGER.info(
            f"缺失值填充完成：{missing_before} -> {missing_after} (填充了 {missing_before - missing_after} 个)"
        )

    def handle_outliers(
        self, df: pd.DataFrame, method: str, window: int, threshold: float
    ):
        """处理异常值"""
        if method == "rolling_mad":
            LOGGER.info(
                f"开始处理异常值 (方法：{method}, 窗口：{window}, 阈值：{threshold})..."
            )
        elif method == "iqr":
            LOGGER.info(f"开始处理异常值 (方法：{method})...")
        else:
            LOGGER.warning(f"未知异常值处理方法 {method}！")
            return

        outlier_count = 0
        for col in df.columns:
            if col in ["TIME", "source_file"]:
                continue

            col_dtype = df[col].dtype
            if col_dtype == bool or col_dtype == object:
                continue

            count = self._process_column_outliers(df, col, method, window, threshold)
            outlier_count += count

        LOGGER.info(f"异常值处理完成：共处理 {outlier_count} 个异常值")

    def _process_column_outliers(
        self,
        df: pd.DataFrame,
        col: str,
        method: str,
        window: int,
        threshold: float,
    ) -> int:
        """处理单列的异常值 - 使用滑动窗口+MAD方法"""
        if method == "rolling_mad":
            return self._handle_rolling_mad_outliers(df, col, window, threshold)
        else:
            return self._handle_iqr_outliers(df, col)

    def _handle_rolling_mad_outliers(
        self,
        df: pd.DataFrame,
        col: str,
        window: int,
        threshold: float,
    ) -> int:
        """使用滑动窗口 + MAD方法处理异常值

        异常值定义为: |x - median| > threshold * MAD
        """
        series = df[col].copy()

        rolling_median = series.rolling(
            window=window, center=True, min_periods=1
        ).median()

        rolling_mad = (
            (series - rolling_median)
            .abs()
            .rolling(window=window, center=True, min_periods=1)
            .median()
        )

        mad_threshold = threshold * rolling_mad
        lower_series = rolling_median - mad_threshold
        upper_series = rolling_median + mad_threshold

        lower = lower_series.min()
        upper = upper_series.max()

        return self._clip_column(df, col, lower, upper)

    def _handle_iqr_outliers(self, df: pd.DataFrame, col: str) -> int:
        """使用 IQR 方法处理异常值 - 基于统计分布"""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return self._clip_column(df, col, lower, upper)

    def _clip_column(
        self, df: pd.DataFrame, col: str, lower: float, upper: float
    ) -> int:
        """截断列的异常值到指定范围"""
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        if count > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            LOGGER.debug(f"列 {col} 截断 {count} 个异常值到 [{lower:.2f}, {upper:.2f}]")
        return count

    def standardize(self, df: pd.DataFrame):
        """数据标准化与格式化"""
        LOGGER.info("开始数据标准化与格式化...")

        df["TIME"] = pd.to_datetime(df["TIME"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        numeric_cols = df.select_dtypes(include=[float]).columns
        for col in numeric_cols:
            df[col] = df[col].round(4)

        df.sort_values(by="TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)

        LOGGER.info(f"数据标准化完成：{len(df)} 行，{len(df.columns)} 列")


class DataSaver:
    """数据保存类 - 负责将数据保存为不同格式"""

    @staticmethod
    def save_to_feather(df: pd.DataFrame, output_path: Path) -> bool:
        """保存为 feather 格式"""
        try:
            LOGGER.info(f"正在保存为 feather 文件：{output_path}")

            file_path = output_path.with_suffix(".feather")
            df.reset_index(drop=True).to_feather(file_path)

            size_mb = file_path.stat().st_size / (1024 * 1024)
            LOGGER.info(
                f"保存成功：{file_path.name} | 大小：{size_mb:.2f} MB | 行数：{len(df):,}"
            )
            return True
        except Exception as e:
            LOGGER.error(f"保存为 feather 文件失败：{e}")
            return False

    @staticmethod
    def save_to_csv(feather_path: Path, output_path: Path) -> bool:
        """从 feather 转换为 csv 格式"""
        try:
            df = pd.read_feather(feather_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"正在保存为 csv 文件：{output_path}")
            df.to_csv(output_path, index=False, encoding="utf-8-sig")

            size_mb = output_path.stat().st_size / (1024 * 1024)
            LOGGER.info(
                f"保存成功：{output_path.name} | 大小：{size_mb:.2f} MB | 行数：{len(df):,}"
            )
            return True
        except Exception as e:
            LOGGER.error(f"保存为 csv 文件失败：{e}")
            return False


class DataPipeline:
    """数据预处理主流程类 - 协调各个组件完成完整的数据处理流程"""

    def __init__(self):
        self.config = Config()
        self.loader = None
        self.cleaner = None
        self.saver = DataSaver()

    def run(
        self,
        save_csv: bool = True,
        outlier_method: str = "rolling_mad",
        event_clean_method: str = "mark",
        window: int = 720,
        threshold: float = 3.0,
    ):
        """执行完整的数据预处理流程

        流程：
        1. 加载参数配置
        2. 并行读取所有数据文件
        3. 合并数据并执行基础清洗
        4. 执行特殊工况期数据清洗
        5. 执行数据清洗（缺失值填充、异常值处理、标准化）
        6. 保存清洗后的数据

        Args:
            - save_csv: 是否保存为CSV格式
            - outlier_method: 异常值检测方法
                - "rolling_mad": 滚动MAD方法(默认)
                - "iqr": IQR方法
            - event_clean_method: 特殊工况期清洗方法
                - "mark": 标记特殊工况期数据(默认)
                - "remove": 删除特殊工况期数据
                - "none": 不处理特殊工况期数据
        """
        LOGGER.info("开始数据预处理 Pipeline")

        param_dict, duplicate_points = self.config.load_param_dict()
        self.config.save_param_dict(param_dict)
        LOGGER.info("参数配置加载完成")

        self.loader = DataLoader(self.config.data_dir, duplicate_points)
        raw_data = self.loader.load_all_months()
        if raw_data.empty:
            LOGGER.error("未加载到任何有效数据")
            return
        LOGGER.info(f"原始数据加载完成：{len(raw_data)} 行，{len(raw_data.columns)} 列")
        # 移除非重要列
        raw_data = raw_data.drop(columns=self.config.unimportant_params, errors="ignore")
        LOGGER.info(f"已移除非重要参数列，剩余 {len(raw_data.columns)} 列")

        if event_clean_method != "none":
            LOGGER.info(f"开始特殊工况期数据清洗 (方法: {event_clean_method})...")
            event_periods = load_intervention_records(self.config.data_dir)

            if event_periods:
                event_cleaner = EventPeriodCleaner(event_periods)
                summary_df = event_cleaner.get_event_period_summary()
                summary_df.to_csv(
                    self.config.output_dir / "event_period_summary.csv", index=False
                )
                LOGGER.info(
                    f"特殊工况期摘要已保存至 {self.config.output_dir / 'event_period_summary.csv'}"
                )

                raw_data = event_cleaner.clean_event_periods(
                    raw_data, event_clean_method
                )
            else:
                LOGGER.warning("未找到人工干预记录，跳过特殊工况期清洗")

        self.cleaner = DataCleaner(param_dict)
        cleaned_data = self.cleaner.clean(raw_data, outlier_method, window, threshold)

        self.saver.save_to_feather(
            cleaned_data, self.config.output_dir / "all_data_cleaned"
        )

        if save_csv:
            self.saver.save_to_csv(
                self.config.output_dir / "all_data_cleaned.feather",
                self.config.output_dir / "all_data_cleaned.csv",
            )

        LOGGER.info("数据预处理 Pipeline 完成！")


def main():
    parser = argparse.ArgumentParser(description="数据预处理 Pipeline")
    parser.add_argument(
        "--to_csv", action="store_true", help="将最终数据保存为 csv 文件"
    )
    parser.add_argument(
        "--outlier_method",
        type=str,
        default="rolling_mad",
        choices=["rolling_mad", "iqr"],
        help="异常值检测方法（rolling_mad/iqr）",
    )
    parser.add_argument(
        "--event_method",
        type=str,
        default="mark",
        choices=["mark", "remove", "none"],
        help="特殊工况期数据清洗方法（mark标记/remove删除/none不处理）",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=720,
        help="异常值检测窗口大小（默认720）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=3.0,
        help="异常值检测阈值（默认3.0）",
    )

    args = parser.parse_args()

    pipeline = DataPipeline()
    pipeline.run(
        save_csv=args.to_csv,
        outlier_method=args.outlier_method,
        event_clean_method=args.event_method,
        window=args.window,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
