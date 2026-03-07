"""
数据预处理 Pipeline
"""

import os
import json
import argparse
from numpy import clip
import pandas as pd
from pathlib import Path
from typing import Dict, List
from logger import get_logger
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


LOGGER = get_logger()


class Config:
    """配置管理类"""

    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.data_dir = self.project_root / "data" / "锅炉2号机组-60秒-online"
        self.output_dir = self.project_root / "data" / "output"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.param_file = self._find_param_file()

        LOGGER.info(f"数据目录：{self.data_dir}")
        LOGGER.info(f"参数文件：{self.param_file}")
        LOGGER.info(f"输出目录：{self.output_dir}")

    def _find_param_file(self) -> Path:
        """查找参数文件"""
        files = list((self.project_root / "data").glob("*参数说明*.xlsx"))
        if files:
            return files[0]
        raise FileNotFoundError("未找到参数文件")

    def load_param_dict(self) -> Dict:
        """加载参数配置字典"""
        config = pd.read_excel(self.param_file, sheet_name="config", header=1)
        config = config.dropna(subset=["点名"])
        return {
            str(row["点名"]): {
                k: (None if pd.isna(v) else v) for k, v in row.items() if k != "点名"
            }
            for _, row in config.iterrows()
        }

    def save_param_dict(self, param_dict: Dict):
        """保存参数配置字典"""
        with open(self.output_dir / "param_dict.json", "w", encoding="utf-8") as f:
            json.dump(param_dict, f, ensure_ascii=False, indent=2)


class DataLoader:
    """数据加载类 - 负责从 Excel 文件中读取和合并数据"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir

    def read_single_day(self, file: Path, sheet: str) -> pd.DataFrame:
        """读取单个工作表"""
        try:
            df = pd.read_excel(file, sheet_name=sheet, header=0, engine="openpyxl")
            if df.empty:
                return pd.DataFrame()

            df.columns = df.columns.astype(str)
            df = df.rename(columns={df.columns[0]: "TIME"})
            df.drop(0, inplace=True, errors="ignore")

            if self._is_invalid_2026_jan(df, file):
                return pd.DataFrame()

            return df
        except Exception as e:
            LOGGER.error(f"读取文件 {file.name} 页签 {sheet} 失败：{e}")
            return pd.DataFrame()

    def _is_invalid_2026_jan(self, df: pd.DataFrame, file: Path) -> bool:
        """检查是否为无效的 2026 年 1 月数据（缓存数据）"""
        if df.empty or "TIME" not in df or "2026年01月" in file.name:
            return False

        try:
            df["TIME"] = pd.to_datetime(df["TIME"])
            is_2026_jan = (df["TIME"].dt.year == 2026) & (df["TIME"].dt.month == 1)
            return is_2026_jan.all()
        except Exception as e:
            LOGGER.warning(f"解析时间戳失败：{e}")
            return False

    def process_file(self, file: Path) -> pd.DataFrame:
        """处理单个文件"""
        try:
            xl = pd.ExcelFile(file, engine="openpyxl")
            valid_sheets = [s for s in xl.sheet_names if s != "config"]

            if not valid_sheets:
                LOGGER.warning(f"文件 {file.name} 中没有找到有效的数字页签")
                return pd.DataFrame()

            if "2026年01月" in file.name:
                return self._process_2026_jan_file(file, valid_sheets)
            else:
                return self._process_normal_file(file, valid_sheets)
        except Exception as e:
            LOGGER.error(f"处理文件 {file.name} 时出错：{e}")
            return pd.DataFrame()

    def _process_2026_jan_file(self, file: Path, sheets: List[str]) -> pd.DataFrame:
        """处理 2026 年 1 月文件（保留所有页签）"""
        day_dfs = []
        for sheet in sheets:
            df = self.read_single_day(file, sheet)
            if not df.empty:
                day_dfs.append(df)

        if not day_dfs:
            return pd.DataFrame()

        combined = pd.concat(day_dfs, ignore_index=True)
        combined["source_file"] = file.name
        return self._clean_dataframe(combined)

    def _process_normal_file(self, file: Path, sheets: List[str]) -> pd.DataFrame:
        """处理普通文件（只保留最后一个有效页签）"""
        sheets.sort(key=int, reverse=True)

        for sheet in sheets:
            df = self.read_single_day(file, sheet)
            if not df.empty:
                df["source_file"] = file.name
                return self._clean_dataframe(df)

        LOGGER.warning(f"文件 {file.name} 中没有找到有效的数据页签")
        return pd.DataFrame()

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """清洗数据框，转换数值类型"""
        exclude_cols = {"TIME", "source_file"}
        for col in df.columns:
            if col not in exclude_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
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

        self._remove_empty_rows(final_df)
        self._remove_invalid_dates(final_df)

        return final_df

    def _remove_empty_rows(self, df: pd.DataFrame):
        """删除空值行"""
        rows_before = len(df)
        df.dropna(how="all", inplace=True)
        df.dropna(
            subset=[col for col in df.columns if col != "source_file"], inplace=True
        )
        rows_after = len(df)
        LOGGER.info(
            f"已清空空值行：{rows_before} -> {rows_after} (删除了 {rows_before - rows_after} 行)"
        )

    def _remove_invalid_dates(self, df: pd.DataFrame):
        """删除已知异常日期"""
        rows_before = len(df)
        mask = ~(
            (df["TIME"].dt.year == 2026)
            & (df["TIME"].dt.month == 1)
            & (df["TIME"].dt.day == 27)
        )
        df.drop(df[~mask].index, inplace=True)
        rows_after = len(df)
        LOGGER.info(
            f"已清除 2026 年 1 月 27 日的数据：{rows_before} -> {rows_after} (删除了 {rows_before - rows_after} 行)"
        )


class DataCleaner:
    """数据清洗类 - 负责缺失值填充、异常值处理和数据标准化"""

    def __init__(self, param_dict: Dict):
        self.param_dict = param_dict

    def clean(self, df: pd.DataFrame, method: str) -> pd.DataFrame:
        """执行完整清洗流程"""
        self.fill_missing_values(df)
        self.handle_outliers(df, method)
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

    def handle_outliers(self, df: pd.DataFrame, method: str):
        """处理异常值

        策略：
        - 有参数配置的变量：使用量程范围（量程L ~ 量程H）检测异常
        - 无参数配置的变量：使用 IQR 方法（四分位距）检测异常
        - 处理方法：clip（截断）或 remove（删除）
        """
        LOGGER.info(f"开始处理异常值 (方法：{method})...")

        outlier_count = 0
        for col in df.columns:
            if col in ["TIME", "source_file"]:
                continue

            count = self._process_column_outliers(df, col, method)
            outlier_count += count

        LOGGER.info(f"异常值处理完成：共处理 {outlier_count} 个异常值")

    def _process_column_outliers(self, df: pd.DataFrame, col: str, method: str) -> int:
        """处理单列的异常值"""
        if col in self.param_dict:
            return self._handle_param_outliers(df, col, method)
        else:
            return self._handle_iqr_outliers(df, col, method)

    def _handle_param_outliers(self, df: pd.DataFrame, col: str, method: str) -> int:
        """使用参数配置处理异常值 - 基于传感器量程范围"""
        lower = self.param_dict[col].get("量程L")
        upper = self.param_dict[col].get("量程H")

        if pd.isna(lower) or pd.isna(upper):
            return 0

        if method == "clip":
            return self._clip_column(df, col, lower, upper)
        elif method == "remove":
            return self._remove_column_outliers(df, col, lower, upper)
        return 0

    def _handle_iqr_outliers(self, df: pd.DataFrame, col: str, method: str) -> int:
        """使用 IQR 方法处理异常值 - 基于统计分布"""
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if method == "clip":
            return self._clip_column(df, col, lower, upper)
        return 0

    def _clip_column(
        self, df: pd.DataFrame, col: str, lower: float, upper: float
    ) -> int:
        """截断列的异常值到指定范围"""
        before_count = ((df[col] < lower) | (df[col] > upper)).sum()
        if before_count > 0:
            df[col] = df[col].clip(lower=lower, upper=upper)
            LOGGER.debug(
                f"列 {col} 截断 {before_count} 个异常值到 [{lower:.2f}, {upper:.2f}]"
            )
        return before_count

    def _remove_column_outliers(
        self, df: pd.DataFrame, col: str, lower: float, upper: float
    ) -> int:
        """删除列的异常值"""
        mask = (df[col] >= lower) & (df[col] <= upper)
        removed = (~mask).sum()
        if removed > 0:
            df.drop(df[~mask].index, inplace=True)
            LOGGER.debug(f"列 {col} 删除 {removed} 个异常值")
        return removed

    def standardize(self, df: pd.DataFrame):
        """数据标准化与格式化"""
        LOGGER.info("开始数据标准化与格式化...")

        df["TIME"] = pd.to_datetime(df["TIME"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        numeric_cols = df.select_dtypes(include=[float]).columns
        for col in numeric_cols:
            df[col] = df[col].round(4)

        df.sort_values(by="TIME", inplace=True)
        df.reset_index(drop=True, inplace=True)

        LOGGER.info(f"数据标准化完成：{len(df.columns)} 列，{len(df)} 行")


class DataSaver:
    """数据保存类 - 负责将数据保存为不同格式"""

    @staticmethod
    def save_to_feather(df: pd.DataFrame, output_path: Path) -> bool:
        """保存为 feather 格式 - 高效的列式存储格式"""
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
    def save_to_excel(feather_path: Path, output_path: Path) -> bool:
        """从 feather 转换为 excel 格式 - 便于人工查看和分析"""
        try:
            df = pd.read_feather(feather_path)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            LOGGER.info(f"正在保存为 excel 文件：{output_path}")
            df.to_excel(output_path, index=False, engine="openpyxl")

            size_mb = output_path.stat().st_size / (1024 * 1024)
            LOGGER.info(
                f"保存成功：{output_path.name} | 大小：{size_mb:.2f} MB | 行数：{len(df):,}"
            )
            return True
        except Exception as e:
            LOGGER.error(f"保存为 excel 文件失败：{e}")
            return False


class DataPipeline:
    """数据预处理主流程类 - 协调各个组件完成完整的数据处理流程"""

    def __init__(self):
        self.config = Config()
        self.loader = DataLoader(self.config.data_dir)
        self.cleaner = None
        self.saver = DataSaver()

    def run(self, save_excel: bool = True, method: str = "clip"):
        """执行完整的数据预处理流程

        流程：
        1. 加载参数配置
        2. 并行读取所有数据文件
        3. 合并数据并执行基础清洗
        4. 执行数据清洗（缺失值填充、异常值处理、标准化）
        5. 保存清洗后的数据
        """
        LOGGER.info("开始数据预处理 Pipeline")

        param_dict = self.config.load_param_dict()
        self.config.save_param_dict(param_dict)
        LOGGER.info("参数配置加载完成")

        raw_data = self.loader.load_all_months()
        if raw_data.empty:
            LOGGER.error("未加载到任何有效数据")
            return

        self.cleaner = DataCleaner(param_dict)
        cleaned_data = self.cleaner.clean(raw_data, method)

        self.saver.save_to_feather(
            cleaned_data, self.config.output_dir / "all_data_cleaned"
        )

        if save_excel:
            self.saver.save_to_excel(
                self.config.output_dir / "all_data_cleaned.feather",
                self.config.output_dir / "all_data_cleaned.xlsx",
            )

        LOGGER.info("数据预处理 Pipeline 完成！")
        LOGGER.info(
            f"清洗后数据集：{self.config.output_dir / 'all_data_cleaned.feather'}"
        )
        LOGGER.info(
            f"Excel 格式：{self.config.output_dir / 'all_data_cleaned.xlsx'}"
        )


def main():
    parser = argparse.ArgumentParser(description="数据预处理 Pipeline")
    parser.add_argument("--to_excel", action="store_true", help="将最终数据保存为 excel 文件")
    parser.add_argument("--method", type=str, default="clip", help="异常值处理方法（clip/remove）",)
    args = parser.parse_args()

    pipeline = DataPipeline()
    pipeline.run(save_excel=args.to_excel, method=args.method)


if __name__ == "__main__":
    main()
