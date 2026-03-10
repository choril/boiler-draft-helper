"""
事件期数据清洗模块
基于人工干预记录表,识别和标记事件期内的数据噪声
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from typing import List
from logger import get_logger


LOGGER = get_logger()


class EventPeriod:
    """事件期数据结构"""

    def __init__(
        self,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        event_type: str,
        device: str,
        reason: str,
    ):
        self.start_time = start_time
        self.end_time = end_time
        self.event_type = event_type
        self.device = device
        self.reason = reason
        self.duration = end_time - start_time

    def __repr__(self):
        return (
            f"EventPeriod({self.start_time} ~ {self.end_time}, "
            f"type={self.event_type}, device={self.device})"
        )


class InterventionRecordParser:
    """人工干预记录解析器"""

    EVENT_PAIRS = {
        "停机": "开机",
        "硬件故障": "检修完毕",
        "MFT预警": None,
    }

    def __init__(self, record_file: Path):
        self.record_file = record_file

    def parse(self) -> List[EventPeriod]:
        """解析人工干预记录表,提取事件期"""
        LOGGER.info(f"正在读取人工干预记录表: {self.record_file}")

        df = pd.read_excel(self.record_file, engine="openpyxl")
        df.columns = ["时间", "设备", "特殊工况", "其他干预原因"]
        df = df.dropna(subset=["时间"])

        df["时间"] = pd.to_datetime(df["时间"], format="mixed")
        df["其他干预原因"] = df["其他干预原因"].ffill()
        df = df.sort_values("时间").reset_index(drop=True)

        event_periods = self._extract_event_periods(df)

        LOGGER.info(f"成功解析 {len(event_periods)} 个事件期")
        return event_periods

    def _extract_event_periods(self, df: pd.DataFrame) -> List[EventPeriod]:
        """从记录表中提取事件期"""
        event_periods = []
        i = 0

        while i < len(df):
            row = df.iloc[i]
            event_type = row["特殊工况"]

            if event_type in self.EVENT_PAIRS:
                end_type = self.EVENT_PAIRS[event_type]

                if end_type:
                    end_row = self._find_matching_end(
                        df, i, end_type, row["其他干预原因"]
                    )

                    if end_row is not None:
                        period = EventPeriod(
                            start_time=row["时间"],
                            end_time=end_row["时间"],
                            event_type=event_type,
                            device=row["设备"],
                            reason=row["其他干预原因"],
                        )
                        event_periods.append(period)
                        LOGGER.debug(f"提取事件期: {period}")
                        i = df.index.get_loc(end_row.name) + 1
                        continue

                if event_type == "MFT预警":
                    period = EventPeriod(
                        start_time=row["时间"] - pd.Timedelta(minutes=10),
                        end_time=row["时间"] + pd.Timedelta(minutes=30),
                        event_type=event_type,
                        device=row["设备"],
                        reason=row["其他干预原因"],
                    )
                    event_periods.append(period)
                    LOGGER.debug(f"提取MFT预警事件期: {period}")

            i += 1

        return event_periods

    def _find_matching_end(
        self, df: pd.DataFrame, start_idx: int, end_type: str, reason: str
    ) -> pd.DataFrame:
        """查找匹配的结束事件"""
        for j in range(start_idx + 1, len(df)):
            end_row = df.iloc[j]
            if end_row["特殊工况"] == end_type and end_row["其他干预原因"] == reason:
                return end_row
        return None


class EventPeriodCleaner:
    """事件期数据清洗器"""

    def __init__(self, event_periods: List[EventPeriod]):
        self.event_periods = event_periods

    def mark_event_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """标记事件期内的数据

        添加以下字段:
        - is_event_period: 是否在事件期内
        - event_type: 事件类型
        - event_device: 相关设备
        - event_reason: 事件原因
        """
        LOGGER.info("开始标记事件期数据...")

        df["is_event_period"] = False
        df["event_type"] = None
        df["event_device"] = None
        df["event_reason"] = None

        df["TIME"] = pd.to_datetime(df["TIME"])

        marked_count = 0
        for period in self.event_periods:
            mask = (df["TIME"] >= period.start_time) & (df["TIME"] <= period.end_time)
            marked_count += mask.sum()

            df.loc[mask, "is_event_period"] = True
            df.loc[mask, "event_type"] = period.event_type
            df.loc[mask, "event_device"] = period.device
            df.loc[mask, "event_reason"] = period.reason

        LOGGER.info(
            f"标记完成: {marked_count} 行数据被标记为事件期 "
            f"(占比 {marked_count/len(df)*100:.2f}%)"
        )

        return df

    def remove_event_periods(self, df: pd.DataFrame) -> pd.DataFrame:
        """删除事件期内的数据"""
        LOGGER.info("开始删除事件期数据...")

        df = self.mark_event_periods(df)

        rows_before = len(df)
        df = df[~df["is_event_period"]].copy()
        rows_after = len(df)

        df = df.drop(
            columns=["is_event_period", "event_type", "event_device", "event_reason"]
        )

        LOGGER.info(
            f"删除完成: {rows_before} -> {rows_after} "
            f"(删除了 {rows_before - rows_after} 行)"
        )

        return df

    def get_event_period_summary(self) -> pd.DataFrame:
        """获取事件期摘要统计"""
        summary_data = []

        for period in self.event_periods:
            summary_data.append(
                {
                    "开始时间": period.start_time,
                    "结束时间": period.end_time,
                    "时长(小时)": f"{period.duration.total_seconds() / 3600:.2f}",
                    "事件类型": period.event_type,
                    "设备": period.device,
                    "原因": period.reason,
                }
            )
        return pd.DataFrame(summary_data)

    def clean_event_periods(self, df: pd.DataFrame, method: str = "mark") -> pd.DataFrame:
        """清洗事件期数据

        Args:
            df: 原始数据
            method: 清洗方法
                - "mark": 仅标记,不删除(默认)
                - "remove": 删除事件期数据

        Returns:
            清洗后的数据
        """
        if method == "mark":
            return self.mark_event_periods(df)
        elif method == "remove":
            return self.remove_event_periods(df)
        else:
            raise ValueError(f"未知的清洗方法: {method}")

def load_intervention_records(data_dir: Path) -> List[EventPeriod]:
    """加载人工干预记录

    Args:
        data_dir: 数据目录路径

    Returns:
        事件期列表
    """
    record_files = list(data_dir.glob("*人工干预记录表.xlsx"))

    if not record_files:
        LOGGER.warning(f"在 {data_dir} 下未找到人工干预记录表")
        return []

    parser = InterventionRecordParser(record_files[0])
    return parser.parse()


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "川宁项目4锅炉2号机组-60秒-20260308"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)

    event_periods = load_intervention_records(DATA_DIR)

    summary_df = EventPeriodCleaner(event_periods).get_event_period_summary()
    with open(OUTPUT_DIR / "event_period_summary.csv", "w") as f:
        f.write(summary_df.to_csv(index=False))
    LOGGER.info(f"事件期摘要已保存至 {OUTPUT_DIR / 'event_period_summary.csv'}")