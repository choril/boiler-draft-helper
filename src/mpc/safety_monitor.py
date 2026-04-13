"""
安全监控 - MPC约束检查和告警

功能：
1. 控制约束检查（边界、调整幅度）
2. 预测安全检查（负压、含氧）
3. 运行状态监控
4. 异常告警和处理建议
5. 负压跳变检测（新增）- 检测人工干预/故障事件
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.config.constraints import (
    PRESSURE_SAFETY_RANGE,
    OXYGEN_SAFETY_RANGE,
    PRESSURE_ALARM_THRESHOLD,
    OXYGEN_ALARM_THRESHOLD,
    BED_TEMP_SAFETY_RANGE,
    BED_TEMP_FLUCTUATION_LIMIT,
    CONTROL_RANGES,
    MAX_SINGLE_ADJUSTMENT,
    check_safety_status,
    check_control_validity,
)
from src.config.variables import CONTROL_VARIABLES, PRESSURE_VARIABLES, OXYGEN_VARIABLES
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SafetyLevel(Enum):
    """安全等级"""
    NORMAL = 0      # 正常运行
    WARNING = 1     # 警告（需关注）
    ALARM = 2       # 告警（需立即处理）
    CRITICAL = 3    # 严重（暂停优化）


class PressureJumpType(Enum):
    """负压跳变类型（根据调研文档）"""
    NORMAL_FLUCTUATION = 0    # 正常波动 (20-40Pa)
    OPERATION_EVENT = 1       # 操作事件 (80-200Pa): 负荷调整、吹灰、除尘器切换
    FAULT_EVENT = 2           # 故障事件 (500-1500Pa): 脱硫系统问题
    CRITICAL_EVENT = 3        # 严重事件 (1500-2000Pa): 设备跳闸、爆管、风机异常


# 跳变阈值（Pa/秒）
PRESSURE_JUMP_THRESHOLDS = {
    'normal': 5,        # <5 Pa/s 正常波动
    'operation': 20,    # 20-80 Pa/s 操作事件
    'fault': 80,        # 80-150 Pa/s 故障事件
    'critical': 150,    # >150 Pa/s 严重事件
}


@dataclass
class SafetyCheckResult:
    """安全检查结果"""
    level: SafetyLevel
    pressure_status: dict
    oxygen_status: dict
    control_status: dict
    issues: list[str]
    recommendations: list[str]


class SafetyMonitor:
    """安全监控器

    实时监控锅炉运行状态和MPC优化结果
    """

    def __init__(
        self,
        pressure_alarm_threshold: float = PRESSURE_ALARM_THRESHOLD,
        oxygen_alarm_threshold: float = OXYGEN_ALARM_THRESHOLD,
        bed_temp_fluctuation_limit: float = BED_TEMP_FLUCTUATION_LIMIT,
    ):
        """
        Args:
            pressure_alarm_threshold: 负压告警阈值
            oxygen_alarm_threshold: 含氧告警阈值
            bed_temp_fluctuation_limit: 床温波动限制
        """
        self.pressure_alarm_threshold = pressure_alarm_threshold
        self.oxygen_alarm_threshold = oxygen_alarm_threshold
        self.bed_temp_fluctuation_limit = bed_temp_fluctuation_limit

        # 床温历史（用于波动检查）
        self.bed_temp_history: list[float] = []
        self.bed_temp_window = 30  # 30分钟窗口

        # 负压历史（用于跳变检测）- 新增
        self.pressure_history: list[float] = []
        self.pressure_timestamps: list[float] = []
        self.pressure_window = 90  # 90秒窗口（跳变持续时间）
        self.last_jump_type: Optional[PressureJumpType] = None
        self.jump_detected_time: Optional[float] = None

    def detect_pressure_jump(
        self,
        current_pressure: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[PressureJumpType, float]:
        """检测负压跳变（新增）

        根据调研文档的跳变类别：
        - 正常波动 (20-40Pa): 烟气压力干扰
        - 操作事件 (80-200Pa): 负荷调整、吹灰、除尘器切换
        - 故障事件 (500-1500Pa): 脱硫系统问题
        - 严重事件 (1500-2000Pa): 设备跳闸、爆管

        Args:
            current_pressure: 当前负压值 (4,) 或单值
            timestamp: 当前时间戳（秒）

        Returns:
            jump_type: 跳变类型
            change_rate: 变化率 (Pa/秒)
        """
        # 处理输入
        if isinstance(current_pressure, np.ndarray) and current_pressure.ndim > 0:
            pressure_mean = np.mean(current_pressure)
        else:
            pressure_mean = float(current_pressure)

        # 更新历史
        if timestamp is None:
            timestamp = len(self.pressure_history)  # 用序号代替时间

        self.pressure_history.append(pressure_mean)
        self.pressure_timestamps.append(timestamp)

        # 保持窗口大小
        if len(self.pressure_history) > self.pressure_window:
            self.pressure_history.pop(0)
            self.pressure_timestamps.pop(0)

        # 计算变化率
        if len(self.pressure_history) < 2:
            return PressureJumpType.NORMAL_FLUCTUATION, 0.0

        # 使用最近的几个点计算变化率
        recent_window = min(10, len(self.pressure_history))  # 最近10秒
        recent_pressures = self.pressure_history[-recent_window:]
        recent_times = self.pressure_timestamps[-recent_window:]

        # 最大变化幅度
        max_pressure = max(recent_pressures)
        min_pressure = min(recent_pressures)
        pressure_change = abs(max_pressure - min_pressure)

        # 时间跨度
        time_span = recent_times[-1] - recent_times[0] if len(recent_times) > 1 else recent_window

        # 变化率 (Pa/秒)
        change_rate = pressure_change / max(time_span, 1.0)

        # 判断跳变类型
        if change_rate < PRESSURE_JUMP_THRESHOLDS['normal']:
            jump_type = PressureJumpType.NORMAL_FLUCTUATION
        elif change_rate < PRESSURE_JUMP_THRESHOLDS['operation']:
            jump_type = PressureJumpType.OPERATION_EVENT
            logger.warning(f"负压操作事件检测: 变化率={change_rate:.1f} Pa/s, 幅度={pressure_change:.1f} Pa")
        elif change_rate < PRESSURE_JUMP_THRESHOLDS['fault']:
            jump_type = PressureJumpType.FAULT_EVENT
            logger.warning(f"负压故障事件检测: 变化率={change_rate:.1f} Pa/s, 幅度={pressure_change:.1f} Pa")
        else:
            jump_type = PressureJumpType.CRITICAL_EVENT
            logger.error(f"负压严重事件检测: 变化率={change_rate:.1f} Pa/s, 幅度={pressure_change:.1f} Pa")

        # 记录跳变
        if jump_type != PressureJumpType.NORMAL_FLUCTUATION:
            self.last_jump_type = jump_type
            self.jump_detected_time = timestamp

        return jump_type, change_rate

    def check_current_state(
        self,
        current_pressure: np.ndarray,
        current_oxygen: np.ndarray,
        current_bed_temp: Optional[float] = None,
    ) -> SafetyCheckResult:
        """检查当前运行状态

        Args:
            current_pressure: 当前负压值 (4,) 或单值
            current_oxygen: 当前含氧值 (3,) 或单值
            current_bed_temp: 当前床温（可选）

        Returns:
            SafetyCheckResult: 检查结果
        """
        # 处理输入
        if isinstance(current_pressure, np.ndarray) and current_pressure.ndim > 0:
            pressure_mean = np.mean(current_pressure)
        else:
            pressure_mean = float(current_pressure)

        if isinstance(current_oxygen, np.ndarray) and current_oxygen.ndim > 0:
            oxygen_mean = np.mean(current_oxygen)
        else:
            oxygen_mean = float(current_oxygen)

        # 负压状态
        pressure_status = {
            'value': pressure_mean,
            'in_safe_range': PRESSURE_SAFETY_RANGE[0] <= pressure_mean <= PRESSURE_SAFETY_RANGE[1],
            'alarm': pressure_mean > self.pressure_alarm_threshold,
        }

        # 含氧状态
        oxygen_status = {
            'value': oxygen_mean,
            'in_safe_range': OXYGEN_SAFETY_RANGE[0] <= oxygen_mean <= OXYGEN_SAFETY_RANGE[1],
            'alarm': oxygen_mean < self.oxygen_alarm_threshold,
        }

        # 床温状态
        bed_temp_status = {}
        if current_bed_temp is not None:
            # 更新历史
            self.bed_temp_history.append(current_bed_temp)
            if len(self.bed_temp_history) > self.bed_temp_window:
                self.bed_temp_history.pop(0)

            # 波动检查
            if len(self.bed_temp_history) >= 10:
                fluctuation = np.std(self.bed_temp_history)
                bed_temp_status['fluctuation'] = fluctuation
                bed_temp_status['fluctuation_alarm'] = fluctuation > self.bed_temp_fluctuation_limit

            # 范围检查
            bed_temp_status['value'] = current_bed_temp
            bed_temp_status['in_safe_range'] = (
                BED_TEMP_SAFETY_RANGE[0] <= current_bed_temp <= BED_TEMP_SAFETY_RANGE[1]
            )

        # 收集问题
        issues = []
        if pressure_status['alarm']:
            issues.append(f"负压告警: {pressure_mean:.1f} Pa > {self.pressure_alarm_threshold} Pa")
        if not pressure_status['in_safe_range']:
            issues.append(f"负压超出安全范围: {pressure_mean:.1f} Pa")
        if oxygen_status['alarm']:
            issues.append(f"含氧告警: {oxygen_mean:.2f} % < {self.oxygen_alarm_threshold} %")
        if not oxygen_status['in_safe_range']:
            issues.append(f"含氧超出安全范围: {oxygen_mean:.2f} %")
        if bed_temp_status.get('fluctuation_alarm', False):
            issues.append(f"床温波动过大: {bed_temp_status['fluctuation']:.1f} ℃")

        # 确定安全等级
        if pressure_status['alarm'] or oxygen_status['alarm']:
            level = SafetyLevel.ALARM
        elif not pressure_status['in_safe_range'] or not oxygen_status['in_safe_range']:
            level = SafetyLevel.WARNING
        elif bed_temp_status.get('fluctuation_alarm', False):
            level = SafetyLevel.WARNING
        elif issues:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.NORMAL

        # 控制状态（初始为空）
        control_status = {'valid': True, 'issues': []}

        # 建议
        recommendations = self._generate_recommendations(level, pressure_status, oxygen_status, bed_temp_status)

        return SafetyCheckResult(
            level=level,
            pressure_status=pressure_status,
            oxygen_status=oxygen_status,
            control_status=control_status,
            issues=issues,
            recommendations=recommendations,
        )

    def check_control_sequence(
        self,
        control_sequence: np.ndarray,
        current_control: Optional[np.ndarray] = None,
    ) -> SafetyCheckResult:
        """检查控制序列

        Args:
            control_sequence: 控制序列 (H, n_u)
            current_control: 当前控制值 (n_u,)（可选）

        Returns:
            SafetyCheckResult: 检查结果
        """
        H = control_sequence.shape[0]
        issues = []

        # 边界检查
        for step in range(H):
            for i, var in enumerate(CONTROL_VARIABLES):
                value = control_sequence[step, i]
                if var in CONTROL_RANGES:
                    low, high = CONTROL_RANGES[var]
                    if value < low or value > high:
                        issues.append(f"步骤{step+1}: {var}={value:.1f} 超出范围 [{low}, {high}]")

        # 调整幅度检查
        if current_control is not None:
            for i, var in enumerate(CONTROL_VARIABLES):
                change = abs(control_sequence[0, i] - current_control[i])

                # 确定调整限制
                if var == 'D62AX002':  # 给煤量
                    max_change = MAX_SINGLE_ADJUSTMENT['coal_feed']
                else:  # 风机频率
                    max_change = MAX_SINGLE_ADJUSTMENT['fan_freq']

                if change > max_change:
                    issues.append(f"第一步: {var}变化{change:.1f} > {max_change}")

        # 步间变化检查
        for step in range(1, H):
            changes = np.abs(control_sequence[step] - control_sequence[step - 1])
            for i, var in enumerate(CONTROL_VARIABLES):
                if var == 'D62AX002':
                    max_change = MAX_SINGLE_ADJUSTMENT['coal_feed']
                else:
                    max_change = MAX_SINGLE_ADJUSTMENT['fan_freq']

                if changes[i] > max_change:
                    issues.append(f"步骤{step}→{step+1}: {var}变化{changes[i]:.1f} > {max_change}")

        # 确定安全等级
        if issues:
            level = SafetyLevel.WARNING
        else:
            level = SafetyLevel.NORMAL

        # 控制状态
        control_status = {
            'valid': len(issues) == 0,
            'issues': issues,
            'sequence': control_sequence,
        }

        return SafetyCheckResult(
            level=level,
            pressure_status={},
            oxygen_status={},
            control_status=control_status,
            issues=issues,
            recommendations=["调整控制序列以满足约束"] if issues else [],
        )

    def check_prediction(
        self,
        prediction: np.ndarray,
    ) -> SafetyCheckResult:
        """检查预测值

        Args:
            prediction: 预测序列 (H, n_y)

        Returns:
            SafetyCheckResult: 检查结果
        """
        H = prediction.shape[0]
        issues = []

        # 负压预测检查
        for step in range(H):
            pressure_pred = prediction[step, :4]  # 4个负压测点
            pressure_mean = np.mean(pressure_pred)

            if pressure_mean > self.pressure_alarm_threshold:
                issues.append(f"预测步骤{step+1}: 负压{pressure_mean:.1f} Pa > 告警阈值")
            if pressure_mean > PRESSURE_SAFETY_RANGE[1] or pressure_mean < PRESSURE_SAFETY_RANGE[0]:
                issues.append(f"预测步骤{step+1}: 负压{pressure_mean:.1f} Pa 超出安全范围")

        # 含氧预测检查
        for step in range(H):
            oxygen_pred = prediction[step, 4:7]  # 3个含氧测点
            oxygen_mean = np.mean(oxygen_pred)

            if oxygen_mean < self.oxygen_alarm_threshold:
                issues.append(f"预测步骤{step+1}: 含氧{oxygen_mean:.2f} % < 告警阈值")
            if oxygen_mean > OXYGEN_SAFETY_RANGE[1] or oxygen_mean < OXYGEN_SAFETY_RANGE[0]:
                issues.append(f"预测步骤{step+1}: 含氧{oxygen_mean:.2f} % 超出安全范围")

        # 确定安全等级
        alarm_count = sum(1 for issue in issues if "告警" in issue)
        if alarm_count > 0:
            level = SafetyLevel.WARNING  # 预测告警只是警告，不是实际告警
        else:
            level = SafetyLevel.NORMAL

        return SafetyCheckResult(
            level=level,
            pressure_status={'prediction': prediction[:, :4]},
            oxygen_status={'prediction': prediction[:, 4:7]},
            control_status={},
            issues=issues,
            recommendations=[],
        )

    def _generate_recommendations(
        self,
        level: SafetyLevel,
        pressure_status: dict,
        oxygen_status: dict,
        bed_temp_status: dict,
    ) -> list[str]:
        """生成处理建议"""
        recommendations = []

        if level == SafetyLevel.ALARM:
            if pressure_status.get('alarm', False):
                recommendations.append("立即增加引风机频率，降低炉膛负压")
            if oxygen_status.get('alarm', False):
                recommendations.append("立即增加二次风机频率，提高含氧量")

        elif level == SafetyLevel.WARNING:
            if not pressure_status.get('in_safe_range', True):
                recommendations.append("调整引风机以维持负压在安全范围")
            if not oxygen_status.get('in_safe_range', True):
                recommendations.append("调整二次风机以维持含氧在安全范围")
            if bed_temp_status.get('fluctuation_alarm', False):
                recommendations.append("床温波动较大，暂停优化，保持当前控制")

        return recommendations

    def should_pause_optimization(self, result: SafetyCheckResult) -> bool:
        """判断是否应暂停优化

        Args:
            result: 安全检查结果

        Returns:
            pause: 是否暂停
        """
        # 严重告警时暂停
        if result.level == SafetyLevel.ALARM:
            return True

        # 床温波动过大时暂停
        if result.pressure_status.get('fluctuation_alarm', False):
            return True

        # 检测到故障/严重跳变事件时暂停（新增）
        if self.last_jump_type in [PressureJumpType.FAULT_EVENT, PressureJumpType.CRITICAL_EVENT]:
            # 检查跳变是否仍在持续（90秒内）
            if self.jump_detected_time is not None:
                current_time = len(self.pressure_history)
                if current_time - self.jump_detected_time < 90:
                    logger.warning("跳变事件持续，暂停MPC优化")
                    return True
                else:
                    # 跳变结束，清除记录
                    self.last_jump_type = None
                    self.jump_detected_time = None
                    logger.info("跳变事件结束，可恢复MPC优化")

        return False


def create_safety_monitor() -> SafetyMonitor:
    """创建安全监控器"""
    return SafetyMonitor()


__all__ = [
    "SafetyLevel",
    "SafetyCheckResult",
    "SafetyMonitor",
    "PressureJumpType",  # 新增
    "PRESSURE_JUMP_THRESHOLDS",  # 新增
    "create_safety_monitor",
]