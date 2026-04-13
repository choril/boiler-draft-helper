"""
物理约束定义 - 控制约束、安全约束、理想范围

约束分类：
- 控制约束：风机频率范围、给煤量范围、单步调整幅度
- 安全约束：负压告警、含氧告警、床温波动限制
- 理想范围：负压理想区间、含氧理想区间、目标值
"""

from typing import Final


# ========== 控制约束 ==========

# 风机频率范围（Hz）
FAN_FREQ_RANGE: Final[tuple[float, float]] = (10.0, 50.0)

# 给煤量范围（t/h）
COAL_FEED_RANGE: Final[tuple[float, float]] = (30.0, 100.0)

# 各控制变量的物理范围
CONTROL_RANGES: Final[dict[str, tuple[float, float]]] = {
    "2LA10A12C11": FAN_FREQ_RANGE,  # 一次风机A
    "2LA20A12C11": FAN_FREQ_RANGE,  # 一次风机B
    "2LA30A12C11": FAN_FREQ_RANGE,  # 二次风机A
    "2LA40A12C11": FAN_FREQ_RANGE,  # 二次风机B
    "DPU61AX107": FAN_FREQ_RANGE,   # 引风机A
    "DPU61AX108": FAN_FREQ_RANGE,   # 引风机B
    "D62AX002": COAL_FEED_RANGE,    # 给煤量
}

# 单步调整幅度限制
MAX_SINGLE_ADJUSTMENT: Final[dict[str, float]] = {
    "fan_freq": 5.0,   # 风机频率最大单步调整（Hz）
    "coal_feed": 5.0,  # 给煤量最大单步调整（t/h）
}

# 控制变量变化率约束（用于平滑控制）
CONTROL_CHANGE_RATE_LIMIT: Final[dict[str, float]] = {
    "2LA10A12C11": 5.0,  # Hz/min
    "2LA20A12C11": 5.0,
    "2LA30A12C11": 5.0,
    "2LA40A12C11": 5.0,
    "DPU61AX107": 5.0,
    "DPU61AX108": 5.0,
    "D62AX002": 5.0,     # t/h/min
}


# ========== 安全约束 ==========

# 负压安全范围（Pa）
# 负压过低（接近0或正压）会导致烟气回流，危险！
PRESSURE_SAFETY_RANGE: Final[tuple[float, float]] = (-200.0, -20.0)

# 负压告警阈值
PRESSURE_ALARM_THRESHOLD: Final[float] = -20.0  # Pa (负压过小告警)

# 含氧量安全范围（%）
# 含氧过低会导致燃烧不充分，危险！
OXYGEN_SAFETY_RANGE: Final[tuple[float, float]] = (1.0, 6.0)

# 含氧告警阈值
OXYGEN_ALARM_THRESHOLD: Final[float] = 1.0  # % (含氧过低告警)

# 床温波动限制（℃）
BED_TEMP_FLUCTUATION_LIMIT: Final[float] = 50.0

# 床温安全范围（℃）
BED_TEMP_SAFETY_RANGE: Final[tuple[float, float]] = (800.0, 950.0)


# ========== 理想运行范围 ==========

# 负压理想范围（Pa）
PRESSURE_IDEAL_RANGE: Final[tuple[float, float]] = (-150.0, -80.0)

# 负压目标值（Pa）
PRESSURE_TARGET: Final[float] = -115.0

# 含氧量理想范围（%）
OXYGEN_IDEAL_RANGE: Final[tuple[float, float]] = (1.5, 2.5)

# 含氧量目标值（%）
OXYGEN_TARGET: Final[float] = 2.0

# 给煤量理想值（t/h）
COAL_FEED_IDEAL: Final[float] = 68.0

# 给煤量正常值（t/h）
COAL_FEED_NORMAL: Final[float] = 40.0


# ========== 风煤比约束 ==========

# 风煤比范围（风量/给煤量）
# 根据燃烧效率确定的风煤比匹配范围
COAL_AIR_RATIO_RANGES: Final[list[tuple[float, float, float, float]]] = [
    # (给煤量下限, 给煤量上限, 一次风量范围, 二次风量范围)
    (0.0, 20.0, 40000.0, 60000.0),
    (20.0, 45.0, 75000.0, 130000.0),
    (45.0, 68.0, 180000.0, 160000.0),
]


# ========== MPC约束结构 ==========

CONTROL_CONSTRAINTS: Final[dict] = {
    "ranges": CONTROL_RANGES,
    "max_adjustment": MAX_SINGLE_ADJUSTMENT,
    "change_rate_limit": CONTROL_CHANGE_RATE_LIMIT,
}

SAFETY_CONSTRAINTS: Final[dict] = {
    "pressure_range": PRESSURE_SAFETY_RANGE,
    "pressure_alarm": PRESSURE_ALARM_THRESHOLD,
    "oxygen_range": OXYGEN_SAFETY_RANGE,
    "oxygen_alarm": OXYGEN_ALARM_THRESHOLD,
    "bed_temp_range": BED_TEMP_SAFETY_RANGE,
    "bed_temp_fluctuation": BED_TEMP_FLUCTUATION_LIMIT,
}

IDEAL_RANGES: Final[dict] = {
    "pressure": PRESSURE_IDEAL_RANGE,
    "pressure_target": PRESSURE_TARGET,
    "oxygen": OXYGEN_IDEAL_RANGE,
    "oxygen_target": OXYGEN_TARGET,
    "coal_ideal": COAL_FEED_IDEAL,
    "coal_normal": COAL_FEED_NORMAL,
}


# ========== 辅助函数 ==========

def check_control_validity(control_values: dict[str, float]) -> dict[str, list[str]]:
    """检查控制变量是否在有效范围内

    Args:
        control_values: 控制变量值字典

    Returns:
        问题列表 {'out_of_range': [...], 'exceeds_adjustment': [...]}
    """
    issues = {'out_of_range': [], 'exceeds_adjustment': []}

    for var, value in control_values.items():
        if var in CONTROL_RANGES:
            low, high = CONTROL_RANGES[var]
            if value < low or value > high:
                issues['out_of_range'].append(f"{var}: {value} not in [{low}, {high}]")

    return issues


def check_safety_status(
    pressure: float,
    oxygen: float,
    bed_temp: float | None = None,
) -> dict:
    """检查当前安全状态

    Args:
        pressure: 负压值（Pa）
        oxygen: 含氧量（%）
        bed_temp: 床温（℃），可选

    Returns:
        安全状态字典 {'pressure_alarm': bool, 'oxygen_alarm': bool, ...}
    """
    status = {
        'pressure_alarm': pressure > PRESSURE_ALARM_THRESHOLD,
        'oxygen_alarm': oxygen < OXYGEN_ALARM_THRESHOLD,
        'pressure_in_safe': PRESSURE_SAFETY_RANGE[0] <= pressure <= PRESSURE_SAFETY_RANGE[1],
        'oxygen_in_safe': OXYGEN_SAFETY_RANGE[0] <= oxygen <= OXYGEN_SAFETY_RANGE[1],
        'pressure_in_ideal': PRESSURE_IDEAL_RANGE[0] <= pressure <= PRESSURE_IDEAL_RANGE[1],
        'oxygen_in_ideal': OXYGEN_IDEAL_RANGE[0] <= oxygen <= OXYGEN_IDEAL_RANGE[1],
        'overall_safe': True,
    }

    # 综合安全判断
    status['overall_safe'] = (
        status['pressure_in_safe'] and
        status['oxygen_in_safe'] and
        not status['pressure_alarm'] and
        not status['oxygen_alarm']
    )

    if bed_temp is not None:
        status['bed_temp_in_safe'] = (
            BED_TEMP_SAFETY_RANGE[0] <= bed_temp <= BED_TEMP_SAFETY_RANGE[1]
        )
        status['overall_safe'] = status['overall_safe'] and status['bed_temp_in_safe']

    return status


def calculate_pressure_deviation(pressure: float) -> float:
    """计算负压与目标值的偏差

    Args:
        pressure: 负压值（Pa）

    Returns:
        偏差值（绝对值）
    """
    return abs(pressure - PRESSURE_TARGET)


def calculate_oxygen_deviation(oxygen: float) -> float:
    """计算含氧量与目标值的偏差

    Args:
        oxygen: 含氧量（%）

    Returns:
        偏差值（绝对值）
    """
    return abs(oxygen - OXYGEN_TARGET)


__all__ = [
    "FAN_FREQ_RANGE",
    "COAL_FEED_RANGE",
    "CONTROL_RANGES",
    "MAX_SINGLE_ADJUSTMENT",
    "CONTROL_CHANGE_RATE_LIMIT",
    "PRESSURE_SAFETY_RANGE",
    "PRESSURE_ALARM_THRESHOLD",
    "OXYGEN_SAFETY_RANGE",
    "OXYGEN_ALARM_THRESHOLD",
    "BED_TEMP_FLUCTUATION_LIMIT",
    "BED_TEMP_SAFETY_RANGE",
    "PRESSURE_IDEAL_RANGE",
    "PRESSURE_TARGET",
    "OXYGEN_IDEAL_RANGE",
    "OXYGEN_TARGET",
    "COAL_FEED_IDEAL",
    "COAL_FEED_NORMAL",
    "COAL_AIR_RATIO_RANGES",
    "CONTROL_CONSTRAINTS",
    "SAFETY_CONSTRAINTS",
    "IDEAL_RANGES",
    "check_control_validity",
    "check_safety_status",
    "calculate_pressure_deviation",
    "calculate_oxygen_deviation",
]