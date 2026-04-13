"""
变量定义 - 目标变量Y、控制输入U、状态变量X

变量分类：
- Y: 目标变量（7维）- 4个负压测点 + 3个含氧测点
- U: 控制输入（7维）- 风机频率和给煤量
- X: 状态变量（约50维）- 床温、风量、负荷等
"""

from typing import Final

# ========== 目标变量 Y（7维）==========

# 负压测点（4维）
PRESSURE_VARIABLES: Final[list[str]] = [
    "2BK10CP004",  # 炉膛压力1（主测点）
    "2BK2CP004",   # 炉膛压力2
    "2BK10CP005",  # 炉膛压力3
    "2BK2CP005",   # 炐膛压力4
]

# 含氧量测点（3维）
OXYGEN_VARIABLES: Final[list[str]] = [
    "2BK10CQ1",  # 含氧量1（主测点）
    "2BK2CQ1",   # 含氧量2
    "2BK2CQ2",   # 含氧量3
]

# 全部目标变量
TARGET_VARIABLES: Final[list[str]] = PRESSURE_VARIABLES + OXYGEN_VARIABLES

# 主要目标变量（用于优化目标）
MAIN_TARGETS: Final[list[str]] = [
    "2BK10CP004",  # 主负压测点
    "2BK10CQ1",    # 主含氧测点
]


# ========== 控制输入 U（7维）==========

# 风机控制（频率）
CONTROL_VARIABLES: Final[list[str]] = [
    # 一次风机（快速提升负荷）
    "2LA10A12C11",  # 一次风机A输出频率
    "2LA20A12C11",  # 一次风机B输出频率
    # 二次风机（调氧量）
    "2LA30A12C11",  # 二次风机A输出频率
    "2LA40A12C11",  # 二次风机B输出频率
    # 引风机（调负压）
    "DPU61AX107",   # 引风机A输出频率
    "DPU61AX108",   # 引风机B输出频率
    # 给煤量
    "D62AX002",     # 给煤量
]

# 控制变量分组（便于MPC分组优化）
FAN_CONTROL_GROUPS: Final[dict] = {
    "primary_fan": ["2LA10A12C11", "2LA20A12C11"],   # 一次风机
    "secondary_fan": ["2LA30A12C11", "2LA40A12C11"], # 二次风机
    "induced_fan": ["DPU61AX107", "DPU61AX108"],     # 引风机
    "coal_feed": ["D62AX002"],                        # 给煤量
}

# 控制变量的物理含义映射
CONTROL_MEANINGS: Final[dict] = {
    "2LA10A12C11": "一次风机A频率",
    "2LA20A12C11": "一次风机B频率",
    "2LA30A12C11": "二次风机A频率",
    "2LA40A12C11": "二次风机B频率",
    "DPU61AX107": "引风机A频率",
    "DPU61AX108": "引风机B频率",
    "D62AX002": "给煤量",
}


# ========== 状态变量 X（约50维）==========

# 核心状态变量（必须包含）
CORE_STATE_VARIABLES: Final[list[str]] = [
    "MSFLOW",      # 主蒸汽流量（负荷）
    "D66P53A10",   # 床温（燃烧状态）
    "D61AX023",    # 一次风量
    "D61AX024",    # 二次风量
]

# 风机相关状态变量
FAN_STATE_VARIABLES: Final[list[str]] = [
    # 一次风机状态
    "2LB10CS001",  # 一次风机A转速
    "2LB20CS001",  # 一次风机B转速
    "2BBA14Q11",   # 一次风机A电流
    "2BBB12Q11",   # 一次风机B电流
    # 二次风机状态
    "2LB30CS901",  # 二次风机A转速
    "2LB40CS901",  # 二次风机B转速
    "2BBA13Q11",   # 二次风机A电流
    "2BBB11Q11",   # 二次风机B电流
    # 引风机状态
    "2NC10CS901",  # 引风机A转速
    "2NC2CS901",   # 引风机B转速
    "2BBA15Q11",   # 引风机A电流
    "2BBB13Q11",   # 引风机B电流
]

# 温度相关状态变量
TEMP_STATE_VARIABLES: Final[list[str]] = [
    "2LA10CT11",   # 一次风机A出口温度
    "2LA2CT11",    # 一次风机B出口温度
]

# 排除的变量（不用于建模）
EXCLUDE_VARIABLES: Final[list[str]] = [
    "TIME",        # 时间戳
    "source_file", # 数据来源文件
]

# 状态变量需要从数据中动态筛选，排除 Y、U、EXCLUDE
# 最终状态变量列表 = 所有列 - TARGET_VARIABLES - CONTROL_VARIABLES - EXCLUDE_VARIABLES


# ========== 变量维度 ==========

N_TARGETS: Final[int] = len(TARGET_VARIABLES)      # 7
N_CONTROLS: Final[int] = len(CONTROL_VARIABLES)    # 7
N_CORE_STATES: Final[int] = len(CORE_STATE_VARIABLES)  # 4


# ========== 辅助函数 ==========

def get_state_variables(all_columns: list[str]) -> list[str]:
    """从所有列中筛选状态变量

    Args:
        all_columns: 数据框的所有列名

    Returns:
        状态变量列表（排除时间、目标、控制变量）
    """
    exclude = EXCLUDE_VARIABLES + TARGET_VARIABLES + CONTROL_VARIABLES
    return [col for col in all_columns if col not in exclude]


def classify_variable(var_name: str) -> str:
    """判断变量类型

    Args:
        var_name: 变量名

    Returns:
        变量类型: 'target', 'control', 'state', 'exclude', 'unknown'
    """
    if var_name in EXCLUDE_VARIABLES:
        return 'exclude'
    elif var_name in TARGET_VARIABLES:
        return 'target'
    elif var_name in CONTROL_VARIABLES:
        return 'control'
    else:
        return 'state'


__all__ = [
    "PRESSURE_VARIABLES",
    "OXYGEN_VARIABLES",
    "TARGET_VARIABLES",
    "MAIN_TARGETS",
    "CONTROL_VARIABLES",
    "FAN_CONTROL_GROUPS",
    "CONTROL_MEANINGS",
    "CORE_STATE_VARIABLES",
    "FAN_STATE_VARIABLES",
    "TEMP_STATE_VARIABLES",
    "EXCLUDE_VARIABLES",
    "N_TARGETS",
    "N_CONTROLS",
    "N_CORE_STATES",
    "get_state_variables",
    "classify_variable",
]