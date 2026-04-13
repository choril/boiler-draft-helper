from typing import Final

# ========== 目标变量 ==========
PRESSURE_VARIABLES: Final[list[str]] = [
    "2BK10CP004",  # 炉膛压力1
    "2BK2CP004",  # 炉膛压力2
    "2BK10CP005",  # 炉膛压力3
    "2BK2CP005",  # 炉膛压力4
]

OXYGEN_VARIABLES: Final[list[str]] = [
    "2BK10CQ1",  # 含氧量1
    "2BK2CQ1",  # 含氧量2
    "2BK2CQ2",  # 含氧量3
]


TARGET_VARIABLES: Final[list[str]] = PRESSURE_VARIABLES + OXYGEN_VARIABLES

EXCLUDE_VARIABLES: Final[list[str]] = ["TIME", "source_file"] + TARGET_VARIABLES

# ========= 非重要参数（共22个）=========
UNIMPORTANT_PARAMS: Final[list[str]] = [
    "2BK10CP01",
    "2BK10CP11",
    "2BK10CP12",
    "2BK2CP01",
    "2BK2CP11",
    "2BK2CP12",
    "2BK10CT226",
    "2BK10CT229",
    "2BK10CT232",
    "2BK2CT226",
    "2BK2CT232",
    "2LA30CT11",
    "2LA40CT11",
    "2NC20A11C01",
    "D63P74B1",
    "2NC10A11C01",
    "BPAFCDMD",
    "D64P62B1",
    "APAFCDMD",
    "2LA40CP01",
    "D63P71B1",
    "2LA30A11C01"
]

# ========== 关键参数（核心监测指标）==========
KEY_PARAMS: Final[list[str]] = [
    # 给煤量 - 主要调控动作
    "D62AX002",
    # 床温 - 燃烧状态关键指标
    "D66P53A10",
    # 风量 - 燃烧配比
    "D61AX023",  # 一次风风量
    "D61AX024",  # 二次风风量
    # 目标变量
    "2BK10CP004",  # 炉膛压力（主目标）
    "2BK10CQ1",    # 含氧量（主目标）
    # 主蒸汽 - 负荷指标
    "MSFLOW",
    # 出风温 - 风机出口温度
    "2LA10CT11",
    "2LA2CT11",
]

# ========== 控制参数（工人调节的参数）==========
CONTROL_PARAMS: Final[list[str]] = [
    # 给煤量
    "D62AX002",
    # 二次风机A（调氧量）
    "2LA30A12C11",  # 输出频率
    # 二次风机B
    "2LA40A12C11",  # 输出频率
    # 引风机A（调负压）
    "DPU61AX107",   # 输出频率
    # 引风机B
    "DPU61AX108",   # 输出频率
    # 一次风机A（快速提升负荷）
    "2LA10A12C11",  # 输出频率
    # 一次风机B
    "2LA20A12C11",  # 输出频率
]

# ========== 监测参数（需要关注但不直接调节）==========
MONITOR_PARAMS: Final[list[str]] = [
    "MSFLOW",      # 主蒸汽流量（负荷）
    "D66P53A10",   # 床温
    "D61AX023",    # 一次风风量
    "D61AX024",    # 二次风风量
    "2LA10CT11",   # 出风温1
    "2LA2CT11",    # 出风温2
    "2BK10CP004",
    "2BK10CQ1",    # 含氧量（主目标）
]

# ========== 统计参数（用于窗口统计等特征提取）==========
PARAMS_FOR_STATS: Final[list[str]] = CONTROL_PARAMS + MONITOR_PARAMS

# ========== 专家经验范围 ==========
EXPERT_RANGES: Final[dict] = {
    "pressure_ideal": (-150, -80),  # 炉膛压力理想范围
    "pressure_normal": (-230, -20),  # 炉膛压力正常范围
    "oxygen_ideal": (1.5, 2.5),  # 含氧量理想范围
    "oxygen_target": 2.0,  # 含氧量目标值
    "coal_ideal": 68,  # 给煤量理想值
    "coal_normal": 40,  # 给煤量正常值
}

# ========== 标准化参数（基于实际数据计算）==========
# 用于将物理范围转换为标准化范围
SCALER_PARAMS: Final[dict] = {
    "pressure_mean": -118.90,  # 负压均值
    "pressure_std": 46.24,     # 负压标准差
    "oxygen_mean": 3.17,       # 含氧量均值
    "oxygen_std": 0.88,        # 含氧量标准差
}

# ========== 标准化后的物理范围（用于损失函数约束）==========
NORMALIZED_RANGES: Final[dict] = {
    "pressure_ideal": (-0.67, 0.84),   # 理想范围（标准化后）
    "pressure_normal": (-2.4, 2.14),   # 正常范围（标准化后）
    "oxygen_ideal": (-1.89, -0.75),    # 理想范围（标准化后）
    # 用于模型约束的宽松范围（覆盖大部分正常情况）
    "pressure_constraint": (-3.0, 3.0),  # 标准化后的约束范围
    "oxygen_constraint": (-3.0, 1.0),    # 标准化后的约束范围（含氧量上限较小）
}

# ========== 风机参数配置 ==========
FAN_PARAMS: Final[dict] = {
    "primary_fan": {
        "speed_a": "2LB10CS001",
        "speed_b": "2LB20CS001",
        "current_a": "2BBA14Q11",
        "current_b": "2BBB12Q11",
        "outlet_pressure_a": "2LA10CP01",
        "outlet_pressure_b": "2HLA2CP001",
        "output_a": "2LA10A12C11",
        "output_b": "2LA20A12C11",
        "valve_a": "APAFCDMD",  # A阀门开度
        "valve_b": "BPAFCDMD",  # B阀门开度
        "air_flow": "D61AX023",
    },
    "secondary_fan": {
        "speed_a": "2LB30CS901",
        "speed_b": "2LB40CS901",
        "current_a": "2BBA13Q11",
        "current_b": "2BBB11Q11",
        "outlet_pressure_a": "2HLA30CP01",
        "outlet_pressure_b": "2LA40CP01",
        "output_a": "2LA30A12C11",
        "output_b": "2LA40A12C11",
        "valve_a": "2LA30A11C01",  # A阀门开度
        "valve_b": "2LA40A11C01",  # B阀门开度
        "air_flow": "D61AX024",
    },
    "induced_fan": {
        "speed_a": "2NC10CS901",
        "speed_b": "2NC2CS901",
        "current_a": "2BBA15Q11",
        "current_b": "2BBB13Q11",
        "output_a": "DPU61AX107",
        "output_b": "DPU61AX108",
        "valve_a": "2NC10A11C01",  # A阀门开度
        "valve_b": "2NC20A11C01",  # B阀门开度
        "inlet_pressure_a": "2NA10CP004",  # 引风机A入风压力
        "inlet_pressure_b": "2NA2CP004",  # 引风机B入风压力
    },
}

# ========== 风机专家经验范围 ==========
FAN_EXPERT_RANGES: Final[dict] = {
    "primary_fan_a": {
        "ideal_current": 70,  # 一次风机A理想电流
        "ideal_percent": 31,  # 一次风机A理想控制占比
        "normal_current": 58,  # 一次风机A正常电流
        "normal_percent": 28,  # 一次风机A正常控制占比
    },
    "primary_fan_b": {
        "ideal_current": 77,  # 一次风机B理想电流
        "ideal_percent": 37,  # 一次风机B理想控制占比
        "normal_current": 56,  # 一次风机B正常电流
        "normal_percent": 34,  # 一次风机B正常控制占比
    },
    "secondary_fan_a": {
        "ideal_current": 45,  # 二次风机A理想电流
        "ideal_percent": 41,  # 二次风机A理想控制占比
        "normal_current": 28,  # 二次风机A正常电流
        "normal_percent": 9,  # 二次风机A正常控制占比
    },
    "secondary_fan_b": {
        "ideal_current": 45,  # 二次风机B理想电流
        "ideal_percent": 52,  # 二次风机B理想控制占比
        "normal_current": 27,  # 二次风机B正常电流
        "normal_percent": 26,  # 二次风机B正常控制占比
    },
    "induced_fan_a": {
        "ideal_current": 120,  # 引风机A理想电流
        "ideal_percent": 31,  # 引风机A理想控制占比
        "normal_current": 76,  # 引风机A正常电流
        "normal_percent": 18,  # 引风机A正常控制占比
    },
    "induced_fan_b": {
        "ideal_current": 127,  # 引风机B理想电流
        "ideal_percent": 25,  # 引风机B理想控制占比
        "normal_current": 85,  # 引风机B正常电流
        "normal_percent": 13,  # 引风机B正常控制占比
    },
}

# ========== 风煤比配置 ==========
COAL_AIR_RATIOS: Final[list[tuple]] = [
    (0, 20, 60000, 40000),
    (20, 45, 130000, 75000),
    (45, 68, 160000, 180000),
]

# ========== 默认窗口配置 ==========
DEFAULT_WINDOW_SIZES: Final[list[int]] = [10, 30, 60]
DEFAULT_STATISTICS: Final[list[str]] = [
    "mean",  # 均值
    "std",  # 标准差
    "max",  # 最大值
    "min",  # 最小值
    "median",  # 中位数
    "skew",  # 偏度
    "kurt",  # 峰度
    "cv",  # 系数_of_variation
]
DEFAULT_LAGS: Final[list[int]] = [1, 5, 10, 30]
DEFAULT_DIFF_WINDOWS: Final[list[int]] = [10, 30]

# ========== 多步预测配置 ==========
MULTISTEP_CONFIG: Final[dict] = {
    "default_horizons": [1, 2, 3, 5, 7, 10],
    "max_horizon": 10,
    "min_horizon": 1,
    "prediction_modes": ["direct", "iterative"],
    "default_mode": "direct",
}

# ========== 历史趋势配置 ==========
HISTORY_TREND_CONFIG: Final[dict] = {
    "target_vars": ["2BK10CP004", "2BK10CQ1"],
    "history_lags": [1, 2, 3, 5, 10, 20, 30],
    "trend_windows": [5, 10, 20, 30],
    "prediction_steps": [1, 2, 3, 5, 7, 10],
    "autocorr_lags": [1, 5, 10, 20],
    "momentum_windows": [5, 10, 20],
}

# ========== GPU配置 ==========
GPU_CONFIG: Final[dict] = {
    "xgboost_use_gpu": True,
    "xgboost_device": "cuda",
    "tensorflow_use_gpu": True,
    "tensorflow_memory_growth": True,
    "parallel_training": True,
}

# ========== 特征模式 ==========
SELF_DERIVED_PATTERNS = [
    "_lag_", "_trend_slope_", "_trend_accel_",
    "_mean_", "_std_", "_diff", "_change"
]

CAUSAL_FEATURE_PATTERNS = [
    "coal_air_ratio", "primary_secondary_air_ratio", "total_air_flow",
    "load_change_rate", "id_fan_change_rate", "bed_temp_stability",
    "_in_ideal_range", "coal_load_ratio", "load_coal_ratio",
    "pressure_deviation_", "oxygen_deviation_", "_consistency",
    "id_fan_pressure_gain", "pa_fan_oxygen_gain", "sa_fan_oxygen_gain",
]