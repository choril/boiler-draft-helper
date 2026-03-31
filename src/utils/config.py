from typing import Final

PRESSURE_VARIABLES: Final[list[str]] = [
    "2BK10CP004",  # 炉膛压力1 (主目标)
    "2BK2CP004",  # 炉膛压力2
    "2BK10CP005",  # 炉膛压力3
    "2BK2CP005",  # 炉膛压力4
]

OXYGEN_VARIABLES: Final[list[str]] = [
    "2BK10CQ1",  # 含氧量1 (主目标)
    "2BK2CQ1",  # 含氧量2
    "2BK2CQ2",  # 含氧量3
]

PRESSURE_MAIN: Final[str] = "2BK10CP004"
OXYGEN_MAIN: Final[str] = "2BK10CQ1"

TARGET_VARIABLES: Final[list[str]] = PRESSURE_VARIABLES + OXYGEN_VARIABLES
MAIN_TARGETS: Final[list[str]] = [PRESSURE_MAIN, OXYGEN_MAIN]

EXCLUDE_VARIABLES: Final[list[str]] = ["TIME", "source_file"] + TARGET_VARIABLES

KEY_PARAMS: Final[list[str]] = [
    "D62AX002",  # 给煤量
    "D66P53A10",  # 床温
    "D61AX023",  # 一次风风量
    "D61AX024",  # 二次风风量
    "2BK10CP004",  # 炉膛压力
    "2BK10CQ1",  # 含氧量1
    "2NC10CS901",  # 引风机A转速
    "2NC2CS901",  # 引风机B转速
    "2LB10CS001",  # 一次风机A转速
    "2LB20CS001",  # 一次风机B转速
    "2LB30CS901",  # 二次风机A转速
    "2LB40CS901",  # 二次风机B转速
    "MSFLOW",  # 主蒸汽
    "2LA10CT11",  # 出风温1
    "2LA2CT11",  # 出风温2
    "2BBA14Q11",  # 一次风机A电流
    "2BBB12Q11",  # 一次风机B电流
    "2BBA13Q11",  # 二次风机A电流
    "2BBB11Q11",  # 二次风机B电流
    "2BBA15Q11",  # 引风机A电流
    "2BBB13Q11",  # 引风机B电流
    "2LA10A12C11",  # 一次风机A输出
    "2LA20A12C11",  # 一次风机B输出
    "2LA30A12C11",  # 二次风机A输出
    "2LA40A12C11",  # 二次风机B输出
    "DPU61AX107",  # 引风机A输出
    "DPU61AX108",  # 引风机B输出
]

EXPERT_RANGES: Final[dict] = {
    "pressure_ideal": (-150, -80),  # 炉膛压力理想范围
    "pressure_normal": (-230, -20),  # 炉膛压力正常范围
    "oxygen_ideal": (1.7, 2.3),  # 含氧量理想范围
    "oxygen_target": 2.0,  # 含氧量目标值
    "coal_ideal": 68,  # 给煤量理想值
    "coal_normal": 40,  # 给煤量正常值
}

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
        "air_flow": "D61AX024",
    },
    "induced_fan": {
        "speed_a": "2NC10CS901",
        "speed_b": "2NC2CS901",
        "current_a": "2BBA15Q11",
        "current_b": "2BBB13Q11",
        "output_a": "DPU61AX107",
        "output_b": "DPU61AX108",
        "inlet_pressure_a": "2NA10CP004",  # 引风机A入风压力
        "inlet_pressure_b": "2NA2CP004",  # 引风机B入风压力
    },
}

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

COAL_AIR_RATIOS: Final[list[tuple]] = [
    (0, 20, 60000, 40000),
    (20, 45, 130000, 75000),
    (45, 68, 160000, 180000),
]

CONTROL_PARAMS: Final[list[str]] = [
    "2LB10CS001",  # 一次风机A转速
    "APAFCDMD",    # 一次风机A阀门开度
    "2LB20CS001",  # 一次风机B转速
    "BPAFCDMD",    # 一次风机B阀门开度
    "2LB30CS901",  # 二次风机A转速
    "2LA30A11C01",  # 二次风机A阀门开度
    "2LB40CS901",  # 二次风机B转速
    "2LA40A11C01",  # 二次风机B阀门开度
    "2NC10CS901",  # 引风机A转速
    "2NC10A11C01",  # 引风机A阀开度
    "2NC2CS901",  # 引风机B转速
    "2NC20A11C01",  # 引风机B阀门开度
    "D62AX002",  # 给煤量
]

KEY_PARAMS_FOR_STATS: Final[list[str]] = [
    "D62AX002",  # 给煤量
    "D66P53A10",  # 含氧量1
    "D61AX023",  # 一次风风量
    "D61AX024",  # 二次风风量
    "2LA10CT11",  # 出风温1
    "2LA2CT11",  # 出风温2
    "2LB30CS901",  # 二次风机A转速
    "2BBA13Q11",  # 二次风机A电流
    "2LA30A12C11",  # 二次风机A输出
    "2LB40CS901",  # 二次风机B转速
    "2BBB11Q11",  # 二次风机B电流
    "2LA40A12C11",  # 二次风机B输出
    "2LB10CS001",  # 一次风机A转速
    "2BBA14Q11",  # 一次风机A电流
    "2LA10A12C11",  # 一次风机A输出
    "2LB20CS001",  # 一次风机B转速
    "2BBB12Q11",  # 一次风机B电流
    "2LA20A12C11",  # 一次风机B输出
    "2NC10CS901",  # 引风机A转速
    "2BBA15Q11",  # 引风机A电流
    "DPU61AX107",  # 引风机A输出
    "2NC2CS901",  # 引风机B转速
    "2BBB13Q11",  # 引风机B电流
    "DPU61AX108",  # 引风机B输出
]

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

MULTISTEP_CONFIG: Final[dict] = {
    "default_horizons": [1, 2, 3, 5, 7, 10],
    "max_horizon": 10,
    "min_horizon": 1,
    "prediction_modes": ["direct", "iterative"],
    "default_mode": "direct",
}

HISTORY_TREND_CONFIG: Final[dict] = {
    "target_vars": ["2BK10CP004", "2BK10CQ1"],
    "history_lags": [1, 2, 3, 5, 10, 20, 30],
    "trend_windows": [5, 10, 20, 30],
    "prediction_steps": [1, 2, 3, 5, 7, 10],
    "autocorr_lags": [1, 5, 10, 20],
    "momentum_windows": [5, 10, 20],
}

GPU_CONFIG: Final[dict] = {
    "xgboost_use_gpu": True,
    "xgboost_device": "cuda",
    "tensorflow_use_gpu": True,
    "tensorflow_memory_growth": True,
    "parallel_training": True,
}


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