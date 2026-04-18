# 锅炉机组含氧量和炉膛压力预测项目

## 项目概述

本项目使用机器学习和深度学习方法，基于锅炉机组的历史运行数据，预测未来时间步的含氧量和炉膛压力。

## 项目结构

```
.
├── 建模方案.md              # 完整建模方案文档
├── model_pipeline.py        # 模型训练和评估主程序
├── requirements.txt         # Python依赖包
├── output/                  # 输出目录
│   ├── all_data_cleaned.feather    # 清洗后的原始数据
│   ├── target_variables_analysis.png  # 目标变量分析
│   ├── correlation_analysis.png       # 相关性分析
│   └── lag_correlation_analysis.png   # 滞后相关性分析
└── model_output/            # 模型输出目录（自动生成）
    ├── train_features.feather
    ├── val_features.feather
    ├── test_features.feather
    └── feature_importance_*.csv
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 特征工程

```bash
python model_pipeline.py --stage feature_engineering
```

### 3. 训练模型

**训练含氧量预测模型：**
```bash
python model_pipeline.py --stage train --model lightgbm --target oxygen
```

**训练炉膛压力预测模型：**
```bash
python model_pipeline.py --stage train --model lightgbm --target pressure
```

支持的模型类型：
- `lightgbm`: LightGBM（推荐，速度快，效果好）
- `xgboost`: XGBoost
- `rf`: Random Forest

### 4. 评估模型

```bash
python model_pipeline.py --stage evaluate --model_path models/lightgbm_oxygen.pkl
```

### 5. 一键执行（全部流程）

```bash
python model_pipeline.py --stage all --model lightgbm --target oxygen
```

## 目标变量说明

### 含氧量目标变量
- **主目标**: `2BK10CQ1` - 含氧量测点1
- **备选**: `2BK2CQ1`, `2BK2CQ2`

### 炉膛压力目标变量
- **主目标**: `2BK10CP004` - 炉膛压力测点1
- **备选**: `2BK2CP004`, `2BK10CP005`, `2BK2CP005`

## 关键控制参数

1. **D62AX002**: 给煤量（最重要的调控参数）
2. **DPU61AX107/DPU61AX108**: 引风机频率（调节负压）
3. **2LA30A12C11/2LA40A12C11**: 二次风机频率（调节含氧量）
4. **2LA10A12C11/2LA20A12C11**: 一次风机频率

## 数据说明

- **时间范围**: 2024年10月 - 2026年1月（约15个月）
- **采样频率**: 1分钟
- **样本数量**: 585,939条
- **特征数量**: 52个数值特征

## 建模方案要点

### 推荐方案: 分层集成模型

**阶段1: 特征工程**
- 滞后特征: t-1, t-5, t-15, t-30分钟
- 滑动窗口统计: 均值、标准差（窗口5, 15, 30分钟）
- 时间特征: 小时、星期几、月份、循环编码
- 差分特征: 一阶差分、变化率

**阶段2: 多模型并行**
- 含氧量预测: LightGBM/XGBoost（表格数据效果好）
- 炉膛压力预测: LSTM/Transformer（时序依赖复杂）

**阶段3: 模型融合**
- 简单平均或加权平均
- Stacking集成

### 预测策略

- **短期预测**（5-15分钟）: 递归多步预测
- **中期预测**（30-60分钟）: 直接多步或Seq2Seq

### 评估指标

- **RMSE**: 均方根误差（主要指标）
- **MAE**: 平均绝对误差
- **MAPE**: 平均绝对百分比误差
- **R²**: 决定系数

## 性能目标

- **含氧量预测**: RMSE < 0.5%，MAPE < 5%
- **炉膛压力预测**: RMSE < 20 Pa，MAPE < 15%

## 数据分析结果

### 相关性分析

**含氧量**与以下参数强相关（|r| > 0.3）：
- 主蒸汽流量 (MSFLOW): r = -0.59
- 给煤量 (D62AX002): r = -0.53
- 床温 (D66P53A10): r = -0.51
- 二次风风量 (D61AX024): r = -0.46

**炉膛压力**与所有控制参数相关性均较弱（|r| < 0.08），说明：
- 炉膛压力受随机扰动影响大
- 需要更复杂的时序模型
- 预测难度较高

### 滞后分析

- **含氧量**: 大部分特征即时响应（0分钟滞后）
- **炉膛压力**: 引风机影响有7-13分钟滞后

## 注意事项

1. **数据质量**: 检查缺失值和异常值
2. **特征筛选**: 避免维度灾难，保留Top 20-30特征
3. **过拟合**: 使用时间序列交叉验证
4. **数据漂移**: 定期重训练模型（建议每月）

## 进阶开发

### 深度学习模型

如需使用LSTM或Transformer，请安装TensorFlow/PyTorch：

```bash
pip install tensorflow>=2.10.0
# 或
pip install torch>=1.13.0
```

### 超参数优化

```python
import optuna

# 使用Optuna进行自动调参
# 参考: https://optuna.org/
```

### 模型部署

建议使用以下方式部署：
1. **REST API**: Flask/FastAPI封装模型
2. **定时预测**: Airflow/Cron定时任务
3. **实时预测**: Kafka + 流式处理

## 参考资料

- LightGBM文档: https://lightgbm.readthedocs.io/
- XGBoost文档: https://xgboost.readthedocs.io/
- 时间序列预测指南: https://otexts.com/fpp3/

## 联系与支持

如有问题，请参考建模方案.md获取详细技术细节。
