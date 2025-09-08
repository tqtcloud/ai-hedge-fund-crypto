# MacdStrategy 价格水平识别功能使用指南

## 功能概述

MacdStrategy类现在包含了强大的价格水平识别功能，可以识别关键的技术分析价位：

- **支撑位** (Support Levels)：基于历史低点和成交量确认
- **阻力位** (Resistance Levels)：基于历史高点和成交量确认  
- **枢轴点** (Pivot Point)：使用标准公式 (H+L+C)/3
- **突破临界点** (Breakout Threshold)：基于ATR波动率计算

## 使用方法

### 基本用法

```python
from strategies.macd_strategy import MacdStrategy
import pandas as pd

# 创建策略实例
strategy = MacdStrategy()

# 准备OHLC数据 (DataFrame必须包含: high, low, close, volume列)
df = pd.DataFrame({
    'high': [...],      # 最高价
    'low': [...],       # 最低价  
    'close': [...],     # 收盘价
    'volume': [...],    # 成交量(可选，用于确认)
})

# 识别价格水平
price_levels = strategy.identify_price_levels(df)
```

### 返回结果格式

```python
{
    'support_levels': [45800.12, 45200.34, 44500.56],    # 3个支撑位(从高到低)
    'resistance_levels': [46500.78, 47100.90, 47800.12], # 3个阻力位(从低到高)
    'pivot_point': 46150.45,                              # 枢轴点
    'breakout_threshold': 230.67                          # 突破临界点
}
```

## 技术算法说明

### 支撑阻力位识别

1. **局部极值检测**：使用10期窗口识别历史高低点
2. **成交量确认**：高成交量的极值点权重更高
3. **移动平均补充**：使用20期和50期MA作为动态支撑阻力
4. **价格百分比**：根据当前价格计算合理的支撑阻力区间

### 枢轴点计算

使用标准枢轴点公式：
```
Pivot Point = (Previous High + Previous Low + Previous Close) / 3
```

### 突破临界点

基于ATR（平均真实波幅）计算：
- 主要使用ATR14的1.5倍作为突破阈值
- 备用方案：价格波动率的2倍
- 限制范围：0.1% - 5%之间

## 数据要求

### 必需列
- `high`: 最高价
- `low`: 最低价  
- `close`: 收盘价

### 可选列
- `volume`: 成交量(用于确认支撑阻力的重要性)

### 数据量要求
- 最少50个数据点才能进行可靠分析
- 数据不足时返回默认值(全部为0.0)

## 逻辑验证

系统会自动验证以下逻辑一致性：

1. **支撑位必须低于当前价格**
2. **阻力位必须高于当前价格**  
3. **支撑位按从高到低排序**
4. **阻力位按从低到高排序**
5. **价位间保持至少0.5%的差距**

## 错误处理

- 空数据或数据不足：返回默认零值
- 缺少必需列：返回默认零值
- 计算异常：打印错误信息并返回默认值

## 使用建议

1. **数据质量**：确保OHLC数据准确且连续
2. **时间框架**：不同时间框架的价位意义不同
3. **结合其他指标**：价格水平应与其他技术指标结合使用
4. **动态更新**：随着新数据加入需要重新计算价位

## 实际应用示例

```python
# 获取价格水平
levels = strategy.identify_price_levels(df)

# 当前价格
current_price = df['close'].iloc[-1]

# 判断价格位置
print(f"当前价格: {current_price:.2f}")
print(f"最近支撑位: {levels['support_levels'][0]:.2f}")
print(f"最近阻力位: {levels['resistance_levels'][0]:.2f}")
print(f"枢轴点: {levels['pivot_point']:.2f}")

# 突破信号
if abs(current_price - levels['resistance_levels'][0]) < levels['breakout_threshold']:
    print("可能突破上方阻力位！")
elif abs(current_price - levels['support_levels'][0]) < levels['breakout_threshold']:
    print("可能跌破下方支撑位！")
```