# 跨时间框架分析实现文档

## 概述

在MacdStrategy类中成功实现了`cross_timeframe_analysis`方法，该方法用于分析多个时间框架的信号一致性，识别主导时间框架，检测冲突区域，评估趋势对齐情况，并综合评估整体信号强度。

## 方法签名

```python
def cross_timeframe_analysis(self, timeframe_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Any]
```

## 输入参数

- `timeframe_signals`: 包含多个时间框架信号数据的字典
  ```python
  {
    "5m": {"signal": "bullish", "confidence": 75, "strategy_signals": {...}},
    "15m": {"signal": "bearish", "confidence": 60, "strategy_signals": {...}},
    "30m": {"signal": "bullish", "confidence": 80, "strategy_signals": {...}},
    "1h": {"signal": "neutral", "confidence": 50, "strategy_signals": {...}},
    "4h": {"signal": "bullish", "confidence": 85, "strategy_signals": {...}}
  }
  ```

## 输出格式

```python
{
  "timeframe_consensus": float,        # 时间框架一致性 0-1
  "dominant_timeframe": "5m|15m|30m|1h|4h",
  "conflict_areas": ["timeframe_pairs"],
  "trend_alignment": "aligned|divergent|mixed",
  "overall_signal_strength": "weak|moderate|strong"
}
```

## 核心算法逻辑

### 1. 时间框架一致性 (timeframe_consensus)

**算法要点：**
- 将信号转换为数值：bullish(+1), neutral(0), bearish(-1)
- 使用时间框架权重：5m(0.1) < 15m(0.15) < 30m(0.2) < 1h(0.25) < 4h(0.3)
- 计算加权方差来衡量信号分散程度
- 综合考虑信号强度因子和置信度因子

**公式：**
```
一致性 = 基础一致性 × (0.6 + 0.2×信号强度因子 + 0.2×置信度因子)
基础一致性 = 1.0 - 加权方差/4.0
```

### 2. 主导时间框架 (dominant_timeframe)

**评分因子：**
- 基础评分：置信度 × 时间框架权重
- 信号强度加成：非中性信号 +0.2
- 策略一致性加成：基于strategy_signals的内部一致性
- 长期加成：4h/1h(+0.15), 30m(+0.1)  
- 高置信度加成：≥80%(+0.2), ≥60%(+0.1)

### 3. 冲突区域识别 (conflict_areas)

**冲突条件：**
- 信号方向完全相反（bullish vs bearish）
- 至少一个信号的置信度 ≥ 50%

**输出格式：**
按时间框架优先级排序，如`["4h_vs_15m", "30m_vs_15m"]`

### 4. 趋势对齐评估 (trend_alignment)

**分级标准：**
- `aligned`: 高一致性(≥0.75)且主导方向占比≥70%
- `divergent`: 低一致性(<0.4)且多空严重分歧
- `mixed`: 介于两者之间的混合状态

### 5. 整体信号强度 (overall_signal_strength)

**评分构成：**
- 一致性贡献：consensus_score × 0.4
- 趋势对齐贡献：aligned(0.25), mixed(0.15), divergent(0.05)
- 冲突惩罚：每个冲突 -0.05，最多 -0.2
- 置信度贡献：平均置信度 × 0.2

**分级标准：**
- `strong`: 综合评分 ≥ 0.6
- `moderate`: 综合评分 0.35-0.6
- `weak`: 综合评分 < 0.35

## 测试结果示例

**输入：**
- 5m: bullish(75%), 15m: bearish(60%), 30m: bullish(80%), 1h: neutral(50%), 4h: bullish(85%)

**输出：**
```json
{
  "timeframe_consensus": 0.7804,
  "dominant_timeframe": "4h", 
  "conflict_areas": ["15m_vs_5m", "30m_vs_15m", "4h_vs_15m"],
  "trend_alignment": "aligned",
  "overall_signal_strength": "moderate"
}
```

## 边界情况处理

1. **空输入**：返回默认值（一致性0.5，主导时间框架1h等）
2. **单时间框架**：返回默认值，避免除零错误
3. **全中性信号**：正常计算，体现中性状态的一致性
4. **强冲突信号**：准确识别所有冲突对，降低整体信号强度

## 集成方式

该方法已集成到MacdStrategy的`__call__`方法中，在所有单个时间框架分析完成后，对每个ticker执行跨时间框架分析，结果添加到`technical_analysis[ticker]["cross_timeframe_analysis"]`字段中。

## 性能特点

- **鲁棒性强**：完整的异常处理和边界情况处理
- **算法透明**：每个评分因子都有明确的计算逻辑和权重
- **结果可解释**：输出结果能够清晰反映多时间框架信号的特征
- **计算高效**：使用NumPy进行向量化计算，性能优良

## 符合规范

实现完全符合`futures_trading_output_specification.md`中定义的输出格式要求，为后续的风险管理和投资组合管理Agent提供可靠的多时间框架信号分析基础。