# 多时间框架分析功能文档

## 概述

`PortfolioManagementNode.analyze_timeframes` 方法实现了多时间框架信号一致性分析功能，根据 `futures_trading_output_specification.md` 的规范要求，提供完整的时间框架分析结果。

## 功能特性

### 1. 核心功能
- **信号一致性分析**: 分析5分钟、15分钟、30分钟、1小时和4小时时间框架的信号一致性
- **权重计算**: 基于时间框架重要性的动态权重分配
- **共识评分**: 0-1范围的量化共识指标
- **主导时间框架识别**: 自动识别最具影响力的时间框架
- **冲突信号检测**: 识别和量化信号冲突程度
- **方向置信度**: 综合评估整体交易方向的可信程度

### 2. 时间框架权重策略

```python
timeframe_priorities = {
    "5m": 1.0,    # 短期噪音，权重最低
    "15m": 1.5,   # 短期趋势
    "30m": 2.0,   # 中短期趋势  
    "1h": 3.0,    # 中期趋势
    "4h": 4.0     # 长期趋势，权重最高
}
```

## 方法签名

```python
def analyze_timeframes(
    self,
    ticker: str,
    analyst_signals: Dict[str, Any],
    portfolio: Dict[str, Any]
) -> Dict[str, Any]:
```

### 参数说明

- **ticker**: 交易对符号（如"BTCUSDT"）
- **analyst_signals**: 技术分析师信号数据，包含各时间框架的信号和置信度
- **portfolio**: 投资组合数据（当前实现中未直接使用，为扩展性保留）

### 返回值结构

```python
{
    "consensus_score": float,           # 共识评分 0-1
    "dominant_timeframe": str,          # 主导时间框架 "5m|15m|30m|1h|4h"
    "signal_alignment": str,            # 信号对齐强度 "strong|moderate|weak"
    "conflicting_signals": int,         # 冲突信号数量
    "timeframe_weights": {              # 各时间框架标准化权重
        "5m": float,
        "15m": float, 
        "30m": float,
        "1h": float,
        "4h": float
    },
    "overall_direction_confidence": float # 整体方向置信度 0-1
}
```

## 核心算法

### 1. 共识评分计算

```python
# 基于信号方向标准差的一致性评分
if len(signals_list) > 1:
    avg_signal = sum(signals_list) / len(signals_list)
    variance = sum((s - avg_signal) ** 2 for s in signals_list) / len(signals_list)
    std_deviation = variance ** 0.5
    consensus_score = max(0.0, 1.0 - std_deviation / 2.0)
else:
    consensus_score = confidences_list[0]  # 单一信号时使用置信度
```

### 2. 主导时间框架识别

```python
# 权重最高且有明确信号的时间框架
for timeframe, data in timeframe_signals.items():
    if data["signal"] != "neutral":
        weighted_strength = abs(data["weighted_signal"])
        if weighted_strength > max_weighted_signal:
            max_weighted_signal = weighted_strength
            dominant_timeframe = timeframe
```

### 3. 信号对齐强度评估

```python
# 基于共识评分和冲突信号数量的三级分类
if consensus_score >= 0.7 and conflicting_signals <= 1:
    signal_alignment = "strong"
elif consensus_score >= 0.5 and conflicting_signals <= 2:
    signal_alignment = "moderate"
else:
    signal_alignment = "weak"
```

## 测试场景和结果

### 场景1: 强烈看多信号（一致性高）
- **输入**: 所有时间框架都显示bullish信号，置信度75-92%
- **结果**: 
  - 共识评分: 0.955
  - 信号对齐: strong
  - 冲突信号: 0

### 场景2: 冲突信号（多空混合）
- **输入**: 短期bearish，长期bullish，存在明显冲突
- **结果**: 
  - 共识评分: 0.5
  - 信号对齐: moderate  
  - 冲突信号: 2

### 场景3: 弱信号（大部分中性）
- **输入**: 多数时间框架为neutral，仅30m显示bullish
- **结果**: 
  - 共识评分: 0.55
  - 信号对齐: moderate
  - 主导时间框架: 30m

## 集成到交易决策

该方法已集成到 `PortfolioManagementNode.__call__` 中，作为 `timeframe_analysis` 字段输出：

```python
# 在__call__方法中的使用
timeframe_analysis = self.analyze_timeframes(
    ticker=ticker,
    analyst_signals=analyst_signals,
    portfolio=portfolio
)
enhanced_decision["timeframe_analysis"] = timeframe_analysis
```

## 扩展性考虑

### 1. 支持cross_timeframe_analysis整合
- 方法可以读取现有的 `cross_timeframe_analysis` 数据
- 通过加权平均方式融合计算结果和现有分析
- 保持向后兼容性

### 2. 容错处理
- 数据缺失时提供合理默认值
- 异常处理确保系统稳定性
- 渐进式降级策略

### 3. 性能优化
- 高效的数值计算
- 最小化内存分配
- 缓存友好的数据结构

## 使用建议

### 1. 强信号场景（signal_alignment="strong"）
- 可以使用较高的仓位比例
- 适合趋势跟踪策略
- 延长持仓时间

### 2. 中等信号场景（signal_alignment="moderate"）
- 使用中等仓位比例
- 设置较紧的止损
- 密切监控信号变化

### 3. 弱信号场景（signal_alignment="weak"）
- 降低仓位规模
- 考虑观望策略
- 等待更明确的信号

## 未来优化方向

1. **机器学习集成**: 使用历史数据训练权重优化模型
2. **动态权重调整**: 根据市场条件动态调整时间框架权重
3. **信号质量评估**: 加入信号历史表现的质量评分
4. **实时更新**: 支持流式数据的实时分析更新