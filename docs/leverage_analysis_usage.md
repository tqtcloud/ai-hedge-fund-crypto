# 杠杆分析功能使用说明

## 概述

RiskManagementNode类新增了`analyze_leverage`方法，用于基于实际市场波动率和当前资金状况分析杠杆倍数建议。该功能为合约交易提供智能化的杠杆建议。

## 功能特性

### 核心分析指标

- **ATR（Average True Range）分析**：基于14期和28期ATR计算市场波动性
- **波动率深度分析**：评估当前市场波动率在历史数据中的百分位位置
- **跨时间框架共识**：结合多个时间框架的技术分析信号
- **资金管理风险**：基于投资组合价值和可用现金进行风险评估

### 输出格式

```json
{
  "leverage_analysis": {
    "recommended_leverage": 5,        // 推荐杠杆倍数（1-125x）
    "max_safe_leverage": 8,          // 最大安全杠杆倍数
    "leverage_options": [3, 5, 6],   // 可选杠杆倍数（保守、推荐、激进）
    "leverage_risk_score": 0.65,     // 杠杆风险评分（0-1，越高风险越大）
    "volatility_adjusted_leverage": 4 // 波动率调整后杠杆倍数
  },
  "current_price": 50000.0,          // 当前价格
  "reasoning": {                     // 分析依据
    "atr_14": 1200.5,               // 14期ATR值
    "atr_percentile": 65.0,         // ATR历史百分位
    "volatility_percentile": 70.0,   // 波动率历史百分位
    "volatility_trend": "increasing", // 波动率趋势
    "timeframe_consensus": 0.7,      // 时间框架共识度
    "portfolio_value": 15000.0,      // 投资组合总价值
    "available_cash": 10000.0        // 可用现金
  }
}
```

## 风险评估逻辑

### 基础风险评分计算

风险评分基于以下四个维度：

1. **ATR百分位风险**（权重30%）：当前ATR在历史数据中的位置
2. **波动率百分位风险**（权重30%）：当前波动率的历史排名
3. **波动率趋势风险**：
   - `increasing`：+0.3（高风险）
   - `stable`：+0.1（中等风险）
   - `decreasing`：+0.0（低风险）
4. **时间框架共识风险**（权重10%）：信号一致性，共识度越低风险越高

### 杠杆倍数计算

#### 最大安全杠杆

```python
base_max = max(1, int(125 * (1.0 - risk_score)))  # 基础最大杠杆
cash_adjusted = max(1, int(base_max * cash_factor))  # 现金调整
max_safe_leverage = min(125, max(1, cash_adjusted))  # 最终结果
```

#### 推荐杠杆

```python
recommended_leverage = int(max_safe_leverage * 0.7)  # 保守系数70%
```

#### 波动率调整杠杆

```python
vol_adjustment = 1.0 - (vol_percentile / 100.0) * 0.3  # 波动率调整
atr_adjustment = 1.0 - (atr_percentile / 100.0) * 0.2  # ATR调整
volatility_adjusted_leverage = int(base_leverage * min(vol_adjustment, atr_adjustment))
```

## 使用场景示例

### 场景1：正常市场条件
- **投资组合价值**：$15,000
- **可用现金**：$10,000
- **ATR百分位**：65%
- **波动率百分位**：70%
- **波动率趋势**：increasing

**输出结果**：
- 推荐杠杆：3x
- 最大安全杠杆：5x
- 风险评分：0.935（高风险）

### 场景2：高现金充足情况
- **投资组合价值**：$60,000
- **可用现金**：$50,000
- **其他条件同上**

**输出结果**：
- 推荐杠杆：11x
- 最大安全杠杆：16x
- 风险评分：0.835（较高风险）

### 场景3：极端高波动率
- **ATR百分位**：95%
- **波动率百分位**：98%
- **波动率趋势**：increasing

**输出结果**：
- 推荐杠杆：1x（最保守）
- 风险评分：1.0（极高风险）

## 集成到交易系统

### 在RiskManagementNode中的使用

```python
# 在__call__方法中自动执行杠杆分析
def __call__(self, state: AgentState) -> Dict[str, Any]:
    # ... 现有的风险管理逻辑 ...
    
    # 添加杠杆分析功能
    leverage_analysis = self.analyze_leverage(state)
    
    # 将结果整合到风险分析中
    for ticker in tickers:
        if ticker in leverage_analysis:
            risk_analysis[ticker].update(leverage_analysis[ticker])
    
    # ... 返回结果 ...
```

### 获取杠杆建议

```python
# 从agent状态中获取杠杆分析结果
risk_signals = state["data"]["analyst_signals"]["risk_management_agent"]

for ticker, analysis in risk_signals.items():
    leverage_data = analysis.get("leverage_analysis", {})
    
    recommended = leverage_data.get("recommended_leverage", 1)
    max_safe = leverage_data.get("max_safe_leverage", 1)
    risk_score = leverage_data.get("leverage_risk_score", 1.0)
    
    print(f"{ticker}推荐杠杆：{recommended}x（最大：{max_safe}x，风险：{risk_score:.3f}）")
```

## 注意事项

1. **杠杆范围限制**：所有杠杆倍数都限制在1-125倍范围内
2. **保守原则**：系统采用保守策略，推荐杠杆通常为最大安全杠杆的70%
3. **动态调整**：杠杆建议会根据实时市场波动率和资金状况动态调整
4. **风险提示**：高风险评分（>0.8）时系统会自动降低杠杆建议
5. **现金管理**：可用现金不足时系统会显著降低杠杆倍数

## 测试验证

项目提供了完整的测试脚本：

```bash
# 运行独立测试脚本
python3 src/test/test_leverage_analysis_standalone.py

# 测试覆盖以下场景：
# 1. 不同资金状况（正常、高现金、低现金、无持仓）
# 2. 极端市场条件（高波动率、空数据）
# 3. 约束验证（杠杆范围、风险评分等）
```

## 更新记录

- **v1.0**（2024-01-01）：实现基础杠杆分析功能
- 支持基于ATR和波动率的风险评估
- 提供三种杠杆选项（保守、推荐、激进）
- 集成到RiskManagementNode的主要工作流程中