# 动态风险指标计算功能使用说明

## 功能概述

在`RiskManagementNode`类中新增了`calculate_dynamic_risk_metrics`方法，用于基于实际历史数据和当前持仓计算动态风险指标。

## 主要风险指标

### 1. VaR (风险价值)
- **var_1day**: 1日风险价值，表示在95%置信度下，1天内可能的最大损失
- **var_7day**: 7日风险价值，表示在95%置信度下，7天内可能的最大损失

### 2. Expected Shortfall (期望损失)
- **expected_shortfall**: 在超过VaR阈值的情况下，预期的平均损失

### 3. Maximum Drawdown (最大回撤)
- **maximum_drawdown**: 基于历史价格数据计算的最大回撤百分比

### 4. Sharpe Ratio Impact (夏普比率影响)
- **sharpe_ratio_impact**: 该资产对投资组合夏普比率的影响评分(-1到1)

### 5. Risk Adjusted Return (风险调整收益)
- **risk_adjusted_return**: 考虑最大回撤和VaR后的风险调整收益率

## 输出格式

```json
{
  "BTCUSDT": {
    "dynamic_risk_metrics": {
      "var_1day": -0.033575,          // 1日VaR (负数表示潜在损失)
      "var_7day": -0.088830,          // 7日VaR
      "expected_shortfall": -0.039973, // 期望损失
      "maximum_drawdown": 0.255068,    // 最大回撤 (正数)
      "sharpe_ratio_impact": -0.348077, // 夏普比率影响
      "risk_adjusted_return": -0.234324 // 风险调整收益
    }
  }
}
```

## 技术实现特点

### 1. 历史模拟法计算VaR
- 使用历史收益率分布的分位数来估算风险
- 通过平方根法则调整不同时间跨度的VaR

### 2. 期望损失计算
- 计算超过VaR阈值的损失的平均值
- 提供比VaR更全面的尾部风险信息

### 3. 动态最大回撤
- 基于历史价格数据的滚动最高点计算回撤
- 反映真实的历史极端损失情况

### 4. 投资组合影响评估
- 考虑当前持仓比例对整个投资组合的影响
- 结合夏普比率进行风险调整收益评估

## 集成到现有流程

该功能已自动集成到`RiskManagementNode`的主流程中：

```python
# 在__call__方法中自动调用
dynamic_risk_analysis = self.calculate_dynamic_risk_metrics(state)

# 结果整合到风险分析输出中
for ticker in tickers:
    if ticker in dynamic_risk_analysis:
        risk_analysis[ticker].update(dynamic_risk_analysis[ticker])
```

## 使用注意事项

1. **数据要求**: 需要至少30个交易日的历史数据才能进行有效计算
2. **默认值**: 当数据不足时，系统会返回合理的默认风险指标
3. **实时更新**: 风险指标会随着新的市场数据动态更新
4. **与其他模块协同**: 可与技术分析数据（如ATR、波动率等）配合使用

## 应用场景

1. **仓位管理**: 基于VaR限制单个资产的仓位规模
2. **风险预警**: 当期望损失超过阈值时触发预警
3. **资产筛选**: 优先选择风险调整收益较高的资产
4. **投资组合优化**: 平衡不同资产的风险贡献度