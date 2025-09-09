# Decision Metadata 实现说明

## 概述

本文档说明在 `PortfolioManagementNode` 类中实现的 `create_decision_metadata` 方法，该方法用于记录LLM实际推理过程的完整元数据，符合 `docs/futures_trading_output_specification.md` 文档规范。

## 实现功能

### 核心方法

#### `create_decision_metadata`
- **位置**: `src/graph/portfolio_management_node.py`
- **功能**: 创建决策元数据，记录LLM实际推理过程的完整元数据
- **返回**: 符合规范要求的 `decision_metadata` 字段字典

### 字段结构

实现的 `decision_metadata` 包含以下完整字段：

```json
{
  "confidence": int,                   // 总体置信度 0-100
  "confidence_breakdown": {
    "technical_analysis": float,      // 技术分析置信度
    "risk_assessment": float,         // 风险评估置信度
    "market_conditions": float,       // 市场条件置信度
    "cost_benefit": float,            // 成本效益置信度
    "execution_feasibility": float    // 执行可行性置信度
  },
  "decision_factors": {
    "primary_drivers": [string, string, string],      // 主要驱动因子 (最多3个)
    "supporting_factors": [string, string],           // 支持因子 (最多2个)
    "risk_factors": [string, string],                 // 风险因子 (最多2个)
    "uncertainty_factors": [string, string]           // 不确定性因子 (最多2个)
  },
  "alternative_scenarios": [
    {
      "condition": string,           // 条件描述
      "alternative_action": string,  // 替代行动
      "probability": float,          // 概率 0-1
      "impact": "low|moderate|high"  // 影响程度
    }
  ],
  "decision_tree_path": [string, string, string],     // 决策树路径 (最多3个)
  "reasoning_chain": [string, string, string],        // 推理链条 (最多3个)
  "supporting_evidence": [string, string],            // 支持证据 (最多2个)
  "contrary_evidence": [string, string]               // 反对证据 (最多2个)
}
```

## 实现特点

### 1. 智能置信度计算
- **技术分析置信度**: 基于多时间框架信号强度和一致性
- **风险评估置信度**: 基于风险控制措施的完整性和合理性
- **市场条件置信度**: 基于市场环境的清晰度和稳定性
- **成本效益置信度**: 基于预期收益和成本的合理性
- **执行可行性置信度**: 基于流动性和执行复杂度

### 2. 决策因子提取
- **主要驱动因子**: 从趋势状态、收益潜力、风险收益比等方面提取
- **支持因子**: 从波动率、流动性等方面提取
- **风险因子**: 从波动率风险、保证金压力等方面提取
- **不确定性因子**: 从趋势转换、资金费率等方面提取

### 3. 替代场景生成
- **趋势反转场景**: 基于市场趋势状态
- **波动率放大场景**: 基于当前波动率水平
- **流动性枯竭场景**: 基于流动性评估
- **技术突破失败场景**: 基于突破概率

### 4. 决策树路径构建
- 市场环境评估 → 技术信号确认 → 风险评估 → 最终决策

### 5. 推理链条生成
- 市场分析 → 技术确认 → 风险控制，形成完整推理过程

### 6. 证据收集
- **支持证据**: 从趋势明确性、收益吸引力、获利概率等方面收集
- **反对证据**: 从波动率风险、流动性问题、资金费率等方面收集

## 集成方式

### 调用流程
在 `PortfolioManagementNode.__call__` 方法中，`create_decision_metadata` 在所有其他分析完成后被调用：

1. `calculate_basic_params` - 基础交易参数
2. `design_risk_management` - 风险管理参数
3. `analyze_timeframes` - 时间框架分析
4. `assess_technical_risk` - 技术风险评估
5. `calculate_cost_benefit` - 成本效益分析
6. `evaluate_market_environment` - 市场环境评估
7. `design_execution_strategy` - 执行策略建议
8. `generate_scenario_analysis` - 情景分析
9. **`create_decision_metadata`** - **决策元数据生成** ✨

### 参数传递
方法接收前面所有分析模块的输出结果作为输入参数，确保决策元数据基于完整的分析结果。

## 验证结果

通过测试验证，实现的 `decision_metadata` 结构完全符合规范要求：

- ✅ 所有必需字段都已包含
- ✅ 数据类型符合规范
- ✅ 数值范围符合要求
- ✅ 列表长度符合限制
- ✅ 置信度计算逻辑合理
- ✅ 决策因子提取智能化
- ✅ 替代场景生成完整

## 示例输出

```json
{
  "confidence": 82,
  "confidence_breakdown": {
    "technical_analysis": 88.0,
    "risk_assessment": 75.0,
    "market_conditions": 95.0,
    "cost_benefit": 73.6,
    "execution_feasibility": 70.0
  },
  "decision_factors": {
    "primary_drivers": ["市场处于trending趋势状态，支持long方向"],
    "supporting_factors": ["波动率处于正常水平，利于交易执行", "市场流动性good，执行成本较低"],
    "risk_factors": ["市场波动风险"],
    "uncertainty_factors": ["价格走势不确定性"]
  },
  "alternative_scenarios": [
    {
      "condition": "市场趋势发生反转，技术指标背离",
      "alternative_action": "平仓止损或减仓观望",
      "probability": 0.2,
      "impact": "moderate"
    }
  ],
  "decision_tree_path": [
    "市场环境评估: trending",
    "技术信号强度: moderate", 
    "风险评估: 需谨慎"
  ],
  "reasoning_chain": [
    "市场分析显示趋势为trending，波动率normal，支持long操作",
    "技术分析确认long信号，多时间框架支持交易方向",
    "设置5倍杠杆，风险收益比1.33，止损位63500.0"
  ],
  "supporting_evidence": [
    "市场趋势明确，支持long方向交易",
    "预期收益率3.00%具有吸引力"
  ],
  "contrary_evidence": [
    "市场不确定性带来潜在风险"
  ]
}
```

## 总结

`create_decision_metadata` 方法的实现为AI量化交易系统提供了完整的决策追踪功能，确保每个交易决策都有详细的元数据记录，便于后续分析、优化和风险管理。实现完全符合 `futures_trading_output_specification.md` 的规范要求，并与现有代码风格保持一致。